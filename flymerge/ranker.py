"""Learn relative candidate utilities from shared-future rollout groups."""
import argparse
import json
from pathlib import Path
import time

from torch import nn

from .afterstate_data import candidate_boards
from .afterstate_model import restore_scorer
from .policy import np, torch
from .ranking_data import DATA, ROOT
from .replay import json_file
from .rollout_teacher import shortlist

SCALE = 100.
PAIRS = torch.triu_indices(6,6,offset=1)


class Ranker:
    def __init__(self,kind,graph):
        self.scorer = restore_scorer(Path('artifacts/afterstate')/f'{kind}.pt',graph)
        self.adjust = nn.Linear(2,1)
        nn.init.zeros_(self.adjust.weight)
        nn.init.zeros_(self.adjust.bias)
        self.kind = kind

    def estimate(self,features,side):
        shape = features.shape
        raw = self.scorer.model(features.reshape(-1,shape[-1])).reshape(*shape[:-1],2)
        score = raw[...,0] + self.adjust(side).squeeze(-1)
        return score,raw[...,1]

    @torch.no_grad()
    def choose(self,obs,policy,models,nuisance,trace=False):
        baseline, actions = shortlist(policy,models,obs,nuisance)
        views,points,ended = candidate_boards(obs,np.array(actions,dtype=np.int64))
        x = torch.from_numpy(views)
        side = torch.from_numpy(np.column_stack((points/SCALE,ended)).astype(np.float32))
        scores,risks = self.estimate(self.scorer.features(x),side)
        position = int(np.lexsort((actions,-scores.numpy()))[0])
        details = dict(baseline=baseline,actions=actions,immediate=points.tolist(),scores=scores.tolist(),
                       predicted_risks=risks.sigmoid().tolist(),board=views[position].tolist())
        if trace and self.kind=='flywire':
            reservoir = self.scorer.reservoir
            chosen = x[position:position+1]
            _, direct, states = reservoir.circuit(chosen,trace=True)
            _, _, zero = reservoir.circuit(torch.zeros_like(chosen),trace=True)
            pack = lambda a: np.round(a[0].numpy().astype(float),5).tolist()
            details.update(direct=pack(direct),steps=[
                dict(recurrent=pack(rec),activation=pack(value),reference=pack(ref),delta=pack(value-ref))
                for (rec,value),(_,ref) in zip(states,zero)])
        return actions[position],details


def contrast(rollouts):
    difference = rollouts[:,PAIRS[0]]-rollouts[:,PAIRS[1]]
    delta = difference.mean(-1)
    error = difference.std(-1,unbiased=False)/(difference.shape[-1]**.5)
    weight = delta.abs()/(delta.abs()+error+.1)
    return (delta>0).float(),weight


def objective(score,risk,rollouts,outcomes):
    labels,weights = contrast(rollouts)
    gap = score[:,PAIRS[0]]-score[:,PAIRS[1]]
    pair = nn.functional.binary_cross_entropy_with_logits(gap,labels,reduction='none')
    rank = (weights*pair).sum()/weights.sum().clamp_min(1.)
    regression = nn.functional.smooth_l1_loss(score,rollouts.mean(-1))
    survival = nn.functional.binary_cross_entropy_with_logits(risk,outcomes.mean(-1))
    return rank+.25*regression+.1*survival


def accuracy(score,rollouts,actions,baseline):
    target = rollouts.mean(-1)
    labels,w = contrast(rollouts)
    comparison = score[:,PAIRS[0]]>score[:,PAIRS[1]]
    wanted = labels.bool()
    nonzero = w>0
    top = np.lexsort((actions,-score.numpy()),axis=-1)[:,0]
    best = np.lexsort((actions,-target.numpy()),axis=-1)[:,0]
    observed = target[torch.arange(len(score)),torch.from_numpy(top)]
    reference = target.max(1).values
    baseline_position = np.argmax(actions == baseline[:,None],axis=1)
    baseline_observed = target[torch.arange(len(score)),torch.from_numpy(baseline_position)]
    return dict(groups=len(score),pair_accuracy=float((comparison==wanted)[nonzero].float().mean()),
                weighted_accuracy=float(((comparison==wanted)*w).sum()/w.sum()),
                winner_match=float(np.mean(top==best)),
                mean_regret_points=float((reference-observed).mean())*SCALE,
                difference_vs_robust_points=float((observed-baseline_observed).mean())*SCALE)


def arrays(kind,archive,model,board_key):
    board = torch.from_numpy(archive[board_key])
    shape = board.shape
    with torch.no_grad():
        embed = torch.cat([model.scorer.features(x) for x in board.flatten(0,1).split(512)])
    return embed.reshape(*shape[:-1],-1)


def train(kind,epochs):
    started = time.perf_counter()
    graph = json.loads(Path('artifacts/graph.json').read_text())
    torch.manual_seed(59913)
    rng = np.random.default_rng(59913)
    model = Ranker(kind,graph)
    with np.load(DATA) as ar:
        split = ar['split']
        features = arrays(kind,ar,model,'simulated')
        sides = torch.from_numpy(ar['sim_features'].astype(np.float32))
        sides[...,0] /= SCALE
        rollouts = torch.from_numpy(ar['scores'].astype(np.float32))/SCALE
        outcomes = torch.from_numpy(ar['risks']).float()
        actions, baseline = ar['actions'],ar['robust']
    optimizer = torch.optim.AdamW(list(model.scorer.model.parameters())+list(model.adjust.parameters()),lr=.001)
    train_ids, dev = np.flatnonzero(split==0),np.flatnonzero(split==1)
    best, history = -float('inf'), []
    for epoch in range(epochs):
        model.scorer.model.train()
        for batch in np.array_split(rng.permutation(train_ids),max(1,len(train_ids)//32)):
            score,risk = model.estimate(features[batch],sides[batch])
            loss = objective(score,risk,rollouts[batch],outcomes[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.scorer.model.eval()
        with torch.no_grad():
            score,_ = model.estimate(features[dev],sides[dev])
            stats = accuracy(score,rollouts[dev],actions[dev],baseline[dev])
        history.append(dict(epoch=epoch+1,**stats))
        print(kind,history[-1],flush=True)
        merit = -stats['mean_regret_points']
        if merit > best:
            best = merit
            torch.save(dict(kind=kind,model=model.scorer.model.state_dict(),adjust=model.adjust.state_dict(),
                            epoch=epoch+1),ROOT/f'{kind}-ranker.pt')
    json_file(ROOT/f'{kind}-training.json',dict(seconds=time.perf_counter()-started,epochs=history,
              selected=max(history,key=lambda r:-r['mean_regret_points'])))


def load_ranker(kind,graph):
    model = Ranker(kind,graph)
    ckpt = torch.load(ROOT/f'{kind}-ranker.pt',map_location='cpu',weights_only=True)
    model.scorer.model.load_state_dict(ckpt['model'])
    model.adjust.load_state_dict(ckpt['adjust'])
    model.scorer.model.eval()
    return model


@torch.no_grad()
def inspect():
    graph = json.loads(Path('artifacts/graph.json').read_text())
    metrics = []
    with np.load(DATA) as archive:
        for kind in ('compact','flywire'):
            model = load_ranker(kind,graph)
            for source,key in (('reconstructed','simulated'),('exact','exact')):
                board = archive[key]
                side = archive['sim_features' if source=='reconstructed' else 'exact_features']
                for profile,flag in (('clean',0),('joint',1)):
                    mask = (archive['split']==2)&(archive['profile']==flag)
                    x = board[mask]
                    embedded = model.scorer.features(torch.from_numpy(x).flatten(0,1)).reshape(len(x),6,-1)
                    extras = torch.from_numpy(side[mask].astype(np.float32))
                    extras[...,0] /= SCALE
                    score,_ = model.estimate(embedded,extras)
                    rows = accuracy(score,torch.from_numpy(archive['scores'][mask].astype(np.float32))/SCALE,
                                    archive['actions'][mask],archive['robust'][mask])
                    metrics.append(dict(kind=kind,input=source,profile=profile,**rows))
    json_file(ROOT/'ranking-accuracy.json',metrics)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['compact','flywire','inspect'])
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    inspect() if args.mode=='inspect' else train(args.mode,args.epochs)
