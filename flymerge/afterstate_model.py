"""Matched value heads over a settled board and the previously visible next fruit."""
import argparse
import math
import time

from torch import nn

from .policy import Policy, np, torch
from .afterstate_data import DATA, ROOT, candidate_boards
from .replay import json_file

ACTIONS = np.arange(15,dtype=np.int64)
SCORE_SCALE = 100.


class Compact(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,24),nn.SiLU())
        self.head = nn.Sequential(nn.Linear(24*7+13,64),nn.SiLU(),nn.Linear(64,2))

    def forward(self, observation):
        fruit = observation[:,:-2].reshape(len(observation),-1,3)
        valid = fruit[...,0] >= 0
        width = max(1,int(valid.sum(1).max()))
        fruit, valid = fruit[:,:width],valid[:,:width]
        features = self.encoder(fruit.clamp_min(0)) * valid[...,None]
        mass = valid.sum(1,keepdim=True)
        cols = (fruit[...,0]*5).long().clamp(0,4)
        groups = nn.functional.one_hot(cols,5).to(features.dtype).transpose(1,2)
        by_column = torch.bmm(groups,features)/(groups @ valid[...,None].float()).clamp_min(1)
        next_tier = (observation[:,-2]*10).round().long()
        next_tier = torch.where(next_tier >= 0,next_tier,11)
        next_features = nn.functional.one_hot(next_tier,12).float()
        maximum = features.masked_fill(~valid[...,None], -torch.inf).amax(1)
        maximum = torch.where(mass > 0,maximum,0.)
        return self.head(torch.cat([maximum, features.sum(1)/mass.clamp_min(1),
                                    by_column.flatten(1),next_features,mass.float()/80],dim=1))


def readout(kind):
    return Compact() if kind == 'compact' else nn.Sequential(nn.Linear(512,64),nn.SiLU(),nn.Linear(64,2))


class Scorer:
    def __init__(self, kind, graph):
        self.kind = kind
        self.model = readout(kind)
        self.reservoir = Policy('flywire',graph,sparse=True).eval() if kind == 'flywire' else None

    def features(self, tensor):
        return self.reservoir.circuit(tensor) if self.reservoir is not None else tensor

    @torch.no_grad()
    def decide(self, obs, future_weight, overflow_penalty, trace=False):
        views, points, ends = candidate_boards(obs,ACTIONS)
        tensor = torch.from_numpy(views)
        predictions = self.model(self.features(tensor))
        future = predictions[:,0].clamp_min(0).numpy()*SCORE_SCALE
        risk = predictions[:,1].sigmoid().numpy()
        scores = points+future_weight*future-overflow_penalty*risk
        if ends.all():
            scores = points
        else:
            scores[ends] = -np.inf
        action = int(scores.argmax())
        details = dict(immediate=points.tolist(), future=future.tolist(), risk=risk.tolist(),
                       terminating=ends.tolist(), board=views[action].tolist())
        if trace and self.reservoir is not None:
            x = tensor[action:action+1]
            _, direct, states = self.reservoir.circuit(x,trace=True)
            _, _, reference = self.reservoir.circuit(torch.zeros_like(x),trace=True)
            pack = lambda a: np.round(a[0].numpy().astype(float),5).tolist()
            details.update(direct=pack(direct),steps=[
                dict(recurrent=pack(rec), activation=pack(z),reference=pack(zero),delta=pack(z-zero))
                for (rec,z),(_,zero) in zip(states,reference)])
        return action, details


def loss_parts(prediction, target, risk, known_score, known_risk):
    score_loss = nn.functional.smooth_l1_loss(prediction[:,0],target,reduction='none')
    risk_loss = nn.functional.binary_cross_entropy_with_logits(prediction[:,1],risk,reduction='none')
    return ((score_loss*known_score).sum()/known_score.sum().clamp_min(1),
            (risk_loss*known_risk).sum()/known_risk.sum().clamp_min(1))


def train(kind, graph, epochs):
    started = time.perf_counter()
    torch.manual_seed(7321)
    rng = np.random.default_rng(7321)
    with np.load(DATA) as archive:
        views = torch.from_numpy(np.concatenate([archive['clean'],archive['sim']]))
        index = np.tile(archive['split'],2)
        y = torch.from_numpy(np.tile(archive['target'],2))/SCORE_SCALE
        r = torch.from_numpy(np.tile(archive['risk'],2)).float()
        sy = torch.from_numpy(np.tile(archive['score_known'],2))
        sr = torch.from_numpy(np.tile(archive['risk_known'],2))
    train_idx, valid_idx = np.flatnonzero(index == 0),np.flatnonzero(index == 1)
    scorer = Scorer(kind,graph)
    with torch.no_grad():
        x = torch.cat([scorer.features(b) for b in views.split(1024)]) if kind == 'flywire' else views
    optimizer = torch.optim.AdamW(scorer.model.parameters(),lr=.001)
    best, learning = float('inf'), []
    for epoch in range(epochs):
        scorer.model.train()
        indices = rng.permutation(train_idx)
        for batch in np.array_split(indices,math.ceil(len(indices)/512)):
            score, risk = loss_parts(scorer.model(x[batch]),y[batch],r[batch],sy[batch],sr[batch])
            optimizer.zero_grad()
            (score+.5*risk).backward()
            optimizer.step()
        scorer.model.eval()
        with torch.no_grad():
            prediction = torch.cat([scorer.model(x[b]) for b in np.array_split(valid_idx,math.ceil(len(valid_idx)/512))])
            a,b = loss_parts(prediction,y[valid_idx],r[valid_idx],sy[valid_idx],sr[valid_idx])
            mae = ((prediction[:,0]-y[valid_idx]).abs()*sy[valid_idx]).sum()/sy[valid_idx].sum()
        value = float(a+.5*b)
        learning.append(dict(epoch=epoch+1,score_loss=float(a),risk_loss=float(b),val_mae_points=float(mae)*SCORE_SCALE))
        print(kind,learning[-1],flush=True)
        if value < best:
            best = value
            torch.save(dict(kind=kind,model=scorer.model.state_dict(),epoch=epoch+1),ROOT/f'{kind}.pt')
    json_file(ROOT/f'{kind}-training.json',dict(seconds=time.perf_counter()-started,
              parameters=sum(p.numel() for p in scorer.model.parameters()),train_examples=len(train_idx),
              validation_examples=len(valid_idx),epochs=learning))


def restore_scorer(path, graph):
    ckpt = torch.load(path,map_location='cpu',weights_only=True)
    scorer = Scorer(ckpt['kind'],graph)
    scorer.model.load_state_dict(ckpt['model'])
    scorer.model.eval()
    return scorer


if __name__ == '__main__':
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['compact','flywire'])
    parser.add_argument('--epochs',type=int,default=9)
    args = parser.parse_args()
    train(args.kind,json.loads(open('artifacts/graph.json').read()),args.epochs)
