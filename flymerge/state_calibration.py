"""Use archived executed drops to assess plausible physics, without causal claims."""
import json
from pathlib import Path

from .planner_experiment import environment,load,overflow
from .policy import np
from .replay import json_file
from .rollout_transfer import reconstruct,NOMINAL
from .rollout_teacher import settle_exact
from .state_estimation import local_history,match,track
from .verify import unpack

ROOT = Path('artifacts/state-estimation')
DEV = tuple(range(20271501,20271505))
FIT = Path('artifacts/transfer/demo-game-over/website.js')
VALIDATE = Path('artifacts/planner/website/website.js')
# The engine's existing domain ranges, tested around nominal along three directions.
OFFSETS = ((.045,.012,.045,.065),(.045,-.012,.045,-.065),(.045,.012,-.045,-.065))


def calibrate_physics():
    pairs = [[tuple(np.array(NOMINAL)+sign*np.array(offset)) for sign in (-1,1)] for offset in OFFSETS]
    settings = [NOMINAL,*[s for pair in pairs for s in pair]]
    tables = {}
    for name,path in (('fit',FIT),('validation',VALIDATE)):
        capture = unpack(path)
        reports=[]
        for physics in settings:
            agree = tp = fp = fn = 0
            errors=[]
            for r in capture['decisions']:
                if r['score'] is None or r['score_before'] is None:
                    continue
                obs = np.array(r['observation'],dtype=np.float32)
                board,count,current,_ = reconstruct(obs)
                if current < 0:
                    continue
                _,predicted,_ = settle_exact(board,count,current,r['action'],physics)
                observed=r['score']-r['score_before']
                a,b=predicted>0,observed>0
                agree += a==b
                tp += a and b
                fp += a and not b
                fn += b and not a
                errors.append(abs(predicted-observed))
            reports.append(dict(physics=physics,n=len(errors),matches=agree,tp=tp,fp=fp,fn=fn,
                                mae_points=float(np.mean(errors))))
        tables[name]=dict(path=str(path),settings=reports)
    fit=tables['fit']['settings']
    best=max(range(len(pairs)),key=lambda i:fit[2*i+1]['matches']+fit[2*i+2]['matches'])
    chosen=[NOMINAL,*pairs[best]]
    json_file(ROOT/'physics-calibration.json',dict(chosen=chosen,axis=best,records=tables,
              fit_rule='best combined score-increase agreement across symmetric pairs on fit capture'))
    return chosen


def calibrate_tracking():
    policy=load()
    rows=[]
    for profile in ('clean','joint'):
        for seed in DEV:
            env=environment(profile,cap=450,capture=True)
            obs,_=env.reset(seed=seed)
            for drop in range(1,301):
                if drop>1 and drop%5==0:
                    history=local_history(env,obs,seed,drop)
                    truth = env.body[:env.n]
                    initial,count,_,_ = reconstruct(obs)
                    pairs = match(initial[:count],truth,25.,0.)
                    good = np.flatnonzero(pairs>=0)
                    for name in ('single','centers','velocity'):
                        state,accepted = (reconstruct(obs),0) if name=='single' else track(obs,history,name=='velocity')
                        pred=state[0][good]
                        target=truth[pairs[good]]
                        rows.append(dict(profile=profile,seed=seed,drop=drop,method=name,matched=len(good),
                                         detections=count,truth_count=env.n,accepted=accepted,
                                         position_sq=float(np.square(pred[:,:2]-target[:,:2]).sum()),
                                         velocity_sq=float(np.square(pred[:,2:4]-target[:,2:4]).sum()),
                                         near_stationary=int((np.linalg.norm(target[:,2:4],axis=1)<.35).sum())))
                action,_=policy.decide(obs,np.random.default_rng(seed+drop),'robust',trace=False)
                obs,_,terminal,truncated,_=env.step(action)
                if terminal or truncated:
                    break
    summary=[]
    for profile in ('clean','joint'):
        for method in ('single','centers','velocity'):
            group=[r for r in rows if r['profile']==profile and r['method']==method]
            matches=sum(r['matched'] for r in group)
            summary.append(dict(profile=profile,method=method,boards=len(group),matched=matches,
                                detections=sum(r['detections'] for r in group),
                                truth_count=sum(r['truth_count'] for r in group),
                                position_rmse=float(np.sqrt(sum(r['position_sq'] for r in group)/matches)),
                                velocity_rmse=float(np.sqrt(sum(r['velocity_sq'] for r in group)/matches)),
                                near_stationary_fraction=sum(r['near_stationary'] for r in group)/matches,
                                accepted=sum(r['accepted'] for r in group)))
    nominal=next(r for r in summary if r['profile']=='joint' and r['method']=='centers')
    moving=next(r for r in summary if r['profile']=='joint' and r['method']=='velocity')
    use=moving['velocity_rmse']<nominal['velocity_rmse'] and moving['position_rmse']<=nominal['position_rmse']
    json_file(ROOT/'tracking-calibration.json',dict(seeds=DEV,summary=summary,use_velocity=use,
              selection_rule='use velocity only if it reduces joint velocity RMSE without increasing center RMSE',rows=rows))
    return use


if __name__=='__main__':
    chosen=calibrate_physics()
    use=calibrate_tracking()
    json_file(ROOT/'frozen.json',dict(physics=chosen,tracking_velocity=use,branches=18,
              condition='shared tail scenarios paired with randomly permuted physics settings'))
