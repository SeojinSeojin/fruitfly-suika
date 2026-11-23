"""Paired held-out replay with a causal detection and estimate overlay."""
import argparse
import hashlib
import json

from imageio_ffmpeg import count_frames_and_secs

from . import planner_video as shared
from .afterstate_eval import load_models
from .planner_experiment import CHECKPOINT,environment,overflow
from .policy import np
from .replay import json_file,pack_script
from .rollout_video import encode_comparison
from .state_calibration import ROOT
from .state_estimation import local_history,track
from .state_experiment import SHOWCASE
from .state_search import decide
from .verify import unpack


class Lane(shared.Lane):
    def __init__(self,row):
        self.row=row
        self.env=environment('joint',cap=450,capture=True)
        self.obs,_=self.env.reset(seed=SHOWCASE)
        self.label={'robust':'Robust one-drop','current':'Single-frame rollout',
                    'tracking':'Tracked-center rollout','physics':'Physics scenario rollout',
                    'both':'Tracking + physics scenarios'}[row['method']]
        self.records,self.ended=[],False

    def step(self,policy,models,config):
        if self.ended:
            return
        env=self.env
        drop=env.drops+1
        old=env.score
        self.current,self.next=env.current,env.upcoming
        expected=self.row['show'][env.drops]
        history=local_history(env,self.obs,SHOWCASE,drop)
        action,trace=decide(self.obs,history,policy,models,self.row['method'],SHOWCASE,drop,config,trace=True)
        if self.row['method']!='robust':
            (board,n,_,_),_=track(self.obs,history,config['tracking_velocity'])
            trace['tracked_diagnostic']=board[:n].tolist()
        self.obs,gain,terminated,truncated,_=env.step(action)
        self.ended=terminated or truncated
        assert (action,gain,env.score)==(expected['action'],expected['gain'],expected['score'])
        self.records.append(dict(drop=drop,action=action,before=old,score=env.score,**trace))


def inputs():
    result=json.loads((ROOT/'results.json').read_text())
    rows=[next(r for r in result['episodes'] if r['profile']=='joint' and r['seed']==SHOWCASE and r['method']==name)
          for name in ('robust',result['best'])]
    config=json.loads((ROOT/'frozen.json').read_text())
    return rows,result['best'],config


def render():
    rows,method,config=inputs()
    a,b=map(Lane,rows)
    policy,models=load_models()
    graph=json.loads(open('artifacts/graph.json').read())
    xyz=np.array(graph['positions'],dtype=float)
    p=np.column_stack((xyz[:,0]*.9+xyz[:,2]*.35,xyz[:,1]))
    shared.POINTS=(p-p.min(0))/np.ptp(p,axis=0)
    path=ROOT/'comparison.mp4'
    encode_comparison(path,a,b,lambda lane:lane.step(policy,models,config))
    meta=dict(seed=SHOWCASE,profile='joint',method=method,bytes=path.stat().st_size,
              fps=shared.FPS,frames_per_drop=len(shared.FRAME_IDS),
              checkpoint=str(CHECKPOINT),sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              episodes={'A':rows[0]['extended'],'C':rows[1]['extended']})
    json_file(ROOT/'comparison.json',meta)
    pack_script(ROOT/'comparison.js','comparison',dict(meta=meta,points=shared.POINTS.tolist(),
                lanes={'A':a.records,'C':b.records}))


def verify():
    data=unpack(ROOT/'comparison.js')
    meta=data['meta']
    assert meta==json.loads((ROOT/'comparison.json').read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==meta['sha256']
    path=ROOT/'comparison.mp4'
    assert path.stat().st_size==meta['bytes']
    frames=max(map(len,data['lanes'].values()))*meta['frames_per_drop']+2*meta['fps']
    assert count_frames_and_secs(str(path))[0]==frames
    rows,_,config=inputs()
    policy,models=load_models()
    for key,row in zip(('A','C'),rows):
        lane=Lane(row)
        for record in data['lanes'][key]:
            lane.step(policy,models,config)
            assert record==lane.records[-1]
        assert lane.ended and dict(**lane.env.info(),overflow=overflow(lane.env))==meta['episodes'][key]
        print(row['method'],len(lane.records),'aligned decisions, scores and traces',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    verify() if parser.parse_args().verify else render()
