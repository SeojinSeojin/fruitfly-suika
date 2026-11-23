"""Play back the predeclared paired game and verify each saved decision."""
import argparse
import hashlib
import json
import subprocess

from imageio_ffmpeg import count_frames_and_secs,get_ffmpeg_exe
from PIL import Image,ImageDraw,ImageFont

from . import planner_video as shared
from .afterstate_eval import load_models
from .planner_experiment import CHECKPOINT,environment,overflow
from .policy import np
from .replay import json_file,pack_script
from .rollout_eval import ROOT,SHOWCASE,decision
from .verify import unpack


class Lane(shared.Lane):
    def __init__(self,row):
        self.source = row
        self.env = environment('joint',cap=450,capture=True)
        self.obs,_ = self.env.reset(seed=SHOWCASE)
        self.label = 'Robust one-drop' if row['method']=='robust' else f"{row['method'].title()} reconstructed rollout"
        self.records,self.ended = [],False

    def step(self,policy,models):
        if self.ended:
            return
        env = self.env
        old = env.score
        self.current,self.next = env.current,env.upcoming
        expected = self.source['decisions'][env.drops]
        action,trace = decision(env,self.obs,policy,models,self.source['method'],SHOWCASE,env.drops+1,trace=True)
        self.obs,gain,terminated,truncated,_ = env.step(action)
        self.ended = terminated or truncated
        assert (action,gain,env.score) == (expected['action'],expected['gain'],expected['score'])
        self.records.append(dict(drop=env.drops,action=action,before=old,score=env.score,**trace))


def sources():
    result = json.loads((ROOT/'results.json').read_text())
    rows = [next(r for r in result['episodes'] if r['seed']==SHOWCASE and r['profile']=='joint' and r['method']==m)
            for m in ('robust',result['selected'])]
    return rows,result['selected']


def encode_comparison(path,a,b,step):
    ffmpeg = get_ffmpeg_exe()
    command = [ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1000x416',
               '-r',str(shared.FPS),'-i','-','-an','-c:v','libx264','-threads','2','-crf','26',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        while not (a.ended and b.ended):
            frozen = [a.ended,b.ended]
            step(a)
            step(b)
            for frame in range(len(shared.FRAME_IDS)):
                image = Image.new('RGB',shared.SIZE,'#162628')
                draw = ImageDraw.Draw(image)
                draw.font = ImageFont.load_default(size=16)
                for lane,offset,finished in zip((a,b),(0,500),frozen):
                    lane.draw(draw,offset,frame,finished)
                draw.text((850,195),'Diagnostic only',fill='#91aaa3')
                encoder.stdin.write(image.tobytes())
        for _ in range(shared.FPS*2):
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoding failed')
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True)


def render():
    rows,variant = sources()
    a,b = map(Lane,rows)
    policy,models = load_models()
    graph = json.loads(open('artifacts/graph.json').read())
    positions = np.array(graph['positions'],dtype=float)
    points = np.column_stack((positions[:,0]*.9+positions[:,2]*.35,positions[:,1]))
    shared.POINTS = (points-points.min(0))/np.ptp(points,axis=0)
    path = ROOT/'comparison.mp4'
    encode_comparison(path,a,b,lambda lane:lane.step(policy,models))
    meta = dict(seed=SHOWCASE,variant=variant,profile='joint',bytes=path.stat().st_size,
                fps=shared.FPS,frames_per_drop=len(shared.FRAME_IDS),checkpoint=str(CHECKPOINT),
                sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                episodes={'A':rows[0]['extended'],'C':rows[1]['extended']})
    json_file(ROOT/'comparison.json',meta)
    pack_script(ROOT/'comparison.js','comparison',dict(meta=meta,points=shared.POINTS.tolist(),
                     lanes={'A':a.records,'C':b.records}))


def verify():
    data = unpack(ROOT/'comparison.js')
    meta = data['meta']
    assert meta == json.loads((ROOT/'comparison.json').read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == meta['sha256']
    path = ROOT/'comparison.mp4'
    assert path.stat().st_size == meta['bytes']
    count = max(map(len,data['lanes'].values()))
    assert count_frames_and_secs(str(path))[0] == count*meta['frames_per_drop']+2*meta['fps']
    rows,_ = sources()
    policy,models = load_models()
    for key,row in zip(('A','C'),rows):
        lane = Lane(row)
        for record in data['lanes'][key]:
            lane.step(policy,models)
            assert record == lane.records[-1]
        assert lane.ended and dict(**lane.env.info(),overflow=overflow(lane.env))==meta['episodes'][key]
        print(row['method'],len(lane.records),'aligned decisions, scores and circuit traces',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    verify() if parser.parse_args().verify else render()
