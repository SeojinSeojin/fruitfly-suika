"""Render paired held-out gameplay and a FlyWire candidate-circuit view."""
import copy
import hashlib
import json
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont

from . import planner_video as shared
from .planner_experiment import CHECKPOINT, environment, overflow
from .policy import np
from .ranking_data import ROOT, SHOWCASE
from .ranking_eval import components,select
from .replay import json_file,pack_script


class Lane(shared.Lane):
    def __init__(self,method):
        self.method = method
        self.env = environment('joint',cap=450,capture=True)
        self.obs,_ = self.env.reset(seed=SHOWCASE)
        self.records,self.ended = [],False
        self.label = {'robust':'Robust one-drop','compact':'Compact ranker','flywire':'FlyWire ranker'}[method]

    def step(self,pipeline):
        if self.ended:
            return
        env = self.env
        self.current,self.next = env.current,env.upcoming
        score = env.score
        action,trace = select(self.method,self.obs,*pipeline,SHOWCASE+env.drops+1,trace=True)
        other = None
        if 'baseline' in trace and trace['baseline'] != action:
            fork = copy.deepcopy(env)
            fork.capture = False
            _,other,_,_,_ = fork.step(trace['baseline'])
        self.obs,gain,terminated,truncated,_ = env.step(action)
        self.ended = terminated or truncated
        self.records.append(dict(drop=env.drops,action=action,before=score,score=env.score,
                                 immediate_gain=gain,alternate_gain=other,**trace))


def example(rows):
    disagreements = [(i,row) for i,row in enumerate(rows)
                     if row['alternate_gain'] is not None and i+20 < len(rows)]
    if not disagreements:
        return None
    i,row = next(((i,r) for i,r in disagreements if r['immediate_gain'] != r['alternate_gain']),
                 disagreements[0])
    return dict(drop=row['drop'],ranker_aim=row['action']+1,robust_aim=row['baseline']+1,
                immediate=row['immediate_gain'],alternate=row['alternate_gain'],
                next_20=rows[i+20]['score']-row['score'])


def summary(lane):
    return dict(**lane.env.info(),overflow=overflow(lane.env),rightmost=sum(r['action']==14 for r in lane.records))


def main():
    results = json.loads((ROOT/'evaluation.json').read_text())
    strongest = results['best_learned']
    pipeline = components()
    graph = json.loads(open('artifacts/graph.json').read())
    positions = np.array(graph['positions'],dtype=float)
    p = np.column_stack((positions[:,0]*.9+positions[:,2]*.35,positions[:,1]))
    shared.POINTS = (p-p.min(0))/np.ptp(p,axis=0)
    a,b = Lane('robust'),Lane(strongest)
    ffmpeg = get_ffmpeg_exe()
    path = ROOT/'comparison.mp4'
    command = [ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1000x416',
               '-r',str(shared.FPS),'-i','-','-an','-c:v','libx264','-threads','2','-crf','26',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        while not (a.ended and b.ended):
            frozen = [a.ended,b.ended]
            a.step(pipeline)
            b.step(pipeline)
            for frame in range(len(shared.FRAME_IDS)):
                image = Image.new('RGB',shared.SIZE,'#162628')
                draw = ImageDraw.Draw(image)
                draw.font = ImageFont.load_default(size=16)
                for lane,offset,finished in zip((a,b),(0,500),frozen):
                    lane.draw(draw,offset,frame,finished)
                encoder.stdin.write(image.tobytes())
        for _ in range(shared.FPS*2):
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoding failed')
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True)
    fly = b
    if strongest != 'flywire':
        fly = Lane('flywire')
        while not fly.ended:
            fly.step(pipeline)
    lanes = {'A':a,'learned':b,'flywire':fly}
    metadata = dict(seed=SHOWCASE,profile='joint',fps=shared.FPS,frames_per_drop=len(shared.FRAME_IDS),
                    bytes=path.stat().st_size, model=strongest,checkpoint=str(CHECKPOINT),
                    sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    ranker_sha256={m:hashlib.sha256((ROOT/f'{m}-ranker.pt').read_bytes()).hexdigest()
                                   for m in ('compact','flywire')},
                    illustration=example(b.records),episodes={k:summary(v) for k,v in lanes.items()})
    json_file(ROOT/'comparison.json',metadata)
    pack_script(ROOT/'comparison.js','comparison',dict(meta=metadata,points=shared.POINTS.tolist(),
                lanes={k:v.records for k,v in lanes.items() if k != 'flywire' or fly is not b}))


if __name__ == '__main__':
    main()
