"""Render paired gameplay and save an inspectable FlyWire afterstate trace."""
from collections import Counter
import copy
import hashlib
import json
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont

from . import planner_video as shared
from .afterstate_data import ROOT, SHOWCASE
from .afterstate_eval import METHODS, load_models
from .planner_experiment import CHECKPOINT, overflow
from .policy import np
from .replay import json_file, pack_script


class LearnedLane(shared.Lane):
    def __init__(self, method, model, weights):
        name = 'Compact' if method == 'C' else 'FlyWire'
        super().__init__(method,f'{method}  {name} afterstate')
        self.model = model
        self.weights = weights

    def step(self, _):
        if self.ended:
            return
        env = self.env
        self.current,self.next = env.current,env.upcoming
        pre, highest = env.score,env.highest
        received = self.obs.tolist()
        action, details = self.model.decide(self.obs,*self.weights,trace=True)
        # Compare immediate outcomes on this exact local board for illustrations.
        alternatives = np.where(details['terminating'],-1.,details['immediate'])
        alternate = int(alternatives.argmax())
        foregone = 0.
        if alternatives[alternate] > details['immediate'][action]:
            candidate = copy.deepcopy(env)
            candidate.capture = False
            _,foregone,term,_,_ = candidate.step(alternate)
            if term or overflow(candidate):
                foregone = 0.
        self.obs,gain,terminal,truncated,_ = env.step(action)
        self.ended = terminal or truncated
        self.records.append(dict(drop=env.drops,action=action,score=env.score,before=pre,
                                 observation=received,foregone=foregone,gain=gain,
                                 tier=env.highest,prior_tier=highest,
                                 settled=env.body[:env.n,[0,1,4]].tolist(),**details))


def illustrate(records):
    """Recover individual merges from fruit-count conservation between drops."""
    last = Counter()
    peaks = []
    for row in records:
        current = int(round(row['observation'][-2]*10))
        expected = last.copy()
        expected[current] += 1
        actual = Counter(int(fruit[2]) for fruit in row['settled'])
        incoming, peak = 0, 0
        for tier in range(10):
            incoming = (expected[tier]+incoming-actual[tier])//2
            if incoming:
                peak = (tier+1)*(tier+2)//2
        peaks.append(peak)
        last = actual
    for i,row in enumerate(records):
        if row['foregone'] <= row['gain']:
            continue
        for j in range(i+1,min(i+6,len(records))):
            if peaks[j] > row['foregone']:
                return dict(drop=row['drop'],actual_gain=row['gain'],foregone=row['foregone'],
                            later_drop=records[j]['drop'],later_gain=records[j]['gain'],
                            individual_merge=peaks[j])
    return None


def main():
    evaluation = json.loads((ROOT/'evaluation.json').read_text())
    strongest = evaluation['best_learned']
    policy,models = load_models()
    graph = json.loads(open('artifacts/graph.json').read())
    points = np.array(graph['positions'],dtype=float)
    points = np.column_stack((points[:,0]*.9+points[:,2]*.35,points[:,1]))
    shared.POINTS = (points-points.min(0))/np.ptp(points,axis=0)
    a = shared.Lane('hybrid','A  Deployed hybrid')
    c = LearnedLane(strongest,models[METHODS[strongest]],evaluation['weights'][strongest])
    # The common seed is defined in the dataset protocol, independent of the earlier planner pilot.
    for lane in (a,c):
        lane.obs,_ = lane.env.reset(seed=SHOWCASE)
    ffmpeg = get_ffmpeg_exe()
    path = ROOT/'comparison.mp4'
    command = [ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1000x416',
               '-r',str(shared.FPS),'-i','-','-an','-c:v','libx264','-threads','2','-crf','26',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        while not (a.ended and c.ended):
            frozen = [a.ended,c.ended]
            a.step(policy)
            c.step(policy)
            for frame in range(len(shared.FRAME_IDS)):
                image = Image.new('RGB',shared.SIZE,'#162628')
                draw = ImageDraw.Draw(image)
                draw.font = ImageFont.load_default(size=16)
                for lane,x,finished in zip((a,c),(0,500),frozen):
                    lane.draw(draw,x,frame,finished)
                encoder.stdin.write(image.tobytes())
        for _ in range(shared.FPS*2):
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoding failed')
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True)
    fly = c
    if strongest != 'D':
        fly = LearnedLane('D',models['flywire'],evaluation['weights']['D'])
        fly.obs,_ = fly.env.reset(seed=SHOWCASE)
        while not fly.ended:
            fly.step(policy)
    shown = {'A':a,'learned':c,'flywire':fly}
    manifest = dict(seed=SHOWCASE,profile='joint',fps=shared.FPS,frames_per_drop=len(shared.FRAME_IDS),
                    checkpoint=str(CHECKPOINT),sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    model=strongest,weights=evaluation['weights'], bytes=path.stat().st_size,
                    model_hashes={m:hashlib.sha256((ROOT/f'{m}.pt').read_bytes()).hexdigest()
                                  for m in ('compact','flywire')},
                    illustration=illustrate(c.records),episodes={
                         key:dict(**lane.env.info(),overflow=overflow(lane.env)) for key,lane in shown.items()})
    json_file(ROOT/'comparison.json',manifest)
    traces = {key:lane.records for key,lane in shown.items() if key!='flywire' or fly is not c}
    pack_script(ROOT/'comparison.js','comparison',dict(meta=manifest,points=shared.POINTS.tolist(),lanes=traces))


if __name__ == '__main__':
    main()
