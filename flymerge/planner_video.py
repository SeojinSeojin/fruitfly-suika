"""Render both policies against the predeclared paired fruit sequence."""
import hashlib
import json
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont

from .game import HEIGHT
from .planner_experiment import CHECKPOINT, ROOT, SHOWCASE, environment, load, overflow
from .policy import np
from .replay import fruit, json_file, pack_script

FPS = 20
FRAME_IDS = (0, 5, 10, 15, 21, 27)
SIZE = (1000, 416)


def color(delta):
    gain = abs(delta)
    end = (235, 164, 97) if delta >= 0 else (84, 169, 233)
    return tuple(round(base+(top-base)*gain) for base,top in zip((51,67,69),end))


class Lane:
    def __init__(self, strategy, label):
        self.env = environment('joint', cap=300, capture=True)
        self.obs, _ = self.env.reset(seed=SHOWCASE)
        self.rng = np.random.default_rng(np.random.SeedSequence([SHOWCASE,8051]))
        self.strategy, self.label = strategy, label
        self.records, self.ended = [], False

    def step(self, policy):
        if self.ended:
            return
        env = self.env
        self.current, self.next = env.current, env.upcoming
        pre = env.score
        action, trace = policy.decide(self.obs, self.rng, self.strategy)
        received = self.obs.tolist()
        self.obs, _, terminal, truncated, _ = env.step(action)
        self.ended = terminal or truncated
        self.records.append(dict(drop=env.drops, action=action, score=env.score, before=pre,
                                 observation=received, **trace))

    def draw(self, draw, offset, index, finished):
        scale = .77
        env = self.env
        data = self.records[-1]
        board, score = env.frames[FRAME_IDS[index] if not finished else -1]
        left, top = 15+offset, 61
        draw.rectangle((left,top,left+env.width*scale,top+HEIGHT*scale), fill='#253e3d', outline='#83a29b')
        aim = left+env.aim*scale
        draw.line((aim,top,aim,top+HEIGHT*scale), fill='#89a898')
        draw.line((left,top+env.overflow_y*scale,left+env.width*scale,top+env.overflow_y*scale), fill='#b98975')
        for x,y,tier in board:
            fruit(draw,left+x*scale,top+y*scale,env.radii[int(tier)]*scale,int(tier))
        draw.text((left,9), self.label, fill='#f4e9ce')
        draw.text((left,33), f"Drop {data['drop']}   Aim {data['action']+1}/15   Score {score}", fill='#d5e8df')
        side = offset+350
        draw.text((side,73),'Current    Next',fill='#c2d8d2')
        fruit(draw,side+21,116,15,self.current)
        fruit(draw,side+91,116,15,self.next)
        draw.text((side,156),'FlyWire activity',fill='#d5e8df')
        draw.text((side,178),'Recurrent step 6',fill='#9ebcb5')
        for i,(x,y) in enumerate(POINTS):
            px,py = side+5+x*130,215+y*120
            draw.ellipse((px-1.6,py-1.6,px+1.6,py+1.6), fill=color(data['steps'][-1]['delta'][i]))
        if finished:
            draw.text((side,354),'Overflow' if overflow(env) else '300-drop cap',fill='#f7cc82')


def main():
    global POINTS
    graph = json.loads(open('artifacts/graph.json').read())
    points = np.array(graph['positions'], dtype=float)
    points = np.column_stack((points[:,0]*.9+points[:,2]*.35,points[:,1]))
    POINTS = (points-points.min(0))/np.ptp(points,axis=0)
    a, c = Lane('hybrid','A  Legacy hybrid'), Lane('robust','C  Perturbation-aware hybrid')
    policy = load()
    path = ROOT/'comparison.mp4'
    ffmpeg = get_ffmpeg_exe()
    command = [ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1000x416',
               '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-crf','26',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        while not (a.ended and c.ended):
            frozen = [a.ended,c.ended]
            a.step(policy)
            c.step(policy)
            for frame in range(len(FRAME_IDS)):
                image = Image.new('RGB',SIZE,'#162628')
                draw = ImageDraw.Draw(image)
                draw.font = ImageFont.load_default(size=16)
                for lane,x,finished in zip((a,c),(0,500),frozen):
                    lane.draw(draw,x,frame,finished)
                encoder.stdin.write(image.tobytes())
        for _ in range(FPS*2):
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoding failed')
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True)
    manifest = dict(seed=SHOWCASE, profile='joint', fps=FPS, frames_per_drop=len(FRAME_IDS),
                    checkpoint=str(CHECKPOINT), sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    video=path.name, bytes=path.stat().st_size, episodes={
                        lane.label[0]:dict(**lane.env.info(), overflow=overflow(lane.env)) for lane in (a,c)})
    json_file(ROOT/'comparison.json',manifest)
    pack_script(ROOT/'comparison.js','comparison',dict(meta=manifest, points=POINTS.tolist(),
                                                        lanes={'A':a.records,'C':c.records}))


if __name__ == '__main__':
    main()
