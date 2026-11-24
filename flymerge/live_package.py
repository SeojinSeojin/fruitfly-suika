"""Copy compact website-capture media from runs/live-baseline into artifacts/live-baseline."""
import json
import shutil
from pathlib import Path

from PIL import Image

RUNS = Path('runs/live-baseline')
ROOT = Path('artifacts/live-baseline')
TEMPLATE = Path('artifacts/planner/website.html')


def main():
    captures, scripts = {}, []
    for k in (1, 2, 3):
        source, target = RUNS/f'attempt-{k}', ROOT/f'attempt-{k}'
        if not (source/'website.js').exists():
            continue
        (target/'screens').mkdir(parents=True, exist_ok=True)
        for name in ('manifest.json', 'website.js', 'end.png', 'paused.png', 'time-cap.jpg'):
            if (source/name).exists():
                shutil.copy2(source/name, target/name)
        for image in sorted((source/'screens').glob('*.jpg')):
            with Image.open(image) as im:
                im.convert('RGB').resize((im.width//2, im.height//2)).save(target/'screens'/image.name, quality=72, optimize=True)
        meta = json.loads((source/'manifest.json').read_text())
        captures[f'liveBaseline{k}'] = dict(label=f'Attempt {k} · {meta["final_score"]} points', folder=f'attempt-{k}/')
        scripts.append(f'<script src="attempt-{k}/website.js"></script>')
    page = TEMPLATE.read_text()
    page = page.replace("window.siteCaptures={plannerWebsite:{label:'Perturbation-aware planner',folder:'website/'}};",
                        f'window.siteCaptures={json.dumps(captures)};')
    page = page.replace('<script src="website/website.js"></script>', '\n'.join(scripts))
    page = page.replace('<h1>Game & circuit replay</h1>', '<h1>Live robust baseline</h1>')
    assert 'liveBaseline' in page and 'website/website.js' not in page
    (ROOT/'index.html').write_text(page)


if __name__ == '__main__':
    main()


def final_drops_video(attempt=2, drops=10, fps=6):
    """Encode the last recorded drops of one attempt, every recorded frame in timestamp order."""
    import imageio_ffmpeg
    import numpy as np
    from PIL import ImageDraw

    record = RUNS/f'attempt-{attempt}'/'record'
    meta = json.loads((record/'frames.json').read_text())
    x0, y0, x1, y1 = meta['board_bounds']
    last = max(f['drop'] for f in meta['frames'])
    frames = [f for f in meta['frames'] if f['drop'] > last-drops]
    end = next(RUNS/f'attempt-{attempt}'/n for n in ('end.png', 'paused.png', 'time-cap.jpg')
               if (RUNS/f'attempt-{attempt}'/n).exists())
    crop = (x0-200, y0-120, x1+20, y1+20)
    size = ((crop[2]-crop[0])//2*2, (crop[3]-crop[1])//2*2)
    out = ROOT/f'attempt-{attempt}-final-drops.mp4'
    writer = imageio_ffmpeg.write_frames(str(out), size, fps=fps, macro_block_size=2, quality=6)
    writer.send(None)
    shots = [(f['t'], f"drop {f['drop']} · {f['tag']} · t={f['t']:.2f}s", record/f['image']) for f in frames]
    shots += [(None, f'final capture ({end.name})', end)]*fps
    for _, label, path in shots:
        im = Image.open(path).convert('RGB').crop(crop).resize(size)
        ImageDraw.Draw(im).text((8, 8), label, fill='white', stroke_width=2, stroke_fill='black')
        writer.send(np.ascontiguousarray(np.asarray(im)))
    writer.close()
    return out, len(shots)
