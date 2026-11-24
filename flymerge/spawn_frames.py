"""Before/after detection panels on archived raw frames (original detector left, improved right)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .spawn_replay import CACHE, LIVE, ROOT, Vision, band_background, detect_both, find_hand, load_rgb

CASES = [
    ('attempt-1', '199-02062-settle', 'missed-grape-attempt1-drop200', 'Attempt 1 · detection frame for drop 200'),
    ('attempt-2', '221-02330-settle', 'missed-grape-attempt2-drop222', 'Attempt 2 · detection frame for drop 222'),
    ('attempt-3', '182-01887-settle', 'watermelon-attempt3-after182', 'Attempt 3 · after drop 182, watermelon formed'),
    ('attempt-2', '054-00589-settle', 'melon-vs-phantoms-attempt2-drop55', 'Attempt 2 · detection frame for drop 55'),
    ('attempt-1', '030-00331-settle', 'ordinary-attempt1-drop31', 'Attempt 1 · ordinary board, drop 31'),
    ('attempt-3', '120-01240-settle', 'ordinary-attempt3-drop121', 'Attempt 3 · ordinary board, drop 121'),
]


def draw(image, found, box, color, band_color, label):
    x0, y0, x1, y1 = box
    out = Image.fromarray(image.copy())
    pen = ImageDraw.Draw(out)
    for x, y, tier, _ in found:
        sx, sy = x0+x*(x1-x0), y0+y*(y1-y0)
        c = band_color if y < 0 else color
        pen.ellipse((sx-8, sy-8, sx+8, sy+8), outline=c, width=3)
        pen.text((sx+10, sy-8), str(tier+1), fill=c, stroke_width=2, stroke_fill='black')
    pen.rectangle((x0-30, 60, x1+30, 86), fill='#172e2d')
    pen.text((x0-22, 66), label, fill='white')
    return out.crop((x0-30, 60, x1+30, y1+12))


def main():
    out = ROOT/'frames'
    out.mkdir(parents=True, exist_ok=True)
    legacy, improved = Vision(CACHE, 8), Vision(CACHE, 8, watermelon_inner=True)
    made = []
    backgrounds = {}
    for run, frame, slug, title in CASES:
        record = LIVE/run/'record'
        meta = json.loads((record/'frames.json').read_text())
        box = tuple(meta['board_bounds'])
        x0, y0, x1, y1 = box
        if run not in backgrounds:
            settle0 = [f for f in meta['frames'] if f['drop'] == 0 and f['tag'] == 'settle'][-1]
            empty = load_rgb(record/settle0['image'])
            early = [record/f['image'] for f in meta['frames'] if f['drop'] <= 8 and f['tag'] in ('settle', 'aim')]
            backgrounds[run] = empty[y0:y1, x0:x1].copy(), band_background(improved, empty, early, box)[0]
        background, extended = backgrounds[run]
        name = next(f['image'] for f in meta['frames'] if f['image'].endswith(frame+'.jpg'))
        pixels = load_rgb(record/name)
        hand = find_hand(improved, pixels, box)
        (old, cov_old), (new, cov_new) = detect_both(legacy, improved, pixels, box, background, extended, hand)
        left = draw(pixels, old, box, '#ffffff', '#ffffff', f'Original · {len(old)} fruit · coverage {cov_old:.0%}')
        right = draw(pixels, new, box, '#5ff0ff', '#ff5fd2', f'Improved · {len(new)} fruit · coverage {cov_new:.0%}')
        canvas = Image.new('RGB', (left.width*2+12, left.height+30), '#162628')
        canvas.paste(left, (0, 30))
        canvas.paste(right, (left.width+12, 30))
        ImageDraw.Draw(canvas).text((10, 8), title, fill='white')
        path = out/f'{slug}.jpg'
        canvas.save(path, quality=84, optimize=True)
        made.append(dict(case=slug, frame=f'{run}/record/{name}', image=str(path.relative_to(ROOT)),
                         original=len(old), improved=len(new), original_coverage=round(cov_old, 3),
                         improved_coverage=round(cov_new, 3)))
    (ROOT/'frames.json').write_text(json.dumps(made, indent=1)+'\n')
    print(json.dumps(made, indent=1))


if __name__ == '__main__':
    main()
