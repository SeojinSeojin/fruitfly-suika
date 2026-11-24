"""Verify and package the spawn-contact website attempts."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .live_package import final_drops_video
from .planner_experiment import CHECKPOINT, load, logp
from .replay import json_file
from .robust_planner import choose
from .rollout_transfer import reconstruct
from .spawn_contact import gate, label_actions
from . import live_package
from .verify import unpack

RUNS = Path('runs/spawn-live')
ROOT = Path('artifacts/spawn-contact/live')
TEMPLATE = Path('artifacts/planner/website.html')


def verify(folder, policy):
    capture = unpack(folder/'website.js')
    meta, decisions = capture['meta'], capture['decisions']
    box = tuple(meta['board_bounds'])
    rng = np.random.default_rng(meta['action_seed'])
    blocked = uncertain = band_boards = 0
    highest = 0
    overrides, first_watermelon = [], None
    for d in decisions:
        obs = np.asarray(d['observation'], np.float32)
        state = rng.bit_generator.state
        legacy, _ = choose(obs, logp(policy, obs), rng)
        assert legacy == d['legacy_action'], (folder, d['drop'])
        body, count, current, _ = reconstruct(obs)
        fruits = [(float(body[i, 0]), float(body[i, 1]), int(body[i, 4])) for i in range(count)]
        fruits += [(x, y, t) for x, y, t in d['band']]
        labels = label_actions(fruits, current, box)
        assert [l['label'] for l in labels] == d['contact_labels'], (folder, d['drop'])
        block, allowed = gate(labels, legacy)
        assert block == d['blocked']
        if block:
            again = np.random.default_rng()
            again.bit_generator.state = state
            assert choose(obs, logp(policy, obs), again, allowed=allowed)[0] == d['action']
        else:
            assert d['action'] == legacy
        blocked += block
        if block:
            overrides.append(dict(drop=d['drop'], current_tier=current+1, original_aim=legacy+1,
                                  replacement_aim=d['action']+1, original_contact=d['legacy_contact'],
                                  replacement_contact=d['contact'],
                                  labels={k: [i+1 for i, l in enumerate(d['contact_labels']) if l == k]
                                          for k in ('fatal', 'uncertain', 'clear')},
                                  score_before=d.get('score_before'), score_after=d.get('score'),
                                  outcome_note='the overridden aim was not executed; its outcome is unknown'))
        if first_watermelon is None and any(t == 10 for *_, t in fruits):
            first_watermelon = d['drop']
        uncertain += d['contact']['label'] == 'uncertain'
        band_boards += bool(d['band'])
        highest = max(highest, max([t for *_, t in fruits], default=0))
    return dict(outcome=meta['outcome'], final_score=meta['final_score'], drops=len(decisions),
                seconds=round(meta.get('seconds') or 0), exceeded_3000=bool((meta['final_score'] or 0) > 3000),
                blocked_moves=blocked, executed_uncertain=uncertain, boards_with_band_fruit=band_boards,
                highest_detected_tier=highest+1, actions_reproduced=len(decisions),
                band_background_shots=meta.get('band_background_shots'),
                watermelon_first_detected_at_drop=first_watermelon,
                drops_after_watermelon_detected=None if first_watermelon is None else len(decisions)-first_watermelon,
                executed_uncertain_drops=[d['drop'] for d in decisions if d['contact']['label'] == 'uncertain'],
                final_contact=decisions[-1]['contact'] if decisions else None), overrides


def terminal_frames(source, target, report):
    """Raw final pre-drop (aim) frame and a strip of every recorded frame after the last release."""
    target.mkdir(parents=True, exist_ok=True)
    frames = json.loads((source/'record'/'frames.json').read_text())
    x0, y0, x1, y1 = frames['board_bounds']
    last = max(f['drop'] for f in frames['frames'])
    aim = [f for f in frames['frames'] if f['drop'] == last and f['tag'] == 'aim']
    if aim:
        shutil.copy2(source/'record'/aim[-1]['image'], target/'final-predrop.jpg')
        report['final_predrop_frame'] = 'final-predrop.jpg'
    after = [f for f in frames['frames'] if f['drop'] == last and f['tag'] != 'aim'][:8]
    crop = (x0-30, 100, x1+30, y1+10)
    tiles = [Image.open(source/'record'/f['image']).convert('RGB').crop(crop) for f in (aim[-1:]+after)]
    if tiles:
        w, h = tiles[0].size
        strip = Image.new('RGB', (len(tiles)*(w//2+6), h//2+22), '#162628')
        from PIL import ImageDraw
        pen = ImageDraw.Draw(strip)
        for i, (tile, f) in enumerate(zip(tiles, aim[-1:]+after)):
            strip.paste(tile.resize((w//2, h//2)), (i*(w//2+6), 22))
            pen.text((i*(w//2+6)+4, 4), f"{f['tag']} t={f['t']:.2f}s", fill='white')
        strip.save(target/'final-release-strip.jpg', quality=82, optimize=True)
        report['final_release_strip'] = 'final-release-strip.jpg'
        report['final_release_frames'] = [dict(tag=f['tag'], t=f['t']) for f in aim[-1:]+after]


def main():
    sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    policy = load()
    reports, captures, scripts, overrides = {}, {}, [], {}
    for k in (1, 2, 3):
        source, target = RUNS/f'attempt-{k}', ROOT/f'attempt-{k}'
        if not (source/'website.js').exists():
            continue
        reports[f'attempt-{k}'], overrides[f'attempt-{k}'] = verify(source, policy)
        terminal_frames(source, target, reports[f'attempt-{k}'])
        (target/'screens').mkdir(parents=True, exist_ok=True)
        for name in ('manifest.json', 'website.js', 'end.png', 'paused.png', 'time-cap.jpg'):
            if (source/name).exists():
                shutil.copy2(source/name, target/name)
        for image in sorted((source/'screens').glob('*.jpg')):
            with Image.open(image) as im:
                im.convert('RGB').resize((im.width//2, im.height//2)).save(target/'screens'/image.name, quality=72, optimize=True)
        live_package.RUNS, live_package.ROOT = RUNS, ROOT
        video, frames = final_drops_video(attempt=k)
        reports[f'attempt-{k}']['final_drops_video'] = str(video.relative_to(ROOT))
        meta = reports[f'attempt-{k}']
        captures[f'spawnLive{k}'] = dict(label=f'Attempt {k} · {meta["final_score"]} points', folder=f'attempt-{k}/')
        scripts.append(f'<script src="attempt-{k}/website.js"></script>')
        print(json.dumps(reports[f'attempt-{k}']))
    page = TEMPLATE.read_text()
    page = page.replace("window.siteCaptures={plannerWebsite:{label:'Perturbation-aware planner',folder:'website/'}};",
                        f'window.siteCaptures={json.dumps(captures)};')
    page = page.replace('<script src="website/website.js"></script>', '\n'.join(scripts))
    page = page.replace('<h1>Game & circuit replay</h1>', '<h1>Spawn-contact live attempts</h1>')
    page = page.replace('href="../replays/', 'href="../../replays/').replace('src="../replays/', 'src="../../replays/').replace('href="../demo/', 'href="../../demo/')
    (ROOT/'index.html').write_text(page)
    json_file(ROOT/'verification.json', dict(checkpoint=str(CHECKPOINT), sha256=sha, attempts=reports))
    json_file(ROOT/'override-ledger.json', dict(
        rule='block the robust choice only when the spawn-contact check labels it fatal and some aim is not fatal',
        overrides=overrides))


if __name__ == '__main__':
    main()
