"""Evaluate checkpoints and render synchronized physics and circuit replays."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import base64
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont

from .policy import np, restore, torch
from .game import HEIGHT, WIDTH, TRANSFER_WIDTH, Orchard

COLORS = ('#ec786e', '#eb9a59', '#e5c356', '#8dbf77', '#61babc', '#7b9cdd',
          '#ad8ade', '#db88b2', '#ef9375', '#d7c881', '#a9ca71')
FPS = 30
FRAMES = 28
ROOT = Path('artifacts')
LABELS = ('drop5', 'drop1000', 'early', 'middle', 'final')


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def pack_script(path, key, data):
    payload = gzip.compress(json.dumps(data, separators=(',', ':')).encode(), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'window.replays[{json.dumps(key)}]="{base64.b64encode(payload).decode()}";\n')


def fruit(draw, x, y, r, tier):
    draw.ellipse((x-r, y-r, x+r, y+r), fill=COLORS[tier], outline='#f4eade', width=2)
    draw.arc((x-r*.67, y-r*.68, x+r*.67, y+r*.64), 195, 255, fill='#fff3d8', width=2)
    draw.text((x, y), str(tier + 1), anchor='mm', fill='#192829')


def paint(body, score, aim, current, upcoming, drop, metadata, env, terminal=False):
    shift = env.width - WIDTH
    side = env.width + 43
    image = Image.new('RGB', (584 + shift, 512), '#162628')
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 52, 20 + env.width, 52 + HEIGHT), radius=13, fill='#253e3d', outline='#83a29b', width=2)
    draw.line((21, 52 + env.overflow_y, 19 + env.width, 52 + env.overflow_y), fill='#b98975', width=1)
    draw.line((aim + 20, 57, aim + 20, 490), fill='#759990', width=1)
    draw.polygon([(aim+20, 65), (aim+14, 54), (aim+26, 54)], fill='#f7cc82')
    for x, y, tier in body:
        fruit(draw, x + 20, y + 52, env.radii[int(tier)], int(tier))
    draw.font = ImageFont.load_default(size=16)
    draw.text((20, 16), f'Orchard Drop  /  FlyWire FAFB v783', fill='#e7f0de')
    draw.text((side, 60), f'Drop {drop:03d}   Score {score}', fill='#f7cc82')
    draw.text((side, 110), 'Current        Next', fill='#e7f0de')
    fruit(draw, side+28, 162, 23, current)
    fruit(draw, side+119, 162, 23, upcoming)
    draw.text((side, 227), f'Training drops {metadata["drops"]:,}', fill='#e7f0de')
    draw.text((side, 259), f'PPO updates {metadata["updates"]}', fill='#e7f0de')
    if not metadata['updates']:
        draw.text((side, 290), 'Pre-update policy', fill='#f7cc82')
    if terminal:
        draw.rounded_rectangle((env.width/2-115, 223, env.width/2+155, 303), radius=10, fill='#122323')
        draw.text((20+env.width/2, 250), 'Episode complete', anchor='mm', fill='#f7cc82')
        draw.text((20+env.width/2, 279), f'Final score {score}', anchor='mm', fill='#e7f0de')
    return image


def episode(policy, seed, offset, capture=None, metadata=None, rules="original"):
    rng = np.random.default_rng(seed + offset)
    env = Orchard(capture=bool(capture), rules=rules)
    observation, _ = env.reset(seed=seed)
    decisions = []
    while True:
        current, upcoming = env.current, env.upcoming
        if capture:
            action, trace = policy.decide(observation, rng)
        else:
            with torch.no_grad():
                tensor = torch.from_numpy(observation[None])
                logits, _ = policy.heads(policy.features(tensor), tensor)
                p = logits[0].softmax(-1).numpy().astype(float)
            action = int(rng.choice(15, p=p / p.sum()))
        pre_score = env.score
        observation_after, _, terminal, truncation, info = env.step(action)
        if capture:
            decisions.append(dict(drop=env.drops, timestamp=(env.drops - 1) * FRAMES / FPS,
                                  observation=observation.tolist(), action=action,
                                  score_before=pre_score, score=env.score, **trace))
            for board, score in env.frames:
                image = paint(board, score, env.aim, current, upcoming, env.drops, metadata, env)
                capture.write(image.tobytes())
        observation = observation_after
        if terminal or truncation:
            if capture:
                image = paint(board, score, env.aim, current, upcoming, env.drops, metadata, env, True)
                for _ in range(60):
                    capture.write(image.tobytes())
            return info, decisions


def render(args, graph, protocol):
    started = time.perf_counter()
    pack_script(args.output / 'replays/graph.js', 'graph', graph)
    if args.output.resolve() != ROOT.resolve():
        template = (ROOT/'replays/index.html').read_text()
        viewer = os.path.relpath(ROOT/'replays', args.output/'replays')
        template = template.replace('href="viewer.css"', f'href="{viewer}/viewer.css"')
        template = template.replace('src="viewer.js"', f'src="{viewer}/viewer.js"')
        template = template.replace('../demo/website.js', '../demo-ad-filtered/website.js')
        if (args.output/'demo-game-over/website.js').exists():
            template = template.replace('<script src="../demo-ad-filtered/website.js"></script>',
                                        '<script src="../demo-ad-filtered/website.js"></script>\n'
                                        '<script src="../demo-game-over/website.js"></script>')
        (args.output/'replays/index.html').write_text(template)
    seed = protocol['video_training_seed']
    fruit_seed = protocol['video_fruit_seed']
    run = args.runs / f'flywire-{seed}'
    manifest_path = args.output / 'videos/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for label in args.labels:
        checkpoint_path = run / f'{label}.pt'
        policy, checkpoint = restore(checkpoint_path, graph)
        with checkpoint_path.open('rb') as source:
            checkpoint_digest = hashlib.file_digest(source, 'sha256').hexdigest()
        meta = {k: checkpoint[k] for k in ('kind', 'seed', 'requested', 'drops', 'updates')}
        rules = checkpoint.get('rules', 'original')
        width = 584 + (TRANSFER_WIDTH-WIDTH if rules == 'transfer' else 0)
        policy_digest = hashlib.sha256(b''.join(
            value.numpy().tobytes() for value in checkpoint['state'].values())).hexdigest()
        previous = next((entry for entry in manifest.values()
                         if (entry['drops'], entry['updates'], entry['seed'], entry['fruit_seed'])
                         == (meta['drops'], meta['updates'], seed, fruit_seed)
                         and entry.get('rules', 'original') == rules
                         and entry.get('policy_sha256') == policy_digest
                         and (args.output / 'videos' / entry['video']).is_file()
                         and (args.output / 'replays' / f"{entry['trace']}.js").is_file()), None)
        if previous:
            manifest[label] = {**previous, **meta, 'checkpoint': str(checkpoint_path),
                               'checkpoint_sha256': checkpoint_digest}
            continue
        path = args.output / 'videos' / f'{label}.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x512', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
               '-threads', '1', '-crf', '25', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ffmpeg:
            info, decisions = episode(policy, fruit_seed, protocol['evaluation_action_seed_offset'],
                                      ffmpeg.stdin, meta, rules)
            ffmpeg.stdin.close()
            if ffmpeg.wait():
                raise RuntimeError(f'Encoding {path} failed')
        subprocess.run([get_ffmpeg_exe(), '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)
        entry = dict(**meta, rules=rules, checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_digest,
                     policy_sha256=policy_digest, fruit_seed=fruit_seed,
                     action_rng_seed=fruit_seed + protocol['evaluation_action_seed_offset'],
                     pre_update=not bool(meta['updates']), video=path.name, trace=label,
                     bytes=path.stat().st_size, episode=info, render_and_decode_seconds=time.perf_counter()-started,
                     fps=FPS, frames_per_drop=FRAMES)
        manifest[label] = entry
        pack_script(args.output / 'replays' / f'{label}.js', label, dict(meta=entry, decisions=decisions))
        started = time.perf_counter()
    json_file(manifest_path, manifest)
    (args.output / 'replays/manifest.js').write_text('window.localManifest=' + json.dumps(manifest) + ';\n')


def evaluate_trial(run, graph, protocol):
    summary = json.loads((run / 'summary.json').read_text())
    rows = []
    for label in ('init', *LABELS):
        policy, ckpt = restore(run / f'{label}.pt', graph)
        for fruit_seed in protocol['held_out_fruit_seeds']:
            info, _ = episode(policy, fruit_seed, protocol['evaluation_action_seed_offset'],
                              rules=ckpt.get('rules', 'original'))
            rows.append(dict(kind=ckpt['kind'], seed=ckpt['seed'], milestone=label,
                             fruit_seed=fruit_seed, drops=ckpt['drops'], updates=ckpt['updates'], **info))
    return run.name, summary, rows


def evaluate(args, graph, protocol):
    started = time.perf_counter()
    rows, summaries, paired = [], {}, []
    kinds = ('flywire', 'rewired', 'mlp')
    trials = [args.runs / f'{kind}-{seed}' for kind in kinds for seed in protocol['training_seeds']]
    if args.eval_workers == 1:
        evaluated = [evaluate_trial(trial, graph, protocol) for trial in trials]
    else:
        with ProcessPoolExecutor(max_workers=args.eval_workers) as pool:
            jobs = [pool.submit(evaluate_trial, trial, graph, protocol) for trial in trials]
            evaluated = [job.result() for job in jobs]
    for name, summary, trial_rows in evaluated:
        summaries[name] = summary
        rows.extend(trial_rows)
    index = {(r['kind'], r['seed'], r['milestone'], r['fruit_seed']): r['score'] for r in rows}
    for kind in kinds[1:]:
        for seed in protocol['training_seeds']:
            for label in ('init', *LABELS):
                differences = [index['flywire', seed, label, f] - index[kind, seed, label, f]
                               for f in protocol['held_out_fruit_seeds']]
                paired.append(dict(comparison=f'flywire minus {kind}', seed=seed, milestone=label,
                                   mean=float(np.mean(differences)), differences=differences))
    output = args.output / 'results'
    json_file(output / 'evaluation.json', rows)
    json_file(output / 'paired.json', paired)
    json_file(output / 'training.json', summaries)
    json_file(output / 'evaluation-runtime.json', dict(seconds=time.perf_counter()-started))
    for run in summaries:
        for filename in ('learning.jsonl', 'episodes.jsonl'):
            destination = output / 'runs' / run / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((args.runs / run / filename).read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'render'])
    parser.add_argument('--eval-workers', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--labels', choices=LABELS, nargs='+', default=LABELS)
    args = parser.parse_args()
    if args.eval_workers < 1:
        parser.error('--eval-workers must be positive')
    graph = json.loads((ROOT / 'graph.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    (render if args.command == 'render' else evaluate)(args, graph, protocol)
