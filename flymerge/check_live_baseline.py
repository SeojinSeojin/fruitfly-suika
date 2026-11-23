"""Verify live-baseline captures against the frozen robust planner and summarise release cadence."""
import argparse
import hashlib
import json
from pathlib import Path

from .planner_experiment import CHECKPOINT, load, np
from .replay import json_file
from .verify import unpack

ROOT = Path('artifacts/live-baseline')
ARCHIVED_ROBUST = Path('artifacts/planner/website')
SHA256 = 'b2316d67a4538d55f2270633497dfe521a1dfd29dfc3ba0406ab8733380e1e7f'


def cadence(decisions):
    stamps = np.array([d['timestamp'] for d in decisions])
    gaps = np.diff(stamps)
    out = dict(decisions=len(decisions), release_interval_median_s=float(np.median(gaps)),
               release_interval_p95_s=float(np.percentile(gaps, 95)))
    waits = [d['post_release_wait'] for d in decisions if d.get('post_release_wait') is not None]
    if waits:
        out.update(post_release_wait_median_s=float(np.median(waits)), post_release_wait_max_s=float(np.max(waits)))
    settle = [d['ready_time']-d['release_time']-d['post_release_wait'] for d in decisions
              if d.get('ready_time') is not None and d.get('post_release_wait') is not None]
    if settle:
        out.update(readiness_after_wait_median_s=float(np.median(settle)))
    return out


def verify(folder, policy, frames_root=None):
    capture = unpack(folder/'website.js')
    manifest = json.loads((folder/'manifest.json').read_text())
    meta, decisions = capture['meta'], capture['decisions']
    assert len(decisions) == len(manifest['drops']) == manifest['completed_drops']
    assert meta['final_score'] == manifest['final_score'] and meta['strategy'] == 'robust'
    rng = np.random.default_rng(meta['action_seed'])
    for drop, (record, saved) in enumerate(zip(decisions, manifest['drops']), 1):
        assert record['drop'] == saved['drop'] == drop and record['score'] == saved['score']
        assert (folder/record['image']).is_file(), record['image']
        action, _ = policy.decide(np.array(record['observation'], dtype=np.float32), rng, 'robust', trace=False)
        assert action == record['action'], (folder, drop, action, record['action'])
    if manifest.get('final_capture'):
        assert (folder/manifest['final_capture']).is_file()
    report = dict(folder=str(folder), outcome=meta['outcome'], final_score=meta['final_score'],
                  drops=len(decisions), actions_reproduced=len(decisions), **cadence(decisions))
    if frames_root and (frames_root/'frames.json').exists():
        frames = json.loads((frames_root/'frames.json').read_text())['frames']
        report['recorded_frames'] = len(frames)
        report['frames_by_tag'] = {t: sum(f['tag'] == t for f in frames) for t in ('settle', 'aim', 'burst')}
        report['missing_frame_files'] = sum(not (frames_root/f['image']).is_file() for f in frames)
    return report


def main(args):
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA256
    policy = load()
    reports = {}
    for k in (1, 2, 3):
        folder = ROOT/f'attempt-{k}'
        if (folder/'website.js').exists():
            reports[f'attempt-{k}'] = verify(folder, policy, Path(args.runs)/f'attempt-{k}'/'record')
            print(json.dumps(reports[f'attempt-{k}']))
    archived = unpack(ARCHIVED_ROBUST/'website.js')['decisions']
    reports['archived-robust-1673'] = dict(folder=str(ARCHIVED_ROBUST), **cadence(archived))
    json_file(ROOT/'verification.json', dict(checkpoint=str(CHECKPOINT), sha256=SHA256, captures=reports))
    print(json.dumps(reports['archived-robust-1673']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='runs/live-baseline')
    main(parser.parse_args())
