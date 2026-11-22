"""Verify planner replay alignment with the frozen policy and paired evaluation."""
import argparse
import hashlib
import json

from imageio_ffmpeg import count_frames_and_secs

from .planner_experiment import CHECKPOINT, ROOT, load, np, overflow
from .planner_video import Lane
from .replay import json_file
from .verify import unpack


def verify(local_only):
    policy = load()
    paired = unpack(ROOT/'comparison.js')
    metadata = paired['meta']
    assert metadata == json.loads((ROOT/'comparison.json').read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == metadata['sha256']
    frames = max(map(len, paired['lanes'].values()))*metadata['frames_per_drop']+2*metadata['fps']
    assert count_frames_and_secs(str(ROOT/metadata['video']))[0] == frames
    results = json.loads((ROOT/'results.json').read_text())
    for label, strategy in (('A','hybrid'), ('C','robust')):
        lane = Lane(strategy,label)
        for record in paired['lanes'][label]:
            lane.step(policy)
            assert lane.records[-1] == record
        expected = next(r['at_300'] for r in results['rows'] if r['method']==label
                        and r['seed']==metadata['seed'] and r['profile']==metadata['profile'])
        assert lane.ended and dict(**lane.env.info(),overflow=overflow(lane.env)) == expected
        assert metadata['episodes'][label] == expected
        print(f'{label}: {len(lane.records)} aligned local decisions')
    if local_only:
        return
    folder = ROOT/'website'
    capture = unpack(folder/'website.js')
    manifest = json.loads((folder/'manifest.json').read_text())
    assert len(capture['decisions']) == len(manifest['drops']) == manifest['completed_drops']
    assert capture['meta']['final_score'] == manifest['final_score']
    rng = np.random.default_rng(capture['meta']['action_seed'])
    site_actions = []
    for drop, (record, saved) in enumerate(zip(capture['decisions'], manifest['drops']),1):
        assert record['drop'] == saved['drop'] == drop
        assert record['score'] == saved['score']
        assert (folder/record['image']).is_file()
        obs = np.array(record['observation'],dtype=np.float32)
        action, trace = policy.decide(obs,rng,'robust')
        assert action == record['action']
        assert trace == {key:record[key] for key in trace}
        legacy, _ = policy.decide(obs,None,'hybrid',trace=False)
        site_actions.append(dict(drop=drop, executed=action, legacy=legacy))
    if manifest.get('final_capture'):
        assert (folder/manifest['final_capture']).is_file()
    json_file(ROOT/'website-actions.json', dict(decisions=site_actions,
              robust_rightmost=sum(row['executed']==14 for row in site_actions),
              legacy_rightmost=sum(row['legacy']==14 for row in site_actions),
              changed=sum(row['executed']!=row['legacy'] for row in site_actions)))
    print(f"Website: {manifest['completed_drops']} aligned decisions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only', action='store_true')
    verify(parser.parse_args().local_only)
