"""Verify frozen-policy alignment and decode every committed video."""
import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe

from .game import Orchard
from .policy import np, restore
from .replay import ROOT


def unpack(path):
    encoded = path.read_text().split('=', 1)[1].strip().removesuffix(';')
    return json.loads(gzip.decompress(base64.b64decode(json.loads(encoded))))


def main():
    graph = json.loads((ROOT/'graph.json').read_text())
    manifest = json.loads((ROOT/'videos/manifest.json').read_text())
    assert unpack(ROOT/'replays/graph.js') == graph
    replay_manifest = (ROOT/'replays/manifest.js').read_text().split('=', 1)[1].strip().removesuffix(';')
    assert json.loads(replay_manifest) == manifest
    assert set(manifest) == {'drop5','drop1000','early','middle','final'}
    for label, metadata in manifest.items():
        checkpoint_path = Path(metadata['checkpoint'])
        with checkpoint_path.open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == metadata['checkpoint_sha256']
        policy, checkpoint = restore(checkpoint_path, graph)
        assert all(checkpoint[key] == metadata[key] for key in ('requested','drops','updates'))
        video = ROOT/'videos'/metadata['video']
        assert metadata['bytes'] == video.stat().st_size
        subprocess.run([get_ffmpeg_exe(), '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
        replay = unpack(ROOT/'replays'/f"{metadata['trace']}.js")
        rng = np.random.default_rng(metadata['action_rng_seed'])
        env = Orchard()
        observation, _ = env.reset(seed=metadata['fruit_seed'])
        decisions = replay['decisions']
        assert count_frames_and_secs(str(video))[0] == metadata['frames_per_drop'] * len(decisions) + 60
        for index, decision in enumerate(decisions):
            assert index+1 == decision['drop']
            assert index*metadata['frames_per_drop']/metadata['fps'] == decision['timestamp']
            np.testing.assert_array_equal(observation, decision['observation'])
            action, trace = policy.decide(observation, rng)
            assert action == decision['action']
            assert trace == {key: decision[key] for key in trace}
            assert env.score == decision['score_before']
            observation, _, terminated, truncated, info = env.step(action)
            assert info['score'] == decision['score']
            assert (terminated or truncated) == (index == len(decisions)-1)
        assert info == metadata['episode']
        print(f'{label}: {len(decisions)} aligned drops, {metadata["bytes"]:,} bytes')
    demo_path = ROOT/'demo/website.js'
    if demo_path.exists():
        demo = unpack(demo_path)
        records = json.loads((demo_path.parent/'manifest.json').read_text())
        policy, checkpoint = restore(Path(demo['meta']['checkpoint']), graph)
        assert checkpoint['drops'] == demo['meta']['training_drops']
        assert checkpoint['updates'] == demo['meta']['updates']
        rng = np.random.default_rng(demo['meta']['action_seed'])
        assert len(demo['decisions']) == records['completed_drops']
        for drop, (decision, row) in enumerate(zip(demo['decisions'], records['drops']), 1):
            assert (demo_path.parent/decision['image']).is_file()
            assert decision['drop'] == row['drop'] == drop
            assert decision['score'] == row['score']
            action, trace = policy.decide(np.array(decision['observation'], dtype=np.float32), rng)
            assert action == decision['action']
            assert trace == {key: decision[key] for key in trace}
        print(f'website: {len(demo["decisions"])} aligned drops')


if __name__ == '__main__':
    main()
