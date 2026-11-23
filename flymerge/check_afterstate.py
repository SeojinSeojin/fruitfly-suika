"""Verify afterstate videos, circuit traces and optional transfer captures."""
import hashlib
import json

from imageio_ffmpeg import count_frames_and_secs

from .afterstate_data import ROOT, SHOWCASE
from .afterstate_eval import METHODS, load_models
from .afterstate_video import LearnedLane, illustrate
from .planner_experiment import CHECKPOINT, overflow
from .planner_video import Lane
from .policy import np
from .verify import unpack


def main():
    policy,models = load_models()
    data = unpack(ROOT/'comparison.js')
    meta = data['meta']
    assert meta == json.loads((ROOT/'comparison.json').read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == meta['sha256']
    assert (ROOT/'comparison.mp4').stat().st_size == meta['bytes']
    for model,digest in meta['model_hashes'].items():
        assert hashlib.sha256((ROOT/f'{model}.pt').read_bytes()).hexdigest() == digest
    assert meta['illustration'] == illustrate(data['lanes']['learned'])
    frames = max(len(data['lanes'][k]) for k in ('A','learned'))*meta['frames_per_drop']+2*meta['fps']
    assert count_frames_and_secs(str(ROOT/'comparison.mp4'))[0] == frames
    evaluation = json.loads((ROOT/'evaluation.json').read_text())
    for key, records in data['lanes'].items():
        method = {'A':'A','learned':meta['model'],'flywire':'D'}[key]
        lane = Lane('hybrid','A') if method == 'A' else LearnedLane(method,models[METHODS[method]],meta['weights'][method])
        lane.obs,_ = lane.env.reset(seed=SHOWCASE)
        for record in records:
            lane.step(policy)
            assert lane.records[-1] == record
        actual = dict(**lane.env.info(),overflow=overflow(lane.env))
        expected = next(r['common'] for r in evaluation['rows'] if r['method']==method
                        and r['profile']==meta['profile'] and r['seed']==SHOWCASE)
        assert lane.ended and actual == expected == meta['episodes'][key]
        print(key,len(records),'aligned decisions, outcome and trace',flush=True)
    for folder in sorted(ROOT.glob('live-*')):
        capture = unpack(folder/'website.js')
        manifest = json.loads((folder/'manifest.json').read_text())
        model = models[manifest['strategy'].removeprefix('afterstate-')]
        assert len(capture['decisions']) == manifest['completed_drops']
        for i,(decision,row) in enumerate(zip(capture['decisions'],manifest['drops']),1):
            assert decision['drop'] == row['drop'] == i
            assert decision['score'] == row['score']
            assert (folder/decision['image']).is_file()
            obs = np.array(decision['observation'],dtype=np.float32)
            action, trace = model.decide(obs,manifest['future_weight'],manifest['overflow_penalty'],trace=True)
            assert action == decision['action']
            if 'steps' not in trace:
                _, extra = policy.decide(obs,None,'greedy')
                trace.update(extra)
            assert trace == {k:decision[k] for k in trace}
        if manifest.get('final_capture'):
            assert (folder/manifest['final_capture']).is_file()
        print(folder.name,manifest['completed_drops'],'aligned website decisions',flush=True)


if __name__ == '__main__':
    main()
