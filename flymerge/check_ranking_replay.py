"""Verify held-out gameplay, recorded afterstate activity, and video frames."""
import hashlib
import json

from imageio_ffmpeg import count_frames_and_secs

from .planner_experiment import CHECKPOINT
from .ranking_data import ROOT,SHOWCASE
from .ranking_eval import components
from .ranking_video import Lane,example,summary
from .verify import unpack


def main():
    data = unpack(ROOT/'comparison.js')
    metadata = data['meta']
    assert metadata == json.loads((ROOT/'comparison.json').read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == metadata['sha256']
    for kind,digest in metadata['ranker_sha256'].items():
        assert hashlib.sha256((ROOT/f'{kind}-ranker.pt').read_bytes()).hexdigest() == digest
    size = max(len(data['lanes'][k]) for k in ('A','learned'))
    frames = size*metadata['frames_per_drop']+2*metadata['fps']
    assert (ROOT/'comparison.mp4').stat().st_size == metadata['bytes']
    assert count_frames_and_secs(str(ROOT/'comparison.mp4'))[0] == frames
    assert metadata['illustration'] == example(data['lanes']['learned'])
    result = json.loads((ROOT/'evaluation.json').read_text())
    pipeline = components()
    for key,records in data['lanes'].items():
        method = {'A':'robust','learned':metadata['model'],'flywire':'flywire'}[key]
        lane = Lane(method)
        for record in records:
            lane.step(pipeline)
            assert lane.records[-1] == record
        expected = next(r['extended'] for r in result['episodes'] if r['method']==method
                        and r['profile']=='joint' and r['seed']==SHOWCASE)
        assert lane.ended and summary(lane) == expected == metadata['episodes'][key]
        print(method,len(lane.records),'aligned drops, scores and recorded traces',flush=True)


if __name__ == '__main__':
    main()
