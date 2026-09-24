"""Structural and provenance checks for the completed 100-video Q1 release."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

root=Path(__file__).parent/'output_final'
with (root/'all_100_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    summary=list(csv.DictReader(f))
with (root/'word_alignment.csv').open(encoding='utf-8-sig',newline='') as f:
    words=list(csv.DictReader(f))
with (root/'face_frame_log.csv').open(encoding='utf-8-sig',newline='') as f:
    frames=list(csv.DictReader(f))
assert len(summary)==100
assert len({r['sample_id'] for r in summary})==100
assert len(list((root/'features').glob('*.npz')))==100
word_group=Counter(r['sample_id'] for r in words)
frame_group=Counter(r['sample_id'] for r in frames)
methods=Counter(r['method'] for r in words)
for r in summary:
    sid=r['sample_id']
    with np.load(root/'features'/f'{sid}.npz') as d:
        assert d['text'].shape==(50,37),sid
        assert d['audio'].shape==(50,17),sid
        assert d['visual'].shape==(50,65),sid
        assert d['bin_center_sec'].shape==(50,),sid
        assert d['mp_face_count'].shape==(50,),sid
        assert all(np.isfinite(d[k]).all() for k in d.files),sid
        assert np.all(np.diff(d['bin_center_sec'])>0),sid
        assert int(d['mp_face_count'].sum())==int(r['mp_face_frames']),sid
    assert word_group[sid]==int(r['word_count']),sid
    assert frame_group[sid]==int(r['sampled_frames']),sid
    assert int(r['forced_words'])==sum(x['method']=='pocketsphinx_forced' for x in words if x['sample_id']==sid),sid
    for x in (x for x in words if x['sample_id']==sid):
        assert 0<=float(x['start_sec'])<=float(x['end_sec'])<=float(r['duration_sec'])+.02,(sid,x)
report={
  'samples':len(summary),'word_tokens':len(words),'word_methods':dict(methods),
  'clips_with_any_forced_words':sum(int(r['forced_words'])>0 for r in summary),
  'clips_fully_forced':sum(int(r['forced_words'])==int(r['word_count']) for r in summary),
  'sampled_video_frames':len(frames),
  'frames_with_mp_face':sum(int(x['mp_face_detected']) for x in frames),
  'clips_with_any_mp_face':sum(int(r['mp_face_frames'])>0 for r in summary),
  'mean_mp_face_coverage':float(np.mean([float(r['mp_face_coverage']) for r in summary])),
  'feature_bytes':sum(p.stat().st_size for p in (root/'features').glob('*.npz')),
}
(root/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
