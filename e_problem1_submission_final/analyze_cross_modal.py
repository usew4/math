"""Descriptive same-time-bin associations, with validity masks."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root=Path(__file__).parent/'output_final'
names=json.loads((root/'visual_feature_names.json').read_text(encoding='utf-8'))
pairs=[('text_compound','audio_log_rms'),('text_compound','audio_pitch_proxy'),
       ('text_compound','face_smile'),('audio_log_rms','face_smile'),
       ('audio_pitch_proxy','face_jaw_open')]
series={name:[] for pair in pairs for name in pair}
valid={name:[] for name in series}
for path in sorted((root/'features').glob('*.npz')):
    with np.load(path) as d:
        x=d['text']; a=d['audio']; v=d['visual']
        block={
          'text_compound':x[:,35],
          'audio_log_rms':a[:,13],
          'audio_pitch_proxy':a[:,16],
          'face_smile':(v[:,names.index('mouthSmileLeft')]+v[:,names.index('mouthSmileRight')])/2,
          'face_jaw_open':v[:,names.index('jawOpen')],
        }
        masks={
          'text_compound':x[:,36]>0,
          'audio_log_rms':d['speech_mask']>0,
          'audio_pitch_proxy':(d['speech_mask']>0)&(a[:,16]>0),
          'face_smile':d['mp_face_count']>0,
          'face_jaw_open':d['mp_face_count']>0,
        }
        for k in series:
            series[k].append(block[k]);valid[k].append(masks[k])
for k in series:
    series[k]=np.concatenate(series[k]);valid[k]=np.concatenate(valid[k])
rows=[]
for left,right in pairs:
    m=valid[left]&valid[right]
    a,b=series[left][m],series[right][m]
    rho=float(spearmanr(a,b).statistic) if len(a)>2 and len(set(a))>1 and len(set(b))>1 else float('nan')
    rows.append({'feature_1':left,'feature_2':right,'valid_aligned_bins':int(m.sum()),
                 'spearman_rho':round(rho,4)})
with (root/'cross_modal_associations.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:print(r)
