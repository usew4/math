"""Add MediaPipe 52 facial blendshape coefficients to all Q1 clips."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from extract import DEFAULT_ZIP, N_STEPS, ROOT, imageio_ffmpeg, load_manifest, safe_name

sys.path.insert(0, str(ROOT / "vendor_mp14"))
sys.path.insert(1, str(ROOT / "vendor"))
import mediapipe as mp

BLEND_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"]


def run(source: Path, prior: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out/"features").mkdir(exist_ok=True)
    (out/"face_frame_log.csv").write_text("",encoding="utf-8")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=(ROOT/"face_landmarker.task").read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=True)
    summaries = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        with ZipFile(source) as z:
            rows = load_manifest(z)
            video_map = {f"{Path(n).parent.name}/{Path(n).stem}":n for n in z.namelist()
                         if n.lower().endswith(".mp4") and "附件1-" in n}
            for i,row in enumerate(rows,1):
                vid, clip = str(row["video_id"]),str(row["clip_id"])
                name=safe_name(vid,clip)
                with np.load(prior/"features"/f"{name}.npz") as old:
                    data={key:old[key].copy() for key in old.files}
                duration=float(data["bin_center_sec"][-1]*N_STEPS/(N_STEPS-.5))
                face_sum=np.zeros((N_STEPS,53),dtype=np.float32)
                face_count=np.zeros(N_STEPS,dtype=np.int16)
                frame_count=np.zeros(N_STEPS,dtype=np.int16)
                log=[]
                with tempfile.TemporaryDirectory(prefix="e_q1_video_") as td:
                    path=Path(td)/"clip.mp4"
                    with z.open(video_map[f"{vid}/{clip}"]) as src,path.open("wb") as dst:
                        shutil.copyfileobj(src,dst)
                    stream=imageio_ffmpeg.read_frames(str(path),output_params=["-vf","fps=2,scale=320:-2"])
                    try:
                        meta=next(stream)
                        w,h=meta["size"]
                        for j,raw in enumerate(stream):
                            t=(j+.5)/2
                            b=min(N_STEPS-1,int(t/max(duration,1e-6)*N_STEPS))
                            frame=np.asarray(Image.frombytes("RGB",(w,h),raw))
                            result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=frame))
                            found=bool(result.face_blendshapes)
                            if found:
                                scores={c.category_name:float(c.score) for c in result.face_blendshapes[0]}
                                face_sum[b,0]+=1
                                face_sum[b,1:]+=[scores.get(x,0.) for x in BLEND_NAMES]
                                face_count[b]+=1
                            frame_count[b]+=1
                            log.append({"sample_id":name,"time_sec":round(t,3),"bin":b,
                                        "mp_face_detected":int(found)})
                    finally:
                        stream.close()
                extra=np.zeros_like(face_sum)
                seen=frame_count>0
                extra[seen,0]=face_sum[seen,0]/frame_count[seen]
                valid=face_count>0
                extra[valid,1:]=face_sum[valid,1:]/face_count[valid,None]
                data["visual"]=np.concatenate([data["visual"],extra],axis=1)
                data["mp_face_count"]=face_count
                np.savez_compressed(out/"features"/f"{name}.npz",**data)
                summaries.append({"sample_id":name,"sampled_frames":len(log),
                                  "mp_face_frames":int(face_count.sum()),
                                  "mp_face_coverage":round(float(face_count.sum()/max(len(log),1)),4),
                                  "mp_face_bins":int(valid.sum())})
                with (out/"face_frame_log.csv").open("a",encoding="utf-8-sig",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(log[0]))
                    if i==1:writer.writeheader()
                    writer.writerows(log)
                print(f"[{i:3d}/100] {name}: face {int(face_count.sum())}/{len(log)} frames",flush=True)
    with (out/"face_coverage.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    for filename in ["word_alignment.csv","visual_frame_log.csv"]:
        shutil.copy2(prior/filename,out/filename)
    coverage={r["sample_id"]:r for r in summaries}
    with (prior/"all_100_summary.csv").open(encoding="utf-8-sig",newline="") as f:
        summary_rows=list(csv.DictReader(f))
    for r in summary_rows:
        r["visual_shape"]="50x65"
        r.update({k:v for k,v in coverage[r["sample_id"]].items() if k!="sample_id"})
    with (out/"all_100_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)
    meta=json.loads((prior/"run_metadata.json").read_text(encoding="utf-8"))
    meta.update({"visual":"12 baseline descriptors + 1 MediaPipe face detection rate + 52 face blendshapes",
                 "face_model":"MediaPipe Face Landmarker float16/1; mediapipe 0.10.14; sampled at 2 fps",
                 "face_feature_dim":65})
    (out/"run_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"visual_feature_names.json").write_text(json.dumps(
        [f"baseline_{i}" for i in range(12)]+["mp_face_rate"]+BLEND_NAMES,
       ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=DEFAULT_ZIP)
    ap.add_argument("--prior",type=Path,default=ROOT/"output_v2")
    ap.add_argument("--output",type=Path,default=ROOT/"output_final")
    args=ap.parse_args()
    run(args.source,args.prior,args.output)
