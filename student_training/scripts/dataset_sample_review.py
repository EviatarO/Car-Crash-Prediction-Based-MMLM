"""Pull 3 seeded-random (text, clip) samples per candidate reasoning dataset for human review.

Output: outputs/dataset_review_2026-10/<dataset>/<dataset>_<clipid>.txt + clip file(s) beside it,
plus README.md linking everything. Raw downloads go to dataset/public_samples/.

Read-only w.r.t. existing project data; downloads only the minimum (annotation files + 3 clips per set).
Usage:  python student_training/scripts/dataset_sample_review.py [--only caviar,mmau,vru,llava,bddx]
"""
import argparse
import io
import json
import random
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "public_samples"
OUT = ROOT / "outputs" / "dataset_review_2026-10"
HF = "https://huggingface.co/datasets/"
SEED = 0
K = 3


# ---------- helpers ----------
def wc(s):
    return len(str(s).split())


def write_video(frames_bgr, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    for codec in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if vw.isOpened():
            break
    for f in frames_bgr:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        vw.write(f)
    vw.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps, (n / fps if fps else float("nan"))


def write_txt(path, header, clips, blocks):
    """header: dict; clips: list of (label, Path); blocks: list of (title, text|None, note)."""
    lines = ["=" * 78]
    for k, v in header.items():
        lines.append(f"{k}: {v}")
    lines.append("=" * 78)
    lines.append("CLIP FILES (relative to this .txt):")
    for label, p in clips:
        n, fps, dur = video_info(p)
        rel = Path(p).relative_to(path.parent).as_posix()
        lines.append(f"  - {label}: {rel}   [{n} frames @ {fps:.1f} fps = {dur:.1f} s]")
    lines.append("")
    for title, text, note in blocks:
        lines.append(f"--- {title}" + (f"  ({wc(text)} words)" if text else "") + " ---")
        if note:
            lines.append(f"[{note}]")
        if text:
            lines.append(str(text).strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def pick(ids, k=K, seed=SEED):
    rnd = random.Random(seed)
    return rnd.sample(sorted(ids), k)


def get(url, dst):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dst)
    return dst


def fetch_nexar_full(dst, kind, vid):
    """Full Nexar mp4 from the gated HF repo; None if this account has no access (windows still written)."""
    from huggingface_hub import hf_hub_download

    if dst.exists():
        return dst
    try:
        p = hf_hub_download("nexar-ai/nexar_collision_prediction", f"train/{kind}/{vid}.mp4", repo_type="dataset")
    except Exception as e:
        print(f"[nexar] full clip {kind}/{vid} unavailable ({type(e).__name__}: HF access not granted)")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(p).read_bytes())
    return dst


# ---------- 1. CAViAR (Nexar part) + our teacher text ----------
def src_caviar():
    d = json.load(open(get("https://raw.githubusercontent.com/nec-labs-ma/CAViAR/main/data/test.json",
                           RAW / "caviar" / "test.json"), encoding="utf-8"))
    by_id = {x["video_path"]: x for x in d}
    tr = pd.read_csv(ROOT / "dataset" / "train.csv")
    train_ids = {f"{int(i):05d}" for i in tr["id"]}
    assert set(by_id) <= train_ids, "CAViAR ids outside Nexar train"
    print(f"[caviar] {len(by_id)} ids, all in Nexar train.csv (test sets use a separate id space)")

    v12 = {}
    for l in open(ROOT / "outputs/semantic_captions/Caption_V12_Neutral_1761.jsonl", encoding="utf-8"):
        r = json.loads(l)
        v12.setdefault(str(r["video_id"]).zfill(5), []).append(r)
    tr_idx = {f"{int(r.id):05d}": r for r in tr.itertuples()}

    for vid in pick(by_id):
        out = OUT / "caviar_nexar"
        clips = []
        clip = fetch_nexar_full(out / f"caviar_nexar_{vid}_full.mp4", "positive", vid)
        if clip:
            clips.append(("FULL Nexar clip (whole video, impact included)", clip))
        for tte in ("05", "10", "15"):
            fd = ROOT / "dataset" / "train" / f"{vid}_hires_tte{tte}"
            frs = [cv2.imread(str(f)) for f in sorted(fd.glob("frame_*.jpg"))]
            w = out / f"caviar_nexar_{vid}_window_tte{tte}.mp4"
            write_video(frs, w, 8)
            clips.append((f"2 s INPUT WINDOW ending {int(tte)/10:.1f} s before the event (what the model sees)", w))
        tr_r = tr_idx.get(vid)
        header = {
            "dataset": "CAViAR (Nexar part, human-annotated)",
            "clip_id": vid,
            "source": "https://github.com/nec-labs-ma/CAViAR (data/test.json); video: HF nexar-ai/nexar_collision_prediction",
            "license": "annotations: academic research only; videos: Nexar license",
            "nexar_time_of_event_s": getattr(tr_r, "time_of_event", "?"),
            "nexar_time_of_alert_s": getattr(tr_r, "time_of_alert", "?"),
            "label": "positive (Nexar target=1)",
            "text_covers": "the WHOLE video incl. the accident; written by humans, GPT-4 fixed grammar only",
        }
        blocks = []
        for qa in by_id[vid]["qa_pairs"]:
            if qa["benchmark"] in ("Weather & Light", "Road Conditions", "Accident Type"):
                blocks.append((f"{qa['benchmark']} | {qa['question']}", None, f"MCQ answer: {qa['answer']}"))
            else:
                blocks.append((f"{qa['benchmark']} | {qa['question']}", qa["answer"], ""))
        for r in sorted(v12.get(vid, []), key=lambda r: str(r["frames_dir"])):
            blocks.append((f"OUR V12 teacher caption, window {r['frames_dir']} (neutral, outcome words banned)",
                           r["caption_neutral"], f"agent={r.get('primary_agent')}"))
        if vid not in v12:
            blocks.append(("OUR V12 teacher caption", None, "none exists for this clip id in the 1,761-window pool"))
        write_txt(out / f"caviar_nexar_{vid}.txt", header, clips, blocks)

    # three Nexar NEGATIVES with our V12 text (what our "no crash" text looks like); CAViAR has none
    negs = sorted({v for v, rs in v12.items() if all(not r["event_occurs"] for r in rs)})
    for vid in pick(negs):
        out = OUT / "nexar_negative_ours"
        clips = []
        clip = fetch_nexar_full(out / f"nexar_negative_{vid}_full.mp4", "negative", vid)
        if clip:
            clips.append(("FULL Nexar clip (no crash)", clip))
        blocks = []
        for r in sorted(v12[vid], key=lambda r: str(r["frames_dir"])):
            fd = ROOT / "dataset" / "train" / str(r["frames_dir"])
            if fd.exists():
                frs = [cv2.imread(str(f)) for f in sorted(fd.glob("frame_*.jpg"))]
                if frs:
                    w = out / f"nexar_negative_{r['frames_dir']}.mp4"
                    write_video(frs, w, 8)
                    clips.append((f"2 s INPUT WINDOW {r['frames_dir']}", w))
            blocks.append((f"OUR V12 teacher caption, window {r['frames_dir']}", r["caption_neutral"], ""))
        header = {"dataset": "Nexar negative + OUR V12 teacher text (no human text exists)", "clip_id": vid,
                  "label": "negative (no crash)", "text_covers": "one 2 s window; AI-written (Gemini), blind"}
        write_txt(out / f"nexar_negative_{vid}.txt", header, clips, blocks)


# ---------- 2. MM-AU (CAP-DATA) ----------
def src_mmau():
    ann = RAW / "mmau" / "cap_text_annotations_conv.xlsx"  # converted from .xls via Excel COM (xlrd missing)
    d = pd.read_excel(ann, sheet_name="annotation file")
    d["video"] = d["video"].astype(int)
    rows = {int(r["video"]): r for r in d.to_dict("records")}
    url = HF + "JeffreyChou/MM-AU/resolve/main/CAP-DATA_chunks/1-10/1-10.part_aa"
    resp = urllib.request.urlopen(url)
    tf = tarfile.open(fileobj=resp, mode="r|gz")
    seqs, cur, cur_id = [], {}, None
    N_SEQ = 30  # stream only the first ~30 complete sequences (~1 GB), reservoir-sample 3 of them
    rnd = random.Random(SEED)
    keep = []  # reservoir of (seq_id, {name: bytes})
    n_seen = 0

    def flush(sid, frames):
        nonlocal n_seen
        if sid is None or not frames:
            return
        n_seen += 1
        if len(keep) < K:
            keep.append((sid, frames))
        else:
            j = rnd.randint(0, n_seen - 1)
            if j < K:
                keep[j] = (sid, frames)

    for m in tf:
        parts = m.name.split("/")
        if m.isfile() and len(parts) >= 6 and parts[-2] == "images":
            sid = parts[-3]
            if sid != cur_id:
                flush(cur_id, cur)
                cur_id, cur = sid, {}
                if n_seen >= N_SEQ:
                    break
            cur[parts[-1]] = tf.extractfile(m).read()
    else:
        flush(cur_id, cur)
    print(f"[mmau] streamed {n_seen} sequences, kept {[k for k, _ in keep]}")
    for sid, frames in keep:
        vid = int(sid)
        r = rows.get(vid)
        imgs = [cv2.imdecode(np.frombuffer(frames[n], np.uint8), cv2.IMREAD_COLOR) for n in sorted(frames)]
        out = OUT / "mmau_cap"
        clip = out / f"mmau_cap_{sid}.mp4"
        write_video(imgs, clip, 10)
        txt = str(r["texts"]).replace("[CLS]", "").replace("[SEP]", "").strip() if r else None
        header = {
            "dataset": "MM-AU (CAP-DATA part), human-annotated",
            "clip_id": sid,
            "source": "https://huggingface.co/datasets/JeffreyChou/MM-AU (CC BY-NC 4.0; GitHub README says academic use)",
            "frames_in_sequence": len(imgs),
            "annotation_timing(frames)": (f"abnormal_start(tai)={r['abnormal start frame']}, accident(tco)={r['accident frame']}, "
                                          f"abnormal_end(tae)={r['abnormal end frame']}, total={r['total frames']}") if r else "no row",
            "note_fps": "fps is not given in the annotation; clip written at 10 fps",
            "text_covers": "the whole sequence (short label, cause, prevention measure)",
        }
        blocks = []
        if r:
            blocks += [("texts (accident-type description)", txt, ""),
                       ("causes", str(r["causes"]).strip(), ""),
                       ("measures (prevention)", str(r["measures"]).strip(), "")]
        write_txt(out / f"mmau_cap_{sid}.txt", header, [("full sequence", clip)], blocks)


# ---------- 3. VRU-Accident (dense captions on CAP/DADA/DoTA/manual clips) ----------
def src_vru():
    from huggingface_hub import HfFileSystem

    B = HF + "kyh9191/VRU-Accident/resolve/main/"
    pool, seen = {}, {}
    for part in ("CAP_DATA", "DADA_2000", "DoTA", "MANUAL_DATA"):
        j = json.load(open(get(B + f"{part}_Dense_Caption.json", RAW / "vru" / f"{part}_Dense_Caption.json"), encoding="utf-8"))
        for path, v in j.items():
            g = v["gt"]
            if g in seen:  # DoTA/CAP share some identical captions -> skip duplicates, report
                print(f"[vru] duplicate caption {path} == {seen[g]}")
                continue
            seen[g] = path
            pool[path] = g
    fs = HfFileSystem()
    z = zipfile.ZipFile(fs.open("datasets/kyh9191/VRU-Accident/VRU_videos.zip", "rb"))
    for path in pick(pool):
        member = path.replace("./", "") + ".mp4"
        part, name = member.split("/")[-2], member.split("/")[-1][:-4]
        out = OUT / "vru_accident"
        clip = out / f"vru_{part}_{name}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(z.read(member))
        header = {
            "dataset": "VRU-Accident dense caption (creation method undocumented)",
            "clip_id": f"{part}/{name}",
            "source": "https://huggingface.co/datasets/kyh9191/VRU-Accident (Apache-2.0)",
            "text_covers": "the whole video incl. the collision",
            "note": "clips are pedestrian/cyclist crashes only",
        }
        write_txt(out / f"vru_{part}_{name}.txt", header, [("full video", clip)], [("dense caption (gt)", pool[path], "")])


# ---------- 4. LLaVA-Video-178K (generic video, optional Wave-3) ----------
def src_llava():
    caps = json.load(open(get(HF + "lmms-lab/LLaVA-Video-178K/resolve/main/0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_cap_processed.json",
                              RAW / "llava_video" / "0_30_s_youtube_v0_1_cap_processed.json"), encoding="utf-8"))
    cap_by_id = {c["id"]: c for c in caps}
    resp = urllib.request.urlopen(HF + "lmms-lab/LLaVA-Video-178K/resolve/main/0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz")
    tf = tarfile.open(fileobj=resp, mode="r|gz")
    cand = []
    for m in tf:
        if not m.isfile():
            continue
        vid = Path(m.name).stem.replace("ytb_", "")
        if vid in cap_by_id:
            cand.append((vid, tf.extractfile(m).read()))
        if len(cand) >= 40:
            break
    rnd = random.Random(SEED)
    for vid, data in rnd.sample(cand, K):
        out = OUT / "llava_video_178k"
        clip = out / f"llava_{vid}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(data)
        c = cap_by_id[vid]
        convs = c["conversations"]
        header = {"dataset": "LLaVA-Video-178K (GPT-4o synthetic caption; NOT driving)", "clip_id": vid,
                  "source": "https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K (academic use)",
                  "text_covers": "whole short video"}
        blocks = [(f"{t['from']} turn", t["value"].replace("<image>\n", ""), "") for t in convs]
        write_txt(out / f"llava_{vid}.txt", header, [("full video", clip)], blocks)


# ---------- 5. BDD-X (local text only; clips need BDD100K registration) ----------
def src_bddx():
    f = next((ROOT / "dataset" / "BDD-X-Dataset").rglob("BDD-X-Annotations_v1.csv"))
    d = pd.read_csv(f)
    print("[bddx] columns:", d.columns.tolist()[:12])
    rows = d.dropna(how="all")
    rnd = random.Random(SEED)
    for i in rnd.sample(range(len(rows)), K):
        r = rows.iloc[i]
        out = OUT / "bddx_text_only"
        out.mkdir(parents=True, exist_ok=True)
        lines = ["=" * 78, "dataset: BDD-X (human, short). CLIP NOT AVAILABLE: S3 links 404; videos need the BDD100K download (registration).",
                 f"row_index: {i}", "=" * 78]
        for k, v in r.items():
            if isinstance(v, str) and v.strip() and "http" not in v:
                lines.append(f"{k}: {v}   ({wc(v)} words)")
            elif "url" in str(k).lower() or "video" in str(k).lower():
                lines.append(f"{k}: {v}")
        (out / f"bddx_row{i}.txt").write_text("\n".join(lines), encoding="utf-8")



# ---------- 6. MM-AU DADA part (Step 0 of the week-1 plan): clip + 4 encoder windows + r1 targets ----------
DADA_URL = HF + "JeffreyChou/MM-AU/resolve/main/DADA-2000_chunks/DADA2000.part_aa"
STRIDE, N_FR, FPS_SRC = 4, 16, 30


def _clean(t):
    return str(t).replace("[CLS]", "").replace("[SEP]", "").replace("\xa0", " ").strip()


def _dada_tables():
    """Sheet1 (text + timing, keyed by (type, video)) and the official ArA split rows (question + 5 options).
    DADA folder = accident-type id, sub-folder = video number inside it; (type, video) is unique (1,962)
    and every ArA row matches Sheet1 on it with identical accident frames (checked 2026-10-05)."""
    s = pd.read_excel(RAW / "mmau" / "dada_text_annotations.xlsx", sheet_name="Sheet1")
    for c in ("video", "type"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    ara = {}
    for split, f in (("train", "dada_trainData.csv"), ("val", "dada_valData.csv"), ("test", "dada_testData.csv")):
        d = pd.read_csv(RAW / "mmau_ara" / f, encoding="utf-8-sig")
        for r in d.to_dict("records"):
            ara[(int(r["type"]), int(r["video"]))] = {
                "split": split, "question": r["question"], "answer": int(r["answer"]),
                "options": [r[f"a{i}"] for i in range(5)]}
    return s, ara


def _window_indices(end):
    return [end - (N_FR - 1 - i) * STRIDE for i in range(N_FR)]


def _r1_targets(label, tte, event, cause, visible):
    """r1 targets, ground-truth fields only.
    Window visibility rule (2026-10-05): a crash window is used ONLY if the hazard has already started
    inside it (end >= t_ai + 8 frames). Otherwise it is EXCLUDED from both phases - not relabelled as
    no-crash, because a crash does follow.
    Phase 1 (alignment) = event + cause. Phase 2 (SFT) = one schema for crash and no-crash."""
    if label == 1 and not visible:
        excl = "EXCLUDED - hazard not yet visible in this window (window visibility rule)"
        return excl, excl
    if label == 1:
        ph1 = f"{event}; {cause}"
        ph2 = f"Collision: yes. Time to impact: about {tte:.1f} s. Event: {event}. Cause: {cause}."
    else:
        ph1 = "not used (Phase 1 trains on crash windows only)"
        ph2 = "Collision: no. Time to impact: none in view. Event: normal driving before the incident. Cause: none."
    return ph1, ph2


def src_dada():
    tab, ara = _dada_tables()
    rows = {(int(r["type"]), int(r["video"])): r for r in tab.to_dict("records")}
    resp = urllib.request.urlopen(DADA_URL)
    tf = tarfile.open(fileobj=resp, mode="r|gz")
    rnd = random.Random(SEED)
    state = {"n_elig": 0}
    keep = []

    def flush(key, frames):
        if key is None or key not in rows:
            return
        r = rows[key]
        try:
            tai, tco = int(r["abnormal start frame"]), int(r["accident frame"])
        except (TypeError, ValueError):
            return
        if int(r["whether an accident occurred (1/0)"]) != 1:
            return
        need = set(_window_indices(tai - round(0.5 * FPS_SRC)))
        for tte in (0.5, 1.0, 1.5):
            need |= set(_window_indices(tco - round(tte * FPS_SRC)))
        if min(need) < 1 or not need <= set(frames):
            return
        state["n_elig"] += 1
        if len(keep) < K:
            keep.append((key, frames))
        else:
            j = rnd.randint(0, state["n_elig"] - 1)
            if j < K:
                keep[j] = (key, frames)

    N_ELIG = 12  # stream until 12 eligible sequences were seen, reservoir-sample 3 of them
    cur_key, cur = None, {}
    for m in tf:
        parts = m.name.split("/")
        if not (m.isfile() and len(parts) >= 4 and parts[-2] == "images"):
            continue
        try:
            key = (int(parts[-4]), int(parts[-3]))
        except ValueError:
            continue
        if key != cur_key:
            flush(cur_key, cur)
            cur_key, cur = key, {}
            if state["n_elig"] >= N_ELIG:
                break
        cur[int(parts[-1].split(".")[0])] = tf.extractfile(m).read()
    else:
        flush(cur_key, cur)
    print(f"[dada] eligible sequences streamed: {state['n_elig']}, kept {[k for k, _ in keep]}")

    out = OUT / "mmau_dada"
    for (typ, vid), frames in keep:
        r = rows[(typ, vid)]
        tai, tco, tae, tot = (int(r["abnormal start frame"]), int(r["accident frame"]),
                              int(r["abnormal end frame"]), int(r["total frames"]))

        def dec(i):
            return cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR)

        def half(im):
            return cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))

        name = f"mmau_dada_t{typ:02d}_v{vid:03d}"
        full = out / f"{name}_full.mp4"
        write_video([half(dec(i)) for i in sorted(frames)], full, FPS_SRC)
        clips = [("FULL clip, 30 fps, half resolution (whole video incl. the crash)", full)]
        event, cause = _clean(r["texts"]), str(r["causes"]).strip()
        wins = [("pos", 0.5, tco - round(0.5 * FPS_SRC)), ("pos", 1.0, tco - round(1.0 * FPS_SRC)),
                ("pos", 1.5, tco - round(1.5 * FPS_SRC)), ("neg", 0.5, tai - round(0.5 * FPS_SRC))]
        blocks = []
        for kind, tte, end in wins:
            idx = _window_indices(end)
            tag = f"tte{int(tte * 10):02d}" if kind == "pos" else "neg_pre_tai"
            w = out / f"{name}_window_{tag}.mp4"
            write_video([half(dec(i)) for i in idx], w, FPS_SRC / STRIDE)
            rule = f"collision frame - {tte} s" if kind == "pos" else "abnormal start - 0.5 s"
            clips.append((f"2 s window ({rule}): end frame {end}, frames {idx[0]}..{idx[-1]} step {STRIDE} [7.5 fps, original aspect]", w))
            if tag in ("tte10", "neg_pre_tai"):  # what the encoder really sees: full frame squashed to 256x256
                e = out / f"{name}_encoder256_{tag}.mp4"
                write_video([cv2.resize(dec(i), (256, 256), interpolation=cv2.INTER_AREA) for i in idx], e, FPS_SRC / STRIDE)
                clips.append(("  same window as the ENCODER sees it (full frame squashed to 256x256)", e))
            visible = end >= tai + 8
            ph1, ph2 = _r1_targets(1 if kind == "pos" else 0, tte, event, cause, visible)
            vis_txt = (f"hazard visible in window: {visible} (end {end} vs t_ai+8 = {tai + 8})" if kind == "pos" else "no-crash window, before the abnormal start")
            blocks.append((f"r1 TARGET, window {tag}  ({vis_txt})", None,
                           f"Phase 1 (alignment): {ph1}\n"
                           f"Phase 2 (SFT): {ph2}"))
        a = ara.get((typ, vid))
        header = {
            "dataset": "MM-AU / DADA-2000 part (human labels from a fixed list)",
            "clip_id": f"type {typ} / video {vid:03d}  (DADA folder {typ}/{vid:03d})",
            "official ArA split": a["split"] if a else "?",
            "source": "https://huggingface.co/datasets/JeffreyChou/MM-AU (CC BY-NC 4.0; README: academic use)",
            "timing (frames @30fps)": (f"abnormal start t_ai={tai} | collision t_co={tco} | abnormal end t_ae={tae} | "
                                       f"total={tot}  (t_ai->t_co = {(tco - tai) / 30:.1f} s)"),
            "raw codes (NOT used in targets)": (f"weather={r['weather(sunny,rainy,snowy,foggy)1-4']} light={r['light(day,night)1-2']} "
                                                f"scene={r['scenes(highway,tunnel,mountain,urban,rural)1-5']} "
                                                f"road={r['linear(arterials,curve,intersection,T-junction,ramp) 1-5']} accident-type id={typ}"),
        }
        gt = [("GT texts (event / accident-type sentence)", event, ""),
              ("GT causes", cause, ""),
              ("GT measures (prevention advice) - NOT used in targets, not visible", str(r["measures"]).strip(), "")]
        if a:
            opts = "\n".join(f"  {'*' if i == a['answer'] else ' '} a{i}: {o}" for i, o in enumerate(a["options"]))
            gt.append((f"ArA question: {a['question']}  (* = correct option; evaluation only, chance 20%)", None, "\n" + opts))
        write_txt(out / f"{name}.txt", header, clips, gt + blocks)


SOURCES = {"dada": src_dada, "caviar": src_caviar, "mmau": src_mmau, "vru": src_vru, "llava": src_llava, "bddx": src_bddx}


def build_readme():
    lines = ["# Dataset sample review (2026-10)", "",
             "Seeded random picks (seed=0), 3 per dataset. One .txt per clip + the clip file(s) beside it.", ""]
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        lines.append(f"## {d.name}")
        for t in sorted(d.glob("*.txt")):
            vids = sorted(v.name for v in d.glob(t.stem + "*.mp4"))
            lines.append(f"- [{t.name}]({d.name}/{t.name}) — " + ", ".join(f"[{v}]({d.name}/{v})" for v in vids))
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SOURCES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.only.split(","):
        print(f"== {s}")
        SOURCES[s]()
    build_readme()
    print("done ->", OUT)
