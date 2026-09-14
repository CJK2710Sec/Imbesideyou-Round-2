"""
Day 3 - Phase 6.11
Dataset B boundary inference + process discovery + automation analysis

Validated design carried forward from Day 2 / Phase 6:
    Dataset A
      -> 5F D_lean boundary candidate detector
      -> 5G confirmation features
      -> learned 5G logistic confirmation model
      -> boundary_probability >= 0.70

Dataset B
      -> same 5F detector
      -> same 5G confirmation features
      -> confirmation model trained ONLY on Dataset A
      -> confirmed B boundaries
      -> B-only unsupervised process discovery
      -> frequency / duration / workers / variants / interleaving
      -> automation priority ranking

Important:
Dataset B contains different processes and applications from Dataset A and has no
GT. Therefore this script deliberately DOES NOT force Dataset-A process labels
onto Dataset B. It discovers B process families from B segment behavior instead.

Run from repository root:
python day3_phase6_11_dataset_b_analysis_FINAL.py \
  --dataset-a "Dataset A/dataset_a" \
  --dataset-b "Dataset B/dataset_b" \
  --output "Outputs/Day 3/phase6_11"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS VERIFIED AGAINST DAY 2
# ---------------------------------------------------------------------------
D_LEAN_FEATURES = [
    "pre_event_rate",
    "post_event_rate",
    "pre_event_type_entropy",
    "post_event_type_entropy",
    "event_type_jaccard",
    "event_type_changed",
    "event_type_new_count",
    "event_type_disappeared_count",
    "layer_jaccard",
    "layer_changed",
    "layer_new_count",
    "layer_disappeared_count",
    "app_name_jaccard",
    "app_name_changed",
    "window_title_jaccard",
    "window_title_changed",
    "window_title_new_count",
]

CONFIRMATION_THRESHOLD = 0.70
SCORE_THRESHOLD = 0.70
SCORE_WINDOWS_SEC = [0.5, 1.0, 2.0, 3.0]
SCORE_STEP_MS = 100
MIN_PERSISTENCE_POINTS = 3
MIN_SEPARATION_SEC = 1.0
STATE_WINDOW_SEC = 1.0
STABILITY_WINDOW_SEC = 1.0
MATCH_TOLERANCE_SEC = 2.0

# Process-discovery bounds. B has no GT, so we avoid pretending to know K.
MIN_CLUSTERS = 2
MAX_CLUSTERS = 12
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------
def norm(v):
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def entropy(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    c = Counter(vals)
    n = len(vals)
    return float(-sum((x / n) * math.log2(x / n + 1e-12) for x in c.values()))


def iso_to_ms(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def ms_to_iso(ms):
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_float(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# RAW EVENTS
# ---------------------------------------------------------------------------
def load_session_events(session_dir: Path):
    records = []
    username_hashes = set()
    machine_ids = set()

    for event_file in sorted(session_dir.rglob("events.jsonl")):
        try:
            with event_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(e, dict):
                        continue
                    ts = e.get("timestamp_ms")
                    if ts is None:
                        continue
                    try:
                        ts = int(ts)
                    except Exception:
                        continue

                    source = e.get("source") or {}
                    if source.get("username_hash"):
                        username_hashes.add(str(source["username_hash"]))
                    if source.get("machine_id"):
                        machine_ids.add(str(source["machine_id"]))

                    ctx = e.get("context") or {}
                    app = ctx.get("active_app") or {}
                    browser = ctx.get("active_browser_tab") or {}

                    payload = e.get("payload") or {}
                    corr = e.get("correlation") or {}

                    # Keep a compact but useful representation for process discovery.
                    ui_tokens = []
                    for key in (
                        "element_id", "element_text", "element_name", "element_role",
                        "field_name", "control_type", "url", "title", "page_title",
                    ):
                        val = payload.get(key)
                        if val is None:
                            val = browser.get(key)
                        if val is not None:
                            s = str(val).strip()
                            if s:
                                ui_tokens.append(s[:200])

                    records.append({
                        "timestamp_ms": ts,
                        "event_type": norm(e.get("event_type")),
                        "layer": norm(e.get("layer")),
                        "app_name": norm(app.get("app_name")),
                        "process_name": norm(app.get("process_name")),
                        "window_title": norm(app.get("window_title")),
                        "browser_title": norm(browser.get("title")),
                        "browser_url": norm(browser.get("url")),
                        "ui_tokens": ui_tokens,
                        "username_hash": source.get("username_hash"),
                        "machine_id": source.get("machine_id"),
                        "gap_ms": safe_float(corr.get("ms_since_last_event"), 0.0),
                    })
        except Exception as exc:
            print(f"WARNING: could not read {event_file}: {exc}")

    if not records:
        return None

    records.sort(key=lambda x: x["timestamp_ms"])
    return {
        "timestamps": np.asarray([r["timestamp_ms"] for r in records], dtype=np.int64),
        "event_type": np.asarray([r["event_type"] for r in records], dtype=object),
        "layer": np.asarray([r["layer"] for r in records], dtype=object),
        "app_name": np.asarray([r["app_name"] for r in records], dtype=object),
        "process_name": np.asarray([r["process_name"] for r in records], dtype=object),
        "window_title": np.asarray([r["window_title"] for r in records], dtype=object),
        "browser_title": np.asarray([r["browser_title"] for r in records], dtype=object),
        "browser_url": np.asarray([r["browser_url"] for r in records], dtype=object),
        "ui_tokens": [r["ui_tokens"] for r in records],
        "username_hashes": username_hashes,
        "machine_ids": machine_ids,
        "records": records,
    }


def load_dataset_sessions(dataset_root: Path):
    sessions = {}
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root.resolve()}")
    dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("ses_"))
    if not dirs:
        raise RuntimeError(f"No ses_* directories found under {dataset_root.resolve()}")

    for i, d in enumerate(dirs, 1):
        print(f"  [{i}/{len(dirs)}] loading {d.name}", flush=True)
        data = load_session_events(d)
        if data is not None:
            sessions[d.name] = data
    return sessions


# ---------------------------------------------------------------------------
# 5F D_LEAN FEATURE ENGINE
# ---------------------------------------------------------------------------
def temporal_slice(session, center_ms, half_window_ms):
    ts = session["timestamps"]
    left = np.searchsorted(ts, center_ms - half_window_ms, side="left")
    right = np.searchsorted(ts, center_ms + half_window_ms, side="right")
    return left, right


def set_slice(arr, left, right):
    return {x for x in arr[left:right] if x is not None}


def calculate_dlean_features(session, center_ms, window_sec):
    ts = session["timestamps"]
    half = int(window_sec * 1000)

    pre_l = np.searchsorted(ts, center_ms - half, side="left")
    pre_r = np.searchsorted(ts, center_ms, side="left")
    post_l = np.searchsorted(ts, center_ms, side="left")
    post_r = np.searchsorted(ts, center_ms + half, side="right")

    pre_events = session["event_type"][pre_l:pre_r]
    post_events = session["event_type"][post_l:post_r]
    pre_layers = session["layer"][pre_l:pre_r]
    post_layers = session["layer"][post_l:post_r]
    pre_apps = session["app_name"][pre_l:pre_r]
    post_apps = session["app_name"][post_l:post_r]
    pre_titles = session["window_title"][pre_l:pre_r]
    post_titles = session["window_title"][post_l:post_r]

    pre_e = {x for x in pre_events if x is not None}
    post_e = {x for x in post_events if x is not None}
    pre_lay = {x for x in pre_layers if x is not None}
    post_lay = {x for x in post_layers if x is not None}
    pre_app = {x for x in pre_apps if x is not None}
    post_app = {x for x in post_apps if x is not None}
    pre_title = {x for x in pre_titles if x is not None}
    post_title = {x for x in post_titles if x is not None}

    def changed(a, b):
        return int(a != b)

    def new_count(a, b):
        return len(b - a)

    def disappeared_count(a, b):
        return len(a - b)

    denom = max(window_sec, 1e-6)
    return {
        "pre_event_rate": len(pre_events) / denom,
        "post_event_rate": len(post_events) / denom,
        "pre_event_type_entropy": entropy(pre_events),
        "post_event_type_entropy": entropy(post_events),
        "event_type_jaccard": jaccard(pre_e, post_e),
        "event_type_changed": changed(pre_e, post_e),
        "event_type_new_count": new_count(pre_e, post_e),
        "event_type_disappeared_count": disappeared_count(pre_e, post_e),
        "layer_jaccard": jaccard(pre_lay, post_lay),
        "layer_changed": changed(pre_lay, post_lay),
        "layer_new_count": new_count(pre_lay, post_lay),
        "layer_disappeared_count": disappeared_count(pre_lay, post_lay),
        "app_name_jaccard": jaccard(pre_app, post_app),
        "app_name_changed": changed(pre_app, post_app),
        "window_title_jaccard": jaccard(pre_title, post_title),
        "window_title_changed": changed(pre_title, post_title),
        "window_title_new_count": new_count(pre_title, post_title),
    }


def train_5f_model(repo: Path):
    feature_file = repo / "Outputs" / "Day 2" / "day2_boundary_vs_normal" / "boundary_vs_normal_features.csv"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing Day 2 5F training file: {feature_file}")

    df = pd.read_csv(feature_file)
    missing = [c for c in D_LEAN_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"5F training file is missing D_lean columns: {missing}")

    target_col = next((c for c in ["label", "is_boundary", "target", "y"] if c in df.columns), None)
    if target_col is None:
        raise RuntimeError("Could not identify the 5F target column in boundary_vs_normal_features.csv")

    X = df[D_LEAN_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("5F training target has fewer than two classes")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(Xs, y)
    return scaler, model


def score_session_5f(session, scaler, model):
    ts = session["timestamps"]
    start = int(ts[0])
    end = int(ts[-1])
    if end <= start:
        return pd.DataFrame(columns=["timestamp_ms", "mean_all_score"])

    points = np.arange(start, end + 1, SCORE_STEP_MS, dtype=np.int64)
    all_scores = []

    for window in SCORE_WINDOWS_SEC:
        X = np.asarray(
            [[safe_float(calculate_dlean_features(session, int(t), window).get(c)) for c in D_LEAN_FEATURES] for t in points],
            dtype=float,
        )
        probs = model.predict_proba(scaler.transform(X))[:, 1]
        all_scores.append(probs)

    scores = np.mean(np.vstack(all_scores), axis=0)
    return pd.DataFrame({"timestamp_ms": points, "mean_all_score": scores})


def persistent_candidates(score_df):
    if score_df.empty:
        return []
    scores = score_df["mean_all_score"].to_numpy(dtype=float)
    ts = score_df["timestamp_ms"].to_numpy(dtype=np.int64)
    above = scores >= SCORE_THRESHOLD
    regions = []
    start = None
    for i, ok in enumerate(above):
        if ok and start is None:
            start = i
        if (not ok or i == len(above) - 1) and start is not None:
            stop = i if ok and i == len(above) - 1 else i - 1
            if stop - start + 1 >= MIN_PERSISTENCE_POINTS:
                regions.append((start, stop))
            start = None

    peaks = []
    for a, b in regions:
        idx = a + int(np.argmax(scores[a:b + 1]))
        peaks.append(int(ts[idx]))

    peaks.sort()
    merged = []
    min_sep = int(MIN_SEPARATION_SEC * 1000)
    for p in peaks:
        if not merged or p - merged[-1] >= min_sep:
            merged.append(p)
        else:
            prev = merged[-1]
            prev_score = float(score_df.loc[score_df["timestamp_ms"] == prev, "mean_all_score"].iloc[0])
            new_score = float(score_df.loc[score_df["timestamp_ms"] == p, "mean_all_score"].iloc[0])
            if new_score > prev_score:
                merged[-1] = p
    return merged


def build_b_candidates(sessions, scaler, model, output_dir):
    score_rows = []
    candidate_rows = []
    for session_id, session in sessions.items():
        print(f"  5F scoring {session_id}", flush=True)
        score_df = score_session_5f(session, scaler, model)
        for r in score_df.itertuples(index=False):
            score_rows.append({
                "session_id": session_id,
                "timestamp_ms": int(r.timestamp_ms),
                "mean_all_score": float(r.mean_all_score),
            })
        for t in persistent_candidates(score_df):
            row = score_df.loc[score_df["timestamp_ms"] == t]
            score = float(row["mean_all_score"].iloc[0]) if not row.empty else 0.0
            candidate_rows.append({
                "session_id": session_id,
                "candidate_timestamp_ms": int(t),
                "boundary_score": score,
            })

    scores = pd.DataFrame(score_rows)
    candidates = pd.DataFrame(candidate_rows)
    scores.to_csv(output_dir / "phase6_11_5f_continuous_scores_B.csv", index=False)
    candidates.to_csv(output_dir / "phase6_11_5f_candidates_B.csv", index=False)
    return scores, candidates


# ---------------------------------------------------------------------------
# EXACT 5G CONFIRMATION FEATURE CALCULATION
# ---------------------------------------------------------------------------
def extract_state(session, center_ms, start_offset_sec, end_offset_sec):
    ts = session["timestamps"]
    start_ms = center_ms + int(start_offset_sec * 1000)
    end_ms = center_ms + int(end_offset_sec * 1000)
    left = np.searchsorted(ts, start_ms, side="left")
    right = np.searchsorted(ts, end_ms, side="right")

    event_types = session["event_type"][left:right]
    layers = session["layer"][left:right]
    apps = session["app_name"][left:right]
    processes = session["process_name"][left:right]
    titles = session["window_title"][left:right]
    return {
        "event_types": {x for x in event_types if x is not None},
        "layers": {x for x in layers if x is not None},
        "apps": {x for x in apps if x is not None},
        "processes": {x for x in processes if x is not None},
        "titles": {x for x in titles if x is not None},
        "event_list": list(event_types),
        "count": len(event_types),
    }


def state_change(pre, post):
    ej = jaccard(pre["event_types"], post["event_types"])
    lj = jaccard(pre["layers"], post["layers"])
    aj = jaccard(pre["apps"], post["apps"])
    pj = jaccard(pre["processes"], post["processes"])
    tj = jaccard(pre["titles"], post["titles"])
    ec, lc, ac, pc, tc = 1-ej, 1-lj, 1-aj, 1-pj, 1-tj
    structure = 0.35*ec + 0.25*lc + 0.20*tc + 0.10*ac + 0.10*pc
    return {
        "pre_event_count": pre["count"],
        "post_event_count": post["count"],
        "event_jaccard": ej,
        "layer_jaccard": lj,
        "app_jaccard": aj,
        "process_jaccard": pj,
        "title_jaccard": tj,
        "event_change": ec,
        "layer_change": lc,
        "app_change": ac,
        "process_change": pc,
        "title_change": tc,
        "structure_score": structure,
    }


def post_stability(session, center_ms):
    first = extract_state(session, center_ms, 0.0, STABILITY_WINDOW_SEC)
    second = extract_state(session, center_ms, STABILITY_WINDOW_SEC, 2.0)
    es = jaccard(first["event_types"], second["event_types"])
    ls = jaccard(first["layers"], second["layers"])
    aps = jaccard(first["apps"], second["apps"])
    ts = jaccard(first["titles"], second["titles"])
    return {
        "post_event_similarity": es,
        "post_layer_similarity": ls,
        "post_app_similarity": aps,
        "post_title_similarity": ts,
        "post_stability_score": 0.35*es + 0.25*ls + 0.20*aps + 0.20*ts,
    }


def score_persistence(scores, sid, candidate_ts):
    s = scores[scores["session_id"] == sid].sort_values("timestamp_ms")
    
    if s.empty:
        return {
            "score_at_candidate": 0.0,
            "score_pre_mean": 0.0,
            "score_post_mean": 0.0,
            "score_pre_max": 0.0,
            "score_post_max": 0.0,
            "score_persistence": 0.0,
            "persistence_points": 0,
            "persistence_sec": 0.0,
        }

    ts = s["timestamp_ms"].to_numpy(dtype=np.int64)
    scores_arr = s["mean_all_score"].to_numpy(dtype=float)

    idx = int(np.searchsorted(ts, candidate_ts))
    idx = min(max(idx, 0), len(ts) - 1)

    pre = scores_arr[
        (ts >= candidate_ts - 2000) &
        (ts < candidate_ts)
    ]

    post_mask = (
        (ts > candidate_ts) &
        (ts <= candidate_ts + 2000)
    )
    post = scores_arr[post_mask]

    # Number of sampled score points after the candidate
    # whose boundary score remains above the Day 2 threshold.
    persistence_points = int(np.sum(post >= SCORE_THRESHOLD))

    # Phase 5F samples continuously at 100 ms.
    # Convert persistent points to seconds using the actual sampling interval.
    persistence_sec = float(persistence_points * 0.1)

    return {
        "score_at_candidate": float(scores_arr[idx]),

        "score_pre_mean": (
            float(np.mean(pre)) if len(pre) else 0.0
        ),

        "score_post_mean": (
            float(np.mean(post)) if len(post) else 0.0
        ),

        "score_pre_max": (
            float(np.max(pre)) if len(pre) else 0.0
        ),

        "score_post_max": (
            float(np.max(post)) if len(post) else 0.0
        ),

        "score_persistence": (
            float(np.mean(post >= SCORE_THRESHOLD))
            if len(post) else 0.0
        ),

        "persistence_points": persistence_points,
        "persistence_sec": persistence_sec,
    }

def build_confirmation_features_B(candidates, scores, sessions):
    rows = []
    total = len(candidates)
    for i, (_, c) in enumerate(candidates.iterrows(), 1):
        if i == 1 or i % 100 == 0 or i == total:
            print(f"  5G confirmation features: {i:,}/{total:,}", flush=True)
        sid = c["session_id"]
        t = int(c["candidate_timestamp_ms"])
        session = sessions.get(sid)
        if session is None:
            continue
        pre = extract_state(session, t, -STATE_WINDOW_SEC, 0.0)
        post = extract_state(session, t, 0.0, STATE_WINDOW_SEC)
        sf = state_change(pre, post)
        st = post_stability(session, t)
        pf = score_persistence(scores, sid, t)
        original = min(max(safe_float(c["boundary_score"]), 0.0), 1.0)
        confirmation = (
            0.35 * original
            + 0.25 * pf["score_persistence"]
            + 0.25 * sf["structure_score"]
            + 0.15 * st["post_stability_score"]
        )
        rows.append({
            **c.to_dict(),
            **sf,
            **st,
            **pf,
            "score_evidence": original,
            "persistence_evidence": pf["score_persistence"],
            "structure_evidence": sf["structure_score"],
            "stability_evidence": st["post_stability_score"],
            "confirmation_score": confirmation,
        })
    return pd.DataFrame(rows)


def make_A_confirmation_labels(df):
    """Recreate the Day 2 one-to-one labels from the exact GT matching rule."""
    gt_file = Path("Outputs") / "Day 2" / "day2_gt_boundary_analysis" / "gt_switch_boundaries.csv"
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing Dataset A GT boundary file: {gt_file}")
    gt = pd.read_csv(gt_file)
    if "timestamp_ms" not in gt.columns:
        raise RuntimeError("GT boundary file has no timestamp_ms column")
    gt["timestamp_ms"] = pd.to_numeric(gt["timestamp_ms"], errors="coerce")
    gt = gt.dropna(subset=["timestamp_ms"]).copy()
    gt["timestamp_ms"] = gt["timestamp_ms"].astype(np.int64)

    labels = np.zeros(len(df), dtype=int)
    for sid in df["session_id"].unique():
        cidx = df.index[df["session_id"] == sid].tolist()
        cidx.sort(key=lambda i: int(df.loc[i, "candidate_timestamp_ms"]))
        gt_times = gt.loc[gt["session_id"] == sid, "timestamp_ms"].sort_values().to_numpy(dtype=np.int64)
        used = set()
        for idx in cidx:
            t = int(df.loc[idx, "candidate_timestamp_ms"])
            pos = int(np.searchsorted(gt_times, t))
            best = None
            best_err = None
            for off in range(-3, 4):
                j = pos + off
                if j < 0 or j >= len(gt_times) or j in used:
                    continue
                err = abs(t - int(gt_times[j]))
                if best_err is None or err < best_err:
                    best = j
                    best_err = err
            if best is not None and best_err <= MATCH_TOLERANCE_SEC * 1000:
                labels[df.index.get_loc(idx)] = 1
                used.add(best)
    return labels


def train_5g_model(repo: Path):
    feature_file = repo / "Outputs" / "Day 2" / "day2_boundary_confirmation" / "candidate_confirmation_features.csv"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing Day 2 confirmation features: {feature_file}")
    df = pd.read_csv(feature_file)
    labels = make_A_confirmation_labels(df)

    exclude = {
        "session_id", "candidate_timestamp_ms", "candidate_timestamp_sec",
        "region_start_ms", "region_end_ms", "label", "matched_gt_index",
        "gt_distance_ms", "confirmation_score", "boundary_probability",
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in exclude]
    if not features:
        raise RuntimeError("No numeric 5G confirmation features found")

    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = labels
    if len(np.unique(y)) < 2:
        raise RuntimeError("5G training labels contain fewer than two classes")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(Xs, y)
    return features, scaler, model, int(y.sum()), int((y == 0).sum())


def apply_5g_model(df, features, scaler, model):
    if df.empty:
        return df.copy()
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    out = df.copy()
    out["boundary_probability"] = model.predict_proba(scaler.transform(X))[:, 1]
    out["confirmed"] = out["boundary_probability"] >= CONFIRMATION_THRESHOLD
    return out


# ---------------------------------------------------------------------------
# SEGMENT CONSTRUCTION
# ---------------------------------------------------------------------------
def build_segments(confirmed_df, sessions):
    segments = []
    if confirmed_df.empty:
        return segments
    for sid, g in confirmed_df.groupby("session_id"):
        session = sessions.get(sid)
        if session is None:
            continue
        session_start = int(session["timestamps"][0])
        session_end = int(session["timestamps"][-1])
        boundaries = sorted(set(int(x) for x in g["candidate_timestamp_ms"].tolist()))
        # A confirmed boundary defines a transition. Segment 1 begins at session start,
        # segment i ends at the next confirmed boundary.
        cuts = [session_start] + [b for b in boundaries if session_start < b < session_end] + [session_end]
        # De-duplicate and remove zero-width intervals.
        cuts = sorted(set(cuts))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            segments.append({
                "session_id": sid,
                "start_ms": a,
                "end_ms": b,
                "start": ms_to_iso(a),
                "end": ms_to_iso(b),
            })
    return segments


# ---------------------------------------------------------------------------
# B-ONLY PROCESS DISCOVERY
# ---------------------------------------------------------------------------
def segment_tokens(session, start_ms, end_ms):
    ts = session["timestamps"]
    l = np.searchsorted(ts, start_ms, side="left")
    r = np.searchsorted(ts, end_ms, side="right")
    tokens = []

    def add(prefix, value):
        if value is None:
            return
        s = str(value).strip()
        if not s:
            return
        # Keep Japanese strings intact; strip volatile numeric IDs where possible.
        s = re.sub(r"\d{4,}", "#", s)[:180]
        tokens.append(f"{prefix}={s}")

    for i in range(l, r):
        add("evt", session["event_type"][i])
        add("layer", session["layer"][i])
        add("app", session["app_name"][i])
        add("proc", session["process_name"][i])
        add("win", session["window_title"][i])
        add("browser", session["browser_title"][i])
        # UI text/IDs are useful, but cap their influence by using only the first two.
        for ui in session["ui_tokens"][i][:2]:
            add("ui", ui)
    return tokens


def segment_numeric_features(session, start_ms, end_ms):
    ts = session["timestamps"]
    l = np.searchsorted(ts, start_ms, side="left")
    r = np.searchsorted(ts, end_ms, side="right")
    n = max(r-l, 0)
    duration_sec = max((end_ms-start_ms)/1000.0, 0.001)
    events = session["event_type"][l:r]
    apps = session["app_name"][l:r]
    layers = session["layer"][l:r]
    titles = session["window_title"][l:r]
    gaps = session["records"][l:r]
    return {
        "duration_sec": duration_sec,
        "event_count": n,
        "events_per_sec": n/duration_sec,
        "unique_event_types": len({x for x in events if x is not None}),
        "unique_apps": len({x for x in apps if x is not None}),
        "unique_layers": len({x for x in layers if x is not None}),
        "unique_window_titles": len({x for x in titles if x is not None}),
        "keystrokes": sum(1 for x in events if x == "keystroke"),
        "clicks": sum(1 for x in events if x in {"mouse_click", "mouse_double_click", "browser_click"}),
        "clipboard_changes": sum(1 for x in events if x == "clipboard_change"),
        "browser_navigation": sum(1 for x in events if x == "browser_navigation"),
        "form_inputs": sum(1 for x in events if x == "browser_form_input"),
        "screenshots": sum(1 for x in events if x == "screenshot_smart"),
        "app_entropy": entropy(apps),
        "event_entropy": entropy(events),
        "layer_entropy": entropy(layers),
        "long_gaps": sum(1 for rec in gaps if safe_float(rec.get("gap_ms"), 0.0) >= 2000),
    }


def choose_cluster_count(X, n):
    if n < 3:
        return 1, np.array([0]*n, dtype=int), np.nan
    upper = min(MAX_CLUSTERS, n-1)
    best = None
    best_labels = None
    best_score = -np.inf
    for k in range(MIN_CLUSTERS, upper+1):
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best = k
            best_labels = labels
    if best_labels is None:
        return 1, np.zeros(n, dtype=int), np.nan
    return int(best), best_labels, float(best_score)


def discover_processes(segments, sessions):
    if not segments:
        return segments, pd.DataFrame(), {"n_clusters": 0, "silhouette": np.nan}

    docs = []
    numeric_rows = []
    for seg in segments:
        session = sessions[seg["session_id"]]
        docs.append(" ".join(segment_tokens(session, seg["start_ms"], seg["end_ms"])))
        numeric_rows.append(segment_numeric_features(session, seg["start_ms"], seg["end_ms"]))

    # Character/word TF-IDF gives us a robust representation for Japanese text,
    # app names, event names and UI identifiers without requiring tokenization.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
    )
    X_sparse = vectorizer.fit_transform(docs)

    # Add normalized numeric behavior to the sparse process representation.
    num_df = pd.DataFrame(numeric_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    num = StandardScaler().fit_transform(num_df.to_numpy(dtype=float))
    # Convert TF-IDF to dense only after max_features cap; B is ~20k events and
    # segment count should remain manageable.
    X_text = X_sparse.toarray()
    text_norm = np.linalg.norm(X_text, axis=1, keepdims=True)
    X_text = X_text / np.maximum(text_norm, 1e-12)
    # Numeric features are deliberately down-weighted so process identity remains
    # primarily behavioral/contextual rather than duration-driven.
    X = np.hstack([X_text, 0.20 * num])
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(X_norm, 1e-12)

    k, labels, sil = choose_cluster_count(X, len(segments))

    # Stable cluster labels by first occurrence order, independent of sklearn's
    # arbitrary cluster numbering.
    mapping = {}
    next_id = 1
    for lab in labels:
        if int(lab) not in mapping:
            mapping[int(lab)] = f"BPROC_{next_id:02d}"
            next_id += 1

    enriched = []
    for seg, label, nums in zip(segments, labels, numeric_rows):
        row = dict(seg)
        row["label"] = mapping[int(label)]
        row.update(nums)
        enriched.append(row)

    summary_rows = []
    for label, g in pd.DataFrame(enriched).groupby("label"):
        summary_rows.append({
            "label": label,
            "executions": len(g),
            "total_duration_sec": float(g["duration_sec"].sum()),
            "mean_duration_sec": float(g["duration_sec"].mean()),
            "median_duration_sec": float(g["duration_sec"].median()),
            "p90_duration_sec": float(g["duration_sec"].quantile(0.90)),
            "mean_events": float(g["event_count"].mean()),
            "mean_events_per_sec": float(g["events_per_sec"].mean()),
            "mean_unique_apps": float(g["unique_apps"].mean()),
        })

    return enriched, pd.DataFrame(summary_rows).sort_values("executions", ascending=False), {
        "n_clusters": k,
        "silhouette": sil,
    }


# ---------------------------------------------------------------------------
# ANALYSIS / ROI PRIORITY
# ---------------------------------------------------------------------------
def build_process_analysis(segments, sessions):
    if not segments:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(segments)

    # Worker proxy: username_hash if available, otherwise machine_id, otherwise session.
    workers = defaultdict(set)
    for sid, s in sessions.items():
        worker_values = set(s["username_hashes"])
        if not worker_values:
            worker_values = set(s["machine_ids"])
        if not worker_values:
            worker_values = {sid}
        workers[sid] = worker_values

    process_rows = []
    pattern_rows = []
    session_rows = []
    confidence_rows = []

    for label, g in df.groupby("label"):
        worker_set = set()
        session_set = set(g["session_id"])
        for sid in session_set:
            worker_set |= workers[sid]
        duration_total = float(g["duration_sec"].sum())
        durations = g["duration_sec"]
        process_rows.append({
            "label": label,
            "execution_count": len(g),
            "session_count": len(session_set),
            "worker_proxy_count": len(worker_set),
            "total_duration_sec": duration_total,
            "total_duration_min": duration_total/60.0,
            "mean_duration_sec": float(durations.mean()),
            "median_duration_sec": float(durations.median()),
            "p90_duration_sec": float(durations.quantile(0.90)),
            "duration_cv": float(durations.std(ddof=0) / max(durations.mean(), 1e-9)),
            "mean_event_count": float(g["event_count"].mean()),
            "mean_unique_apps": float(g["unique_apps"].mean()),
        })

    # Process sequence patterns within each session.
    for sid, g in df.sort_values(["session_id", "start_ms"]).groupby("session_id"):
        labels = g["label"].tolist()
        transitions = Counter(zip(labels[:-1], labels[1:]))
        aba = sum(1 for i in range(1, len(labels)-1) if labels[i-1] == labels[i+1] and labels[i] != labels[i-1])
        for (a, b), count in transitions.items():
            pattern_rows.append({
                "session_id": sid,
                "from_label": a,
                "to_label": b,
                "count": count,
            })
        session_rows.append({
            "session_id": sid,
            "segment_count": len(g),
            "unique_processes": g["label"].nunique(),
            "aba_patterns": aba,
            "session_duration_sec": float(g["end_ms"].max() - g["start_ms"].min())/1000.0,
        })

    # Boundary/identity confidence summary: B has no GT, so report model evidence only.
    if "boundary_probability" in df.columns:
        confidence_rows.append({
            "metric": "boundary_probability_mean",
            "value": float(df["boundary_probability"].mean()),
        })
        confidence_rows.append({
            "metric": "boundary_probability_median",
            "value": float(df["boundary_probability"].median()),
        })
        confidence_rows.append({
            "metric": "boundary_probability_ge_0_8_rate",
            "value": float((df["boundary_probability"] >= 0.80).mean()),
        })

    process_df = pd.DataFrame(process_rows)
    patterns_df = pd.DataFrame(pattern_rows)
    sessions_df = pd.DataFrame(session_rows)
    confidence_df = pd.DataFrame(confidence_rows)

    # Frequency/time/consistency screening score. This is deliberately a screening
    # heuristic, not a claim of business ROI.
    if not process_df.empty:
        freq = process_df["execution_count"].to_numpy(float)
        time = process_df["total_duration_sec"].to_numpy(float)
        consistency = 1.0 / (1.0 + process_df["duration_cv"].clip(lower=0).to_numpy(float))

        def minmax(x):
            lo, hi = np.min(x), np.max(x)
            return np.zeros_like(x) + 1.0 if hi <= lo else (x-lo)/(hi-lo)

        process_df["frequency_score"] = minmax(freq)
        process_df["time_score"] = minmax(time)
        process_df["consistency_score"] = consistency
        process_df["priority_score"] = (
            0.40*process_df["frequency_score"]
            + 0.40*process_df["time_score"]
            + 0.20*process_df["consistency_score"]
        )
        process_df = process_df.sort_values(["priority_score", "total_duration_sec"], ascending=False).reset_index(drop=True)
        process_df["priority_rank"] = np.arange(1, len(process_df)+1)

    return process_df, patterns_df, sessions_df, confidence_df, df


def make_variant_analysis(df):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label, g in df.groupby("label"):
        # Behavior variants: coarse app/event composition buckets.
        for _, r in g.iterrows():
            pass
        app_sigs = []
        for sid, a in g.groupby("session_id"):
            app_sigs.extend([])
        rows.append({
            "label": label,
            "duration_min": float(g["duration_sec"].sum()/60.0),
            "duration_cv": float(g["duration_sec"].std(ddof=0)/max(g["duration_sec"].mean(),1e-9)),
            "unique_sessions": int(g["session_id"].nunique()),
            "unique_app_count_mean": float(g["unique_apps"].mean()),
            "event_density_mean": float(g["events_per_sec"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Day 3 Phase 6.11 Dataset B analysis")
    parser.add_argument("--dataset-a", default="Dataset A/dataset_a")
    parser.add_argument("--dataset-b", default="Dataset B/dataset_b")
    parser.add_argument("--output", default="Outputs/Day 3/phase6_11")
    args = parser.parse_args()

    repo = Path.cwd()
    dataset_a = repo / args.dataset_a
    dataset_b = repo / args.dataset_b
    out = repo / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("="*78)
    print("DAY 3 - PHASE 6.11 - DATASET B")
    print("Boundary inference -> process discovery -> automation analysis")
    print("="*78)

    # Validate required Day 2 artifacts before expensive processing.
    required = [
        repo / "Outputs/Day 2/day2_boundary_vs_normal/boundary_vs_normal_features.csv",
        repo / "Outputs/Day 2/day2_boundary_confirmation/candidate_confirmation_features.csv",
        repo / "Outputs/Day 2/day2_gt_boundary_analysis/gt_switch_boundaries.csv",
        repo / "Analysis/Analysis Day 2/day2_boundary_candidate detection.py",
        repo / "Analysis/Analysis Day 2/day2_boudary_confirmation.py",
        repo / "Analysis/Analysis Day 2/day2_learned_boundary_confirmation_v2.py",
        repo / "Analysis/Analysis Day 3/day3_sequence_context.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Required validated project files are missing:\n" + "\n".join(missing))
    if not dataset_a.exists():
        raise FileNotFoundError(f"Dataset A not found: {dataset_a.resolve()}")
    if not dataset_b.exists():
        raise FileNotFoundError(f"Dataset B not found: {dataset_b.resolve()}")

    # Train exact Day 2 stages on A.
    print("\n[1/8] Training Day 2 Phase 5F D_lean model on Dataset A...")
    scaler5f, model5f = train_5f_model(repo)

    print("[2/8] Training Day 2 learned 5G confirmation model on Dataset A...")
    conf_features, scaler5g, model5g, pos_n, neg_n = train_5g_model(repo)
    print(f"  5G training rows: positive={pos_n:,}, negative={neg_n:,}")
    print(f"  5G features: {len(conf_features)}")

    print("\n[3/8] Loading Dataset B raw sessions...")
    sessions_b = load_dataset_sessions(dataset_b)
    print(f"  Dataset B sessions loaded: {len(sessions_b)}")

    print("\n[4/8] Running Phase 5F boundary candidate detection on B...")
    scores_b, candidates_b = build_b_candidates(sessions_b, scaler5f, model5f, out)
    print(f"  B continuous score rows: {len(scores_b):,}")
    print(f"  B candidates before confirmation: {len(candidates_b):,}")

    print("\n[5/8] Building exact Phase 5G confirmation features on B...")
    conf_b = build_confirmation_features_B(candidates_b, scores_b, sessions_b)
    conf_b.to_csv(out / "phase6_11_confirmation_features_B.csv", index=False)
    conf_b = apply_5g_model(conf_b, conf_features, scaler5g, model5g)
    conf_b.to_csv(out / "phase6_11_boundary_scores_B.csv", index=False)
    confirmed_b = conf_b[conf_b["boundary_probability"] >= CONFIRMATION_THRESHOLD].copy()
    confirmed_b.to_csv(out / "phase6_11_confirmed_boundaries_B.csv", index=False)
    print(f"  Confirmed B boundaries (p >= {CONFIRMATION_THRESHOLD:.2f}): {len(confirmed_b):,}")

    print("\n[6/8] Constructing B segments...")
    segments = build_segments(confirmed_b, sessions_b)
    print(f"  B segments: {len(segments):,}")

    # Boundary diagnostics without GT.
    boundary_diag = {
        "dataset": "B",
        "sessions": len(sessions_b),
        "raw_events": int(sum(len(s["timestamps"]) for s in sessions_b.values())),
        "continuous_score_rows": int(len(scores_b)),
        "5f_candidates": int(len(candidates_b)),
        "confirmed_boundaries": int(len(confirmed_b)),
        "confirmation_threshold": CONFIRMATION_THRESHOLD,
        "mean_boundary_probability": float(conf_b["boundary_probability"].mean()) if not conf_b.empty else None,
        "median_boundary_probability": float(conf_b["boundary_probability"].median()) if not conf_b.empty else None,
        "note": "Dataset B has no ground truth; no boundary precision/recall is reported.",
    }
    with (out / "phase6_11_boundary_summary.json").open("w", encoding="utf-8") as f:
        json.dump(boundary_diag, f, ensure_ascii=False, indent=2)

    print("\n[7/8] Discovering process families using Dataset B behavior only...")
    segments, discovered_summary, discovery_meta = discover_processes(segments, sessions_b)
    process_df, patterns_df, session_df, confidence_df, segment_df = build_process_analysis(segments, sessions_b)
    variant_df = make_variant_analysis(segment_df)

    # Final deliverable segments.jsonl — exactly the required four fields.
    with (out / "segments.jsonl").open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps({
                "session_id": s["session_id"],
                "start": s["start"],
                "end": s["end"],
                "label": s["label"],
            }, ensure_ascii=False) + "\n")

    segment_df.to_csv(out / "phase6_11_dataset_b_segments_detailed.csv", index=False)
    process_df.to_csv(out / "phase6_11_process_summary.csv", index=False)
    process_df[[
        c for c in [
            "priority_rank", "label", "execution_count", "session_count",
            "worker_proxy_count", "total_duration_min", "mean_duration_sec",
            "duration_cv", "frequency_score", "time_score", "consistency_score",
            "priority_score",
        ] if c in process_df.columns
    ]].to_csv(out / "phase6_11_automation_priority.csv", index=False)
    patterns_df.to_csv(out / "phase6_11_process_patterns.csv", index=False)
    session_df.to_csv(out / "phase6_11_session_summary.csv", index=False)
    confidence_df.to_csv(out / "phase6_11_confidence_summary.csv", index=False)
    discovered_summary.to_csv(out / "phase6_11_process_counts.csv", index=False)
    variant_df.to_csv(out / "phase6_11_process_variants.csv", index=False)

    # Interleaving summary.
    interleave = []
    if not segment_df.empty:
        for sid, g in segment_df.sort_values(["session_id", "start_ms"]).groupby("session_id"):
            labels = g["label"].tolist()
            aba = sum(1 for i in range(1, len(labels)-1) if labels[i-1] == labels[i+1] and labels[i] != labels[i-1])
            interleave.append({
                "session_id": sid,
                "segments": len(labels),
                "unique_processes": len(set(labels)),
                "aba_patterns": aba,
                "interleaved": bool(aba > 0),
            })
    pd.DataFrame(interleave).to_csv(out / "phase6_11_interleaving.csv", index=False)

    final_summary = {
        "phase": "6.11",
        "dataset": "B",
        "sessions": len(sessions_b),
        "raw_events": int(sum(len(s["timestamps"]) for s in sessions_b.values())),
        "5f_candidates": int(len(candidates_b)),
        "confirmed_boundaries": int(len(confirmed_b)),
        "segments": int(len(segments)),
        "process_families_discovered": int(segment_df["label"].nunique()) if not segment_df.empty else 0,
        "cluster_silhouette_cosine": discovery_meta["silhouette"],
        "confirmation_threshold": CONFIRMATION_THRESHOLD,
        "identity_method": "Dataset-B-only unsupervised behavioral clustering",
        "roi_screen": "40% frequency + 40% total time + 20% duration consistency",
        "ground_truth_available": False,
        "warning": "Process labels and automation priority are discovery outputs and require business validation before production automation.",
    }
    with (out / "phase6_11_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print("\n[8/8] PHASE 6.11 COMPLETE")
    print("="*78)
    print(f"Dataset B sessions          : {len(sessions_b):,}")
    print(f"Raw events                  : {final_summary['raw_events']:,}")
    print(f"5F candidates               : {len(candidates_b):,}")
    print(f"Confirmed boundaries        : {len(confirmed_b):,}")
    print(f"Segments                    : {len(segments):,}")
    print(f"Discovered process families : {final_summary['process_families_discovered']:,}")
    print(f"Cluster silhouette          : {discovery_meta['silhouette']}")
    print(f"Output                      : {out.resolve()}")
    print("="*78)


if __name__ == "__main__":
    main()
