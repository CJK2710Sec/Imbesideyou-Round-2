from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal\boundary_vs_normal_features.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_signal_discrimination"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD PHASE 3 DATA
# ============================================================

print("=" * 70)
print("DAY 2 — PHASE 4")
print("SIGNAL DISCRIMINATION")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING PHASE 3 FEATURES")
print("-" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

print(
    f"Sessions: {df['session_id'].nunique():,}"
)

print(
    f"Boundary examples: "
    f"{df[df['label'] == 1]['example_index'].nunique():,}"
)

print(
    f"Normal examples: "
    f"{df[df['label'] == 0]['example_index'].nunique():,}"
)


# ============================================================
# DEFINE FEATURES
# ============================================================

IDENTITY_COLUMNS = {
    "example_index",
    "label",
    "example_type",
    "session_id",
    "timestamp_ms",
    "window_sec",
}

FEATURE_COLUMNS = [
    c for c in df.columns
    if c not in IDENTITY_COLUMNS
]


# ============================================================
# 1. GLOBAL EFFECT SIZE
# ============================================================

print("\n" + "-" * 70)
print("1. GLOBAL FEATURE DISCRIMINATION")
print("-" * 70)


def standardized_mean_difference(boundary, normal):

    boundary = pd.to_numeric(
        boundary,
        errors="coerce"
    ).dropna()

    normal = pd.to_numeric(
        normal,
        errors="coerce"
    ).dropna()

    if len(boundary) < 2 or len(normal) < 2:
        return np.nan

    b_mean = boundary.mean()
    n_mean = normal.mean()

    b_std = boundary.std(ddof=1)
    n_std = normal.std(ddof=1)

    pooled_variance = (
        ((len(boundary) - 1) * b_std ** 2)
        +
        ((len(normal) - 1) * n_std ** 2)
    ) / max(
        len(boundary) + len(normal) - 2,
        1
    )

    pooled_std = math.sqrt(
        max(pooled_variance, 0)
    )

    if pooled_std == 0:
        return 0.0

    return (
        (b_mean - n_mean)
        / pooled_std
    )


global_rows = []

for window_sec in sorted(
    df["window_sec"].unique()
):

    subset = df[
        df["window_sec"] == window_sec
    ]

    boundary = subset[
        subset["label"] == 1
    ]

    normal = subset[
        subset["label"] == 0
    ]

    for feature in FEATURE_COLUMNS:

        b = boundary[feature]
        n = normal[feature]

        smd = standardized_mean_difference(
            b,
            n
        )

        b_mean = pd.to_numeric(
            b,
            errors="coerce"
        ).mean()

        n_mean = pd.to_numeric(
            n,
            errors="coerce"
        ).mean()

        global_rows.append(
            {
                "window_sec": window_sec,
                "feature": feature,
                "boundary_mean": b_mean,
                "normal_mean": n_mean,
                "smd": smd,
                "abs_smd": abs(smd)
                if pd.notna(smd)
                else np.nan,
            }
        )


global_df = pd.DataFrame(
    global_rows
)

global_df = global_df.sort_values(
    [
        "window_sec",
        "abs_smd"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# PRINT TOP FEATURES PER WINDOW
# ============================================================

for window_sec in sorted(
    global_df["window_sec"].unique()
):

    print("\n" + "=" * 70)
    print(f"WINDOW ±{window_sec} SECONDS")
    print("=" * 70)

    subset = global_df[
        global_df["window_sec"] == window_sec
    ].head(15)

    print(
        f"{'Feature':40s} "
        f"{'Boundary':>12s} "
        f"{'Normal':>12s} "
        f"{'SMD':>10s}"
    )

    print("-" * 70)

    for _, row in subset.iterrows():

        print(
            f"{row['feature']:40s} "
            f"{row['boundary_mean']:12.4f} "
            f"{row['normal_mean']:12.4f} "
            f"{row['smd']:10.4f}"
        )


# ============================================================
# 2. SIGNAL CONSISTENCY ACROSS SESSIONS
# ============================================================

print("\n" + "-" * 70)
print("2. CROSS-SESSION CONSISTENCY")
print("-" * 70)


session_rows = []

for window_sec in sorted(
    df["window_sec"].unique()
):

    window_df = df[
        df["window_sec"] == window_sec
    ]

    for feature in FEATURE_COLUMNS:

        session_effects = []

        for session_id, session_df in (
            window_df.groupby("session_id")
        ):

            boundary = session_df[
                session_df["label"] == 1
            ][feature]

            normal = session_df[
                session_df["label"] == 0
            ][feature]

            if (
                len(boundary.dropna()) < 2
                or
                len(normal.dropna()) < 2
            ):
                continue

            smd = standardized_mean_difference(
                boundary,
                normal
            )

            if pd.notna(smd):
                session_effects.append(
                    smd
                )

        if not session_effects:
            continue

        session_effects = np.array(
            session_effects
        )

        positive_fraction = np.mean(
            session_effects > 0
        )

        negative_fraction = np.mean(
            session_effects < 0
        )

        consistency = max(
            positive_fraction,
            negative_fraction
        )

        session_rows.append(
            {
                "window_sec": window_sec,
                "feature": feature,

                "sessions_evaluated":
                    len(session_effects),

                "mean_session_smd":
                    np.mean(session_effects),

                "median_session_smd":
                    np.median(session_effects),

                "std_session_smd":
                    np.std(session_effects),

                "positive_fraction":
                    positive_fraction,

                "negative_fraction":
                    negative_fraction,

                "directional_consistency":
                    consistency,

                "mean_abs_session_smd":
                    np.mean(
                        np.abs(
                            session_effects
                        )
                    ),
            }
        )


session_df = pd.DataFrame(
    session_rows
)


# ============================================================
# PRINT MOST CONSISTENT FEATURES
# ============================================================

for window_sec in sorted(
    session_df["window_sec"].unique()
):

    print("\n" + "=" * 70)
    print(
        f"WINDOW ±{window_sec} SECONDS"
    )
    print("=" * 70)

    subset = session_df[
        session_df["window_sec"] == window_sec
    ].sort_values(
        "mean_abs_session_smd",
        ascending=False
    ).head(15)

    print(
        f"{'Feature':40s} "
        f"{'Mean|SMD|':>12s} "
        f"{'Consistency':>13s} "
        f"{'Sessions':>10s}"
    )

    print("-" * 70)

    for _, row in subset.iterrows():

        print(
            f"{row['feature']:40s} "
            f"{row['mean_abs_session_smd']:12.4f} "
            f"{row['directional_consistency'] * 100:12.1f}% "
            f"{int(row['sessions_evaluated']):10d}"
        )


# ============================================================
# 3. CORRELATION / REDUNDANCY ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("3. FEATURE REDUNDANCY")
print("-" * 70)

correlation_rows = []

for window_sec in sorted(
    df["window_sec"].unique()
):

    subset = df[
        df["window_sec"] == window_sec
    ]

    numeric = subset[
        FEATURE_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    corr = numeric.corr(
        method="spearman"
    )

    features = list(
        corr.columns
    )

    for i in range(len(features)):

        for j in range(i + 1, len(features)):

            f1 = features[i]
            f2 = features[j]

            value = corr.loc[
                f1,
                f2
            ]

            if pd.isna(value):
                continue

            correlation_rows.append(
                {
                    "window_sec": window_sec,
                    "feature_1": f1,
                    "feature_2": f2,
                    "spearman_correlation": value,
                    "abs_correlation": abs(value),
                }
            )


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df = correlation_df.sort_values(
    [
        "window_sec",
        "abs_correlation"
    ],
    ascending=[
        True,
        False
    ]
)


for window_sec in sorted(
    correlation_df["window_sec"].unique()
):

    print("\n" + "=" * 70)
    print(
        f"TOP CORRELATED FEATURES — ±{window_sec}s"
    )
    print("=" * 70)

    subset = correlation_df[
        correlation_df["window_sec"] == window_sec
    ].head(10)

    for _, row in subset.iterrows():

        print(
            f"{row['feature_1']:35s} "
            f"<-> "
            f"{row['feature_2']:35s} "
            f"r={row['spearman_correlation']:.4f}"
        )


# ============================================================
# 4. IDENTIFY CONSTANT / USELESS FEATURES
# ============================================================

print("\n" + "-" * 70)
print("4. CONSTANT / LOW-VARIANCE FEATURES")
print("-" * 70)

variance_rows = []

for window_sec in sorted(
    df["window_sec"].unique()
):

    subset = df[
        df["window_sec"] == window_sec
    ]

    for feature in FEATURE_COLUMNS:

        values = pd.to_numeric(
            subset[feature],
            errors="coerce"
        ).dropna()

        if values.empty:
            variance = np.nan
            unique_count = 0
        else:
            variance = values.var()
            unique_count = values.nunique()

        variance_rows.append(
            {
                "window_sec": window_sec,
                "feature": feature,
                "variance": variance,
                "unique_values":
                    unique_count,
            }
        )


variance_df = pd.DataFrame(
    variance_rows
)

constant_df = variance_df[
    variance_df["unique_values"] <= 1
]

if constant_df.empty:

    print("No constant features found.")

else:

    print(
        constant_df[
            [
                "window_sec",
                "feature",
                "unique_values"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 5. FEATURE RANKING
# ============================================================

print("\n" + "-" * 70)
print("5. FINAL FEATURE RANKING")
print("-" * 70)


ranking_rows = []

for window_sec in sorted(
    global_df["window_sec"].unique()
):

    global_subset = global_df[
        global_df["window_sec"] == window_sec
    ]

    session_subset = session_df[
        session_df["window_sec"] == window_sec
    ]

    for _, row in global_subset.iterrows():

        feature = row["feature"]

        matching = session_subset[
            session_subset["feature"] == feature
        ]

        if matching.empty:
            continue

        consistency_row = matching.iloc[0]

        ranking_rows.append(
            {
                "window_sec": window_sec,
                "feature": feature,

                "global_smd":
                    row["smd"],

                "abs_global_smd":
                    row["abs_smd"],

                "mean_abs_session_smd":
                    consistency_row[
                        "mean_abs_session_smd"
                    ],

                "directional_consistency":
                    consistency_row[
                        "directional_consistency"
                    ],

                "sessions_evaluated":
                    consistency_row[
                        "sessions_evaluated"
                    ],
            }
        )


ranking_df = pd.DataFrame(
    ranking_rows
)


# Combined score:
#
# Strong global effect
# ×
# Strong cross-session consistency
#
ranking_df["combined_score"] = (
    ranking_df["abs_global_smd"]
    *
    ranking_df["directional_consistency"]
)


ranking_df = ranking_df.sort_values(
    [
        "window_sec",
        "combined_score"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# PRINT FINAL TOP SIGNALS
# ============================================================

for window_sec in sorted(
    ranking_df["window_sec"].unique()
):

    print("\n" + "=" * 70)
    print(
        f"TOP SIGNALS — ±{window_sec}s"
    )
    print("=" * 70)

    subset = ranking_df[
        ranking_df["window_sec"] == window_sec
    ].head(15)

    print(
        f"{'Feature':40s} "
        f"{'|SMD|':>10s} "
        f"{'Consistency':>13s} "
        f"{'Score':>10s}"
    )

    print("-" * 70)

    for _, row in subset.iterrows():

        print(
            f"{row['feature']:40s} "
            f"{row['abs_global_smd']:10.4f} "
            f"{row['directional_consistency'] * 100:12.1f}% "
            f"{row['combined_score']:10.4f}"
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

global_path = (
    OUTPUT_DIR /
    "global_feature_discrimination.csv"
)

session_path = (
    OUTPUT_DIR /
    "cross_session_feature_consistency.csv"
)

correlation_path = (
    OUTPUT_DIR /
    "feature_redundancy_correlation.csv"
)

variance_path = (
    OUTPUT_DIR /
    "feature_variance.csv"
)

ranking_path = (
    OUTPUT_DIR /
    "final_signal_ranking.csv"
)

summary_path = (
    OUTPUT_DIR /
    "day2_signal_discrimination_summary.json"
)


global_df.to_csv(
    global_path,
    index=False
)

session_df.to_csv(
    session_path,
    index=False
)

correlation_df.to_csv(
    correlation_path,
    index=False
)

variance_df.to_csv(
    variance_path,
    index=False
)

ranking_df.to_csv(
    ranking_path,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "input_rows": int(len(df)),
    "sessions": int(
        df["session_id"].nunique()
    ),
    "windows_sec": sorted(
        [
            float(x)
            for x in
            df["window_sec"].unique()
        ]
    ),
    "top_signals": {},
    "constant_features": (
        constant_df[
            [
                "window_sec",
                "feature"
            ]
        ].to_dict(
            orient="records"
        )
        if not constant_df.empty
        else []
    ),
}


for window_sec in sorted(
    ranking_df["window_sec"].unique()
):

    subset = ranking_df[
        ranking_df["window_sec"] == window_sec
    ].head(10)

    summary["top_signals"][
        str(window_sec)
    ] = subset[
        [
            "feature",
            "global_smd",
            "abs_global_smd",
            "directional_consistency",
            "combined_score",
        ]
    ].to_dict(
        orient="records"
    )


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 COMPLETE")
print("=" * 70)

print(f"\n{global_path}")
print(f"{session_path}")
print(f"{correlation_path}")
print(f"{variance_path}")
print(f"{ranking_path}")
print(f"{summary_path}")

print("\nNext: inspect TOP SIGNALS and FEATURE REDUNDANCY.")