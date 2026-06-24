from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_HISTORY = Path("runs") / "rlm_reflexion_run_10" / "portfolio_history.json"


def accepted_factor_count(row: pd.Series) -> int:
    admitted = str(row.get("admitted_factors", "") or "").strip()
    if admitted:
        return len([item for item in admitted.split(",") if item.strip()])
    if str(row.get("admission_status", "") or "") == "accepted" and str(
        row.get("admitted_factor", "") or ""
    ).strip():
        return 1
    return 0


def round_best_rank_ic_from_summary(history_dir: Path, round_number: int) -> dict[str, Any]:
    summary_path = history_dir / f"round_{round_number:03d}" / "round_summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    round_ic = payload.get("round_ic") if isinstance(payload, dict) else None
    if not isinstance(round_ic, dict):
        return {}
    return {
        "round_best_rank_ic": round_ic.get("best_ic"),
        "round_best_module": round_ic.get("best_module"),
        "round_best_improved": round_ic.get("improved"),
    }


def add_round_best_rank_ic(df: pd.DataFrame, *, history_dir: Path) -> pd.DataFrame:
    df = df.copy()
    if "round_best_rank_ic" not in df.columns:
        df["round_best_rank_ic"] = pd.NA
    if "round_best_module" not in df.columns:
        df["round_best_module"] = pd.NA
    if "round_best_improved" not in df.columns:
        df["round_best_improved"] = pd.NA

    missing = df["round_best_rank_ic"].isna()
    if missing.any():
        for idx, row in df[missing].iterrows():
            summary = round_best_rank_ic_from_summary(history_dir, int(row["round"]))
            for key, value in summary.items():
                df.at[idx, key] = value
    return df


def load_portfolio_history(path: str | Path) -> pd.DataFrame:
    history_path = Path(path)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds", []) if isinstance(payload, dict) else []
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"no rounds found in {history_path}")

    df = pd.DataFrame([row for row in rounds if isinstance(row, dict)])
    if df.empty:
        raise ValueError(f"no valid round records found in {history_path}")
    if "round" not in df.columns:
        raise ValueError(f"missing round column in {history_path}")

    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"]).sort_values("round")
    df["round"] = df["round"].astype(int)
    df = add_round_best_rank_ic(df, history_dir=history_path.parent)

    numeric_cols = [
        "component_count",
        "score_rows",
        "composite_rank_ic",
        "round_best_rank_ic",
        "cumulative_return",
        "cumulative_return_after_cost",
        "cumulative_benchmark_return",
        "cumulative_excess_return_after_cost",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["accepted_factor_count"] = df.apply(accepted_factor_count, axis=1)
    if "composite_rank_ic" in df.columns:
        df["composite_rank_ic_change"] = df["composite_rank_ic"].diff()
    return df


def resolve_existing_path(raw_path: Any, *, base_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def return_curve_path_from_portfolio_json(portfolio_json: Path) -> Path | None:
    try:
        payload = json.loads(portfolio_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    portfolio = payload.get("portfolio") if isinstance(payload, dict) else None
    artifacts = portfolio.get("artifacts") if isinstance(portfolio, dict) else None
    curve_path = None
    if isinstance(artifacts, dict):
        curve_path = resolve_existing_path(
            artifacts.get("return_curve_csv"),
            base_dir=portfolio_json.parent,
        )
    if curve_path is not None:
        return curve_path
    fallback = portfolio_json.parent / "portfolio" / "qlib_return_curve.csv"
    return fallback if fallback.exists() else None


def load_round_return_curves(history_path: str | Path) -> pd.DataFrame:
    history_path = Path(history_path)
    history_dir = history_path.parent
    history_df = load_portfolio_history(history_path)
    frames: list[pd.DataFrame] = []
    for row in history_df.to_dict("records"):
        portfolio_json = resolve_existing_path(
            row.get("portfolio_json"),
            base_dir=history_dir,
        )
        if portfolio_json is None:
            continue
        curve_path = return_curve_path_from_portfolio_json(portfolio_json)
        if curve_path is None:
            continue
        curve = pd.read_csv(curve_path)
        if "datetime" not in curve.columns:
            curve = curve.reset_index().rename(columns={"index": "datetime"})
        required = {
            "datetime",
            "strategy_return_after_cost",
            "benchmark_return",
        }
        if not required.issubset(curve.columns):
            continue
        curve = curve.copy()
        curve["datetime"] = pd.to_datetime(curve["datetime"], errors="coerce")
        curve = curve.dropna(subset=["datetime"])
        numeric_cols = [
            "strategy_return",
            "strategy_return_after_cost",
            "benchmark_return",
            "excess_return_after_cost",
        ]
        for col in numeric_cols:
            if col in curve.columns:
                curve[col] = pd.to_numeric(curve[col], errors="coerce")
        curve["round"] = int(row["round"])
        curve["return_curve_csv"] = str(curve_path)
        frames.append(curve)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["round", "datetime"])


def plot_rank_ic(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = df["round"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_ic = axes[0]
    if "composite_rank_ic" in df:
        ax_ic.plot(
            rounds,
            df["composite_rank_ic"],
            marker="o",
            linewidth=1.8,
            label="Composite rank IC",
            color="#1f77b4",
        )
    if "round_best_rank_ic" in df:
        ax_ic.plot(
            rounds,
            df["round_best_rank_ic"],
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label="Round best rank IC",
            color="#d62728",
        )
    ax_ic.axhline(0.0, color="#222222", linewidth=0.8)
    ax_ic.set_ylabel("Rank IC")
    ax_ic.set_title("Composite vs Round-Best Rank IC")
    ax_ic.grid(True, alpha=0.25)
    ax_ic.legend(loc="best")

    ax_delta = axes[1]
    if "composite_rank_ic_change" in df:
        ax_delta.bar(
            rounds,
            df["composite_rank_ic_change"].fillna(0.0),
            width=0.5,
            alpha=0.65,
            color="#17becf",
            label="Round-over-round change",
        )
    ax_delta.axhline(0.0, color="#222222", linewidth=0.8)
    ax_delta.set_xlabel("Round")
    ax_delta.set_ylabel("Rank IC change")
    ax_delta.grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_admissions(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = df["round"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        rounds,
        df["accepted_factor_count"],
        width=0.5,
        color="#2ca02c",
        alpha=0.65,
        label="Accepted factors this round",
    )
    ax.set_ylabel("Accepted factors")
    ax.set_xlabel("Round")
    ax.set_title("Accepted Factor Count By Round")
    ax.set_xticks(rounds)
    ax.grid(True, axis="y", alpha=0.25)

    if "component_count" in df:
        ax_total = ax.twinx()
        ax_total.plot(
            rounds,
            df["component_count"],
            marker="o",
            linewidth=1.7,
            color="#555555",
            label="Library component count",
        )
        ax_total.set_ylabel("Library components")
        left_handles, left_labels = ax.get_legend_handles_labels()
        right_handles, right_labels = ax_total.get_legend_handles_labels()
        ax.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    else:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_returns(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = df["round"]
    fig, ax_return = plt.subplots(figsize=(11, 5))
    return_cols = [
        ("cumulative_return_after_cost", "Strategy after cost", "#2ca02c"),
        ("cumulative_benchmark_return", "Benchmark", "#ff7f0e"),
        ("cumulative_excess_return_after_cost", "Excess after cost", "#9467bd"),
    ]
    for col, label, color in return_cols:
        if col in df:
            ax_return.plot(
                rounds,
                df[col],
                marker="o",
                linewidth=1.6,
                label=label,
                color=color,
            )
    ax_return.axhline(0.0, color="#222222", linewidth=0.8)
    ax_return.set_ylabel("Cumulative return")
    ax_return.set_xlabel("Round")
    ax_return.set_title("Portfolio Cumulative Return By Round")
    ax_return.set_xticks(rounds)
    ax_return.grid(True, alpha=0.25)
    ax_return.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_round_return_curves(curves: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    rounds = sorted(curves["round"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    for idx, round_number in enumerate(rounds):
        sub = curves[curves["round"] == round_number].sort_values("datetime")
        ax.plot(
            sub["datetime"],
            sub["strategy_return_after_cost"],
            linewidth=1.1,
            alpha=0.8,
            color=cmap(idx % cmap.N),
            label=f"Round {int(round_number)}",
        )

    benchmark = (
        curves[["datetime", "benchmark_return"]]
        .dropna()
        .groupby("datetime", as_index=False)["benchmark_return"]
        .mean()
        .sort_values("datetime")
    )
    if not benchmark.empty:
        ax.plot(
            benchmark["datetime"],
            benchmark["benchmark_return"],
            color="#111111",
            linestyle="--",
            linewidth=2.0,
            label="Benchmark",
        )

    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Portfolio Return Curves By Round")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_round_excess_curves(curves: pd.DataFrame, output_path: Path) -> None:
    if "excess_return_after_cost" not in curves.columns:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    rounds = sorted(curves["round"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    for idx, round_number in enumerate(rounds):
        sub = curves[curves["round"] == round_number].sort_values("datetime")
        ax.plot(
            sub["datetime"],
            sub["excess_return_after_cost"],
            linewidth=1.1,
            alpha=0.8,
            color=cmap(idx % cmap.N),
            label=f"Round {int(round_number)}",
        )

    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Portfolio Excess Return Curves By Round")
    ax.set_ylabel("Cumulative excess return after cost")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_history(
    history_path: str | Path = DEFAULT_HISTORY,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    history_path = Path(history_path)
    output_dir = Path(output_dir) if output_dir is not None else history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_portfolio_history(history_path)
    rank_ic_path = output_dir / "portfolio_history_rank_ic.png"
    admissions_path = output_dir / "portfolio_history_admissions.png"
    returns_path = output_dir / "portfolio_history_returns.png"
    round_curves_path = output_dir / "portfolio_round_return_curves.png"
    round_excess_path = output_dir / "portfolio_round_excess_curves.png"

    plot_rank_ic(df, rank_ic_path)
    plot_admissions(df, admissions_path)
    plot_returns(df, returns_path)
    curves = load_round_return_curves(history_path)
    curve_plots: dict[str, str] = {}
    if not curves.empty:
        plot_round_return_curves(curves, round_curves_path)
        curve_plots["round_return_curves"] = str(round_curves_path)
        if "excess_return_after_cost" in curves.columns:
            plot_round_excess_curves(curves, round_excess_path)
            curve_plots["round_excess_curves"] = str(round_excess_path)

    return {
        "history_json": str(history_path),
        "plots": {
            "rank_ic": str(rank_ic_path),
            "admissions": str(admissions_path),
            "returns": str(returns_path),
            **curve_plots,
        },
        "round_count": int(len(df)),
        "round_curve_count": (
            0 if curves.empty else int(curves["round"].nunique())
        ),
        "latest_round": int(df["round"].max()),
        "best_composite_rank_ic": (
            None
            if "composite_rank_ic" not in df
            else float(df["composite_rank_ic"].max())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot factor-library portfolio history.")
    parser.add_argument(
        "history",
        nargs="?",
        default=str(DEFAULT_HISTORY),
        help="Path to portfolio_history.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    args = parser.parse_args()
    result = plot_portfolio_history(args.history, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
