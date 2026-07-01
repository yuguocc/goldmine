from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.factor_miner import _import_qlib_adapter, _write_json

from .constants import FAST_SCREEN_RANK_IC_THRESHOLD
from .models import ParallelReflexionConfig
from .utils import (
    _candidate_ic,
    _factor_metric_ic,
    _finite_float,
    _series_from_factor_csv,
)


class FactorLibraryPortfolioService:
    """Build one IC-weighted library factor and run a portfolio backtest."""

    def run(
        self,
        *,
        config: ParallelReflexionConfig,
        round_dir: Path,
    ) -> dict[str, Any]:
        output_dir = round_dir / "factor_library_portfolio"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not config.run_library_portfolio:
            return self._write_result(
                output_dir,
                {"status": "skipped", "reason": "library portfolio disabled"},
            )

        try:
            composite = self.build_composite_factor(
                library_root=config.factor_library_path,
                output_dir=output_dir,
            )
        except Exception as exc:
            return self._write_result(
                output_dir,
                {
                    "status": "composite_build_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        if composite.get("status") != "built":
            return self._write_result(output_dir, composite)

        composite_analysis = self.analyze_composite_factor(
            composite=composite,
            config=config,
            output_dir=output_dir,
        )
        composite = {
            **composite,
            "composite_analysis": composite_analysis,
            "composite_rank_ic": composite_analysis.get("rank_ic"),
            "composite_rank_ic_name": composite_analysis.get("rank_ic_name"),
            "composite_analysis_json": composite_analysis.get("analysis_json"),
        }

        try:
            portfolio = self.run_portfolio(
                composite_factor_csv=Path(str(composite["composite_factor_csv"])),
                config=config,
                output_dir=output_dir / "portfolio",
            )
            result = {
                **composite,
                "status": "portfolio_completed",
                "portfolio": portfolio,
                "portfolio_json": str(output_dir / "library_portfolio.json"),
            }
            return self._write_result(output_dir, result)
        except Exception as exc:
            return self._write_result(
                output_dir,
                {
                    **composite,
                    "status": "portfolio_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def run_final_oos(
        self,
        *,
        config: ParallelReflexionConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir = output_dir / "factor_library_portfolio"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not config.run_oos_test:
            return self._write_result(
                output_dir,
                {"status": "skipped", "reason": "final OOS test disabled"},
            )

        try:
            period = self.oos_period(config)
        except Exception as exc:
            return self._write_result(
                output_dir,
                {
                    "status": "oos_period_invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        oos_config = replace(config, start=period["start"], end=period["end"])
        try:
            composite = self.build_oos_composite_factor(
                config=oos_config,
                library_root=config.factor_library_path,
                output_dir=output_dir,
                warmup_start=period["warmup_start"],
            )
        except Exception as exc:
            return self._write_result(
                output_dir,
                {
                    "status": "oos_composite_build_failed",
                    "oos_period": period,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        composite = {**composite, "oos_period": period}
        if composite.get("status") != "built":
            return self._write_result(output_dir, composite)

        composite_analysis = self.analyze_composite_factor(
            composite=composite,
            config=oos_config,
            output_dir=output_dir,
        )
        composite = {
            **composite,
            "composite_analysis": composite_analysis,
            "composite_rank_ic": composite_analysis.get("rank_ic"),
            "composite_rank_ic_name": composite_analysis.get("rank_ic_name"),
            "composite_analysis_json": composite_analysis.get("analysis_json"),
        }

        try:
            portfolio = self.run_portfolio(
                composite_factor_csv=Path(str(composite["composite_factor_csv"])),
                config=oos_config,
                output_dir=output_dir / "portfolio",
            )
            result = {
                **composite,
                "status": "oos_portfolio_completed",
                "portfolio": portfolio,
                "portfolio_json": str(output_dir / "library_portfolio.json"),
            }
            return self._write_result(output_dir, result)
        except Exception as exc:
            return self._write_result(
                output_dir,
                {
                    **composite,
                    "status": "oos_portfolio_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def build_composite_factor(
        self,
        *,
        library_root: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        from quickbacktest import FactorLibrary

        library = FactorLibrary(library_root)
        components = self.load_components(library_root=library_root, library=library)
        return self.build_composite_from_components(
            components=components,
            output_dir=output_dir,
            empty_reason=(
                "no accepted library factors with rank IC above threshold "
                "and factor_data.csv"
            ),
        )

    def marginal_contribution_check(
        self,
        *,
        config: ParallelReflexionConfig,
        library_root: Path,
        candidate_name: str,
        candidate_metrics: dict[str, Any],
        candidate_factor_data_csv: Path,
        output_dir: Path,
        replacement_factor_names: list[str] | None = None,
    ) -> dict[str, Any]:
        from quickbacktest import FactorLibrary

        output_dir.mkdir(parents=True, exist_ok=True)
        min_delta = _finite_float(config.marginal_contribution_min_delta)
        if min_delta is None:
            min_delta = 0.0
        if not config.marginal_contribution_gate:
            return self._write_gate_result(
                output_dir,
                {
                    "available": False,
                    "passed": True,
                    "verdict": "skipped",
                    "reason": "marginal contribution gate disabled",
                    "min_delta": min_delta,
                },
            )

        library = FactorLibrary(library_root)
        baseline_components = self.load_components(
            library_root=library_root,
            library=library,
        )
        if not baseline_components:
            return self._write_gate_result(
                output_dir,
                {
                    "available": True,
                    "passed": True,
                    "verdict": "skipped",
                    "reason": "no existing accepted components",
                    "baseline_component_count": 0,
                    "min_delta": min_delta,
                },
            )

        candidate_component = self.component_from_factor_data(
            name=candidate_name,
            metrics=candidate_metrics,
            factor_data_csv=candidate_factor_data_csv,
        )
        if candidate_component is None:
            return self._write_gate_result(
                output_dir,
                {
                    "available": False,
                    "passed": False,
                    "verdict": "rejected",
                    "reason": "candidate component unavailable",
                    "baseline_component_count": len(baseline_components),
                    "min_delta": min_delta,
                },
            )

        replacement_names = {
            str(name)
            for name in (replacement_factor_names or [])
            if str(name or "").strip()
        }
        replacement_names.add(candidate_name)
        with_candidate_components = [
            item
            for item in baseline_components
            if str(item.get("name", "")) not in replacement_names
        ]
        with_candidate_components.append(candidate_component)

        baseline = self.composite_rank_ic_for_gate(
            components=baseline_components,
            config=config,
            output_dir=output_dir / "baseline",
        )
        with_candidate = self.composite_rank_ic_for_gate(
            components=with_candidate_components,
            config=config,
            output_dir=output_dir / "with_candidate",
        )
        baseline_ic = _finite_float(baseline.get("rank_ic"))
        with_candidate_ic = _finite_float(with_candidate.get("rank_ic"))
        if baseline_ic is None or with_candidate_ic is None:
            return self._write_gate_result(
                output_dir,
                {
                    "available": False,
                    "passed": False,
                    "verdict": "rejected",
                    "reason": "composite rank IC unavailable",
                    "baseline": baseline,
                    "with_candidate": with_candidate,
                    "baseline_component_count": len(baseline_components),
                    "with_candidate_component_count": len(with_candidate_components),
                    "replacement_factor_names": sorted(
                        replacement_names - {candidate_name}
                    ),
                    "min_delta": min_delta,
                },
            )

        delta = with_candidate_ic - baseline_ic
        passed = delta >= min_delta
        return self._write_gate_result(
            output_dir,
            {
                "available": True,
                "passed": passed,
                "verdict": "accepted" if passed else "rejected",
                "reason": (
                    "candidate improves or preserves composite rank IC"
                    if passed
                    else "candidate reduces composite rank IC"
                ),
                "baseline_rank_ic": baseline_ic,
                "with_candidate_rank_ic": with_candidate_ic,
                "delta_rank_ic": delta,
                "min_delta": min_delta,
                "baseline": baseline,
                "with_candidate": with_candidate,
                "baseline_component_count": len(baseline_components),
                "with_candidate_component_count": len(with_candidate_components),
                "replacement_factor_names": sorted(
                    replacement_names - {candidate_name}
                ),
            },
        )

    def build_oos_composite_factor(
        self,
        *,
        config: ParallelReflexionConfig,
        library_root: Path,
        output_dir: Path,
        warmup_start: str,
    ) -> dict[str, Any]:
        components = self.load_oos_components(
            config=config,
            library_root=library_root,
            output_dir=output_dir,
            warmup_start=warmup_start,
        )
        return self.build_composite_from_components(
            components=components,
            output_dir=output_dir,
            empty_reason=(
                "no accepted library factors could be recomputed on the OOS period"
            ),
        )

    def build_composite_from_components(
        self,
        *,
        components: list[dict[str, Any]],
        output_dir: Path,
        empty_reason: str,
    ) -> dict[str, Any]:
        import pandas as pd

        if not components:
            return {
                "status": "skipped",
                "reason": empty_reason,
                "component_count": 0,
                "rank_ic_threshold": FAST_SCREEN_RANK_IC_THRESHOLD,
            }

        raw = pd.concat(
            [item["series"].rename(item["name"]) for item in components],
            axis=1,
        ).sort_index()
        ranked = raw.groupby(level=0).rank(pct=True) - 0.5
        weights = pd.Series(
            {item["name"]: item["weight"] for item in components},
            dtype=float,
        )
        weighted = ranked.mul(weights, axis=1)
        denominator = ranked.notna().mul(weights.abs(), axis=1).sum(axis=1)
        composite = weighted.sum(axis=1, min_count=1) / denominator.replace(0.0, pd.NA)
        composite = composite.dropna().rename("score")
        if composite.empty:
            return {
                "status": "skipped",
                "reason": "composite factor has no non-null scores",
                "component_count": len(components),
            }

        factor_df = composite.reset_index()
        factor_df.columns = ["trade_time", "code", "score"]
        factor_df["trade_time"] = pd.to_datetime(factor_df["trade_time"]).dt.strftime(
            "%Y-%m-%d"
        )

        composite_path = output_dir / "library_composite_factor.csv"
        analysis_path = output_dir / "library_composite_analysis_factor.csv"
        weights_path = output_dir / "library_composite_weights.json"
        factor_df.to_csv(composite_path, index=False, encoding="utf-8")

        analysis_factor_df = self.with_close_for_analysis(
            factor_df=factor_df,
            components=components,
        )
        analysis_factor_csv = ""
        if analysis_factor_df is not None:
            analysis_factor_df.to_csv(analysis_path, index=False, encoding="utf-8")
            analysis_factor_csv = str(analysis_path)

        weight_payload = {
            "weight_metric": "daily_rank_ic_mean",
            "normalization": "daily cross-sectional rank percentile minus 0.5",
            "components": [
                {
                    "name": item["name"],
                    "ic": item["weight"],
                    "ic_name": item["ic_name"],
                    "factor_data_csv": item["factor_data_csv"],
                }
                for item in components
            ],
            "normalized_weights": {
                str(name): float(value / weights.abs().sum())
                for name, value in weights.items()
            },
        }
        _write_json(weights_path, weight_payload)
        return {
            "status": "built",
            "component_count": len(components),
            "score_rows": int(len(factor_df)),
            "composite_factor_csv": str(composite_path),
            "composite_analysis_factor_csv": analysis_factor_csv,
            "weights_json": str(weights_path),
            "weights": weight_payload,
        }

    def load_oos_components(
        self,
        *,
        config: ParallelReflexionConfig,
        library_root: Path,
        output_dir: Path,
        warmup_start: str,
    ) -> list[dict[str, Any]]:
        import pandas as pd
        from quickbacktest import FactorLibrary

        qlib_adapter = _import_qlib_adapter()
        library = FactorLibrary(library_root)
        signals_dir = output_dir / "signals"
        data_dir = output_dir / "component_factor_data"
        signals_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        components: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for meta in library.list_factors():
            if meta.get("status") != "accepted":
                continue
            name = str(meta.get("name", "") or "").strip()
            signal_class = str(meta.get("signal_class", "") or "").strip()
            if not name or not signal_class:
                skipped.append({"factor": name, "reason": "missing signal_class"})
                continue
            try:
                factor = library.read_factor(name)
            except Exception as exc:
                skipped.append(
                    {
                        "factor": name,
                        "reason": "read_factor_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            weight = _factor_metric_ic(factor.get("metrics"))
            if weight is None or weight <= FAST_SCREEN_RANK_IC_THRESHOLD:
                skipped.append(
                    {
                        "factor": name,
                        "reason": "rank_ic_fast_screen_failed",
                        "ic": weight,
                    }
                )
                continue

            factor_dir = library_root / name
            source_signal = factor_dir / "signal.py"
            target_signal = signals_dir / f"{signal_class}.py"
            if not source_signal.exists():
                skipped.append({"factor": name, "reason": "signal.py missing"})
                continue
            shutil.copyfile(source_signal, target_signal)

            try:
                with qlib_adapter.suppress_qlib_console():
                    factor_df = qlib_adapter.compute_qlib_factor_dataframe(
                        signal_modules=[signal_class],
                        base_dir=output_dir,
                        instruments=config.instruments,
                        start=warmup_start,
                        end=config.end,
                        provider_uri=str(config.provider_uri.resolve()),
                        score_column="score",
                        factor_shift=config.factor_shift,
                    )
            except Exception as exc:
                skipped.append(
                    {
                        "factor": name,
                        "reason": "oos_factor_compute_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue

            factor_df = factor_df.copy()
            factor_df["trade_time"] = pd.to_datetime(factor_df["trade_time"])
            factor_df = factor_df[
                factor_df["trade_time"] >= pd.Timestamp(config.start)
            ].copy()
            if factor_df.empty:
                skipped.append({"factor": name, "reason": "empty_oos_factor_data"})
                continue
            factor_df["trade_time"] = factor_df["trade_time"].dt.strftime("%Y-%m-%d")
            factor_data_csv = data_dir / f"{name}.csv"
            factor_df.to_csv(factor_data_csv, index=False, encoding="utf-8")

            series = _series_from_factor_csv(factor_data_csv)
            if series is None:
                skipped.append({"factor": name, "reason": "oos_series_unavailable"})
                continue
            components.append(
                {
                    "name": name,
                    "weight": weight,
                    "ic_name": self.metric_name(factor.get("metrics")),
                    "factor_data_csv": str(factor_data_csv),
                    "series": series,
                    "close_series": self.close_series_from_factor_csv(factor_data_csv),
                }
            )

        _write_json(output_dir / "oos_component_skips.json", {"skipped": skipped})
        return sorted(
            components,
            key=lambda item: _finite_float(item["weight"]) or float("-inf"),
            reverse=True,
        )

    def load_components(self, *, library_root: Path, library: Any) -> list[dict[str, Any]]:
        components: list[dict[str, Any]] = []
        for meta in library.list_factors():
            if meta.get("status") != "accepted":
                continue
            name = str(meta.get("name", "") or "").strip()
            if not name:
                continue
            factor_data_csv = library_root / name / "factor_data.csv"
            if not factor_data_csv.exists():
                continue
            try:
                factor = library.read_factor(name)
            except Exception:
                continue
            weight = _factor_metric_ic(factor.get("metrics"))
            if weight is None or weight <= FAST_SCREEN_RANK_IC_THRESHOLD:
                continue
            series = _series_from_factor_csv(factor_data_csv)
            if series is None:
                continue
            components.append(
                {
                    "name": name,
                    "weight": weight,
                    "ic_name": self.metric_name(factor.get("metrics")),
                    "factor_data_csv": str(factor_data_csv),
                    "series": series,
                    "close_series": self.close_series_from_factor_csv(factor_data_csv),
                }
            )
        return sorted(
            components,
            key=lambda item: _finite_float(item["weight"]) or float("-inf"),
            reverse=True,
        )

    def component_from_factor_data(
        self,
        *,
        name: str,
        metrics: dict[str, Any],
        factor_data_csv: Path,
    ) -> dict[str, Any] | None:
        weight = _factor_metric_ic(metrics)
        if weight is None or weight <= FAST_SCREEN_RANK_IC_THRESHOLD:
            return None
        series = _series_from_factor_csv(factor_data_csv)
        if series is None:
            return None
        return {
            "name": name,
            "weight": weight,
            "ic_name": self.metric_name(metrics),
            "factor_data_csv": str(factor_data_csv),
            "series": series,
            "close_series": self.close_series_from_factor_csv(factor_data_csv),
        }

    def composite_rank_ic_for_gate(
        self,
        *,
        components: list[dict[str, Any]],
        config: ParallelReflexionConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        composite = self.build_composite_from_components(
            components=components,
            output_dir=output_dir,
            empty_reason="no components available for marginal contribution gate",
        )
        if composite.get("status") != "built":
            return self.compact_composite_gate_result(composite=composite)
        analysis = self.analyze_composite_factor(
            composite=composite,
            config=config,
            output_dir=output_dir,
        )
        return self.compact_composite_gate_result(
            composite=composite,
            analysis=analysis,
        )

    @staticmethod
    def oos_period(config: ParallelReflexionConfig) -> dict[str, Any]:
        import pandas as pd

        in_sample_end = pd.Timestamp(config.end).normalize()
        start = (
            pd.Timestamp(config.oos_start).normalize()
            if config.oos_start
            else in_sample_end + pd.Timedelta(days=1)
        )
        end = (
            pd.Timestamp(config.oos_end).normalize()
            if config.oos_end
            else start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        )
        warmup_start = (
            pd.Timestamp(config.oos_warmup_start).normalize()
            if config.oos_warmup_start
            else start - pd.DateOffset(years=1)
        )
        if start > end:
            raise ValueError("oos_start must be <= oos_end")
        if warmup_start > start:
            raise ValueError("oos_warmup_start must be <= oos_start")
        return {
            "in_sample_end": in_sample_end.strftime("%Y-%m-%d"),
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "warmup_start": warmup_start.strftime("%Y-%m-%d"),
            "warmup_days": int((start - warmup_start).days),
            "source": {
                "start": "configured" if config.oos_start else "default",
                "end": "configured" if config.oos_end else "default",
                "warmup_start": (
                    "configured" if config.oos_warmup_start else "default"
                ),
            },
        }

    @staticmethod
    def close_series_from_factor_csv(path: Path) -> Any:
        import pandas as pd

        df = pd.read_csv(path)
        required = {"trade_time", "code", "close"}
        if not required.issubset(df.columns):
            return None
        frame = df[["trade_time", "code", "close"]].copy()
        frame["trade_time"] = pd.to_datetime(frame["trade_time"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        series = frame.set_index(["trade_time", "code"])["close"].dropna().sort_index()
        return series if not series.empty else None

    @staticmethod
    def with_close_for_analysis(
        *,
        factor_df: Any,
        components: list[dict[str, Any]],
    ) -> Any:
        import pandas as pd

        close_series = None
        for item in components:
            series = item.get("close_series")
            if series is None:
                continue
            close_series = (
                series if close_series is None else close_series.combine_first(series)
            )
        if close_series is None or close_series.empty:
            return None

        close_frame = close_series.rename("close").reset_index()
        close_frame["trade_time"] = pd.to_datetime(close_frame["trade_time"]).dt.strftime(
            "%Y-%m-%d"
        )
        analysis_df = factor_df.merge(
            close_frame,
            on=["trade_time", "code"],
            how="left",
        )
        if not analysis_df["close"].notna().any():
            return None
        return analysis_df

    @staticmethod
    def metric_name(metrics: dict[str, Any] | None) -> str:
        if not isinstance(metrics, dict):
            return ""
        if _finite_float(metrics.get("daily_rank_ic_mean")) is not None:
            return "daily_rank_ic_mean"
        if isinstance(metrics.get("rank_ic_distribution"), dict) and _finite_float(
            metrics["rank_ic_distribution"].get("mean")
        ) is not None:
            return "rank_ic_distribution.mean"
        if _finite_float(metrics.get("rank_ic")) is not None:
            return "rank_ic"
        if _finite_float(metrics.get("daily_ic_mean")) is not None:
            return "daily_ic_mean"
        return ""

    @staticmethod
    def compact_composite_gate_result(
        *,
        composite: dict[str, Any],
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        analysis_payload = analysis if isinstance(analysis, dict) else {}
        return {
            "status": composite.get("status"),
            "reason": composite.get("reason"),
            "component_count": composite.get("component_count"),
            "score_rows": composite.get("score_rows"),
            "composite_factor_csv": composite.get("composite_factor_csv"),
            "composite_analysis_factor_csv": composite.get(
                "composite_analysis_factor_csv"
            ),
            "weights_json": composite.get("weights_json"),
            "analysis_status": analysis_payload.get("status"),
            "analysis_json": analysis_payload.get("analysis_json"),
            "rank_ic": analysis_payload.get("rank_ic"),
            "rank_ic_name": analysis_payload.get("rank_ic_name"),
            "error_type": composite.get("error_type")
            or analysis_payload.get("error_type"),
            "error": composite.get("error") or analysis_payload.get("error"),
        }

    @staticmethod
    def analyze_composite_factor(
        *,
        composite: dict[str, Any],
        config: ParallelReflexionConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        import pandas as pd

        analysis_factor_csv = str(composite.get("composite_analysis_factor_csv") or "")
        if not analysis_factor_csv:
            return {
                "status": "skipped",
                "reason": (
                    "composite analysis requires close prices from component "
                    "factor_data.csv"
                ),
            }

        analysis_path = output_dir / "library_composite_analysis.json"
        try:
            qlib_adapter = _import_qlib_adapter()
            factor_df = pd.read_csv(analysis_factor_csv)
            with qlib_adapter.suppress_qlib_console():
                analysis = qlib_adapter.analyze_qlib_factors(
                    factor_df=factor_df,
                    instruments=config.instruments,
                    start=config.start,
                    end=config.end,
                    factor_columns=["score"],
                    score_column="score",
                    horizon=config.horizon,
                    provider_uri=str(config.provider_uri.resolve()),
                )
            metrics = analysis.get("metrics", {}).get("score", {})
            rank_ic, rank_ic_name = _candidate_ic(metrics=metrics)
            payload = {
                "status": "completed",
                "analysis_json": str(analysis_path),
                "rank_ic": rank_ic,
                "rank_ic_name": rank_ic_name,
                "metrics": metrics,
                "analysis": analysis,
            }
            _write_json(analysis_path, payload)
            return payload
        except Exception as exc:
            payload = {
                "status": "failed",
                "analysis_json": str(analysis_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(analysis_path, payload)
            return payload

    @staticmethod
    def run_portfolio(
        *,
        composite_factor_csv: Path,
        config: ParallelReflexionConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        import pandas as pd

        qlib_adapter = _import_qlib_adapter()
        factor_df = pd.read_csv(composite_factor_csv)
        pred = qlib_adapter.factor_df_to_qlib_signal(factor_df, score_column="score")
        return qlib_adapter.simulate_qlib_portfolio(
            pred=pred,
            benchmark=config.benchmark,
            topk=config.topk,
            n_drop=config.n_drop,
            provider_uri=str(config.provider_uri.resolve()),
            output_dir=output_dir,
        )

    @staticmethod
    def _write_result(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
        _write_json(output_dir / "library_portfolio.json", result)
        return result

    @staticmethod
    def _write_gate_result(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
        _write_json(output_dir / "marginal_contribution_gate.json", result)
        return result


def run_factor_library_portfolio(
    *,
    config: ParallelReflexionConfig,
    round_dir: Path,
) -> dict[str, Any]:
    return FactorLibraryPortfolioService().run(config=config, round_dir=round_dir)


class PortfolioHistoryRecorder:
    """Persist compact per-round portfolio results for later evaluation."""

    json_name = "portfolio_history.json"
    csv_name = "portfolio_history.csv"

    def record_round(
        self,
        *,
        output_dir: Path,
        round_summary: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / self.json_name
        history = self.read_history(history_path)
        round_record = self.round_record(round_summary)
        rounds = [
            item
            for item in history.get("rounds", [])
            if item.get("round") != round_record.get("round")
        ]
        rounds.append(round_record)
        rounds.sort(key=lambda item: int(item.get("round") or 0))
        history = {
            "schema_version": 1,
            "portfolio_source": "factor_library_ic_weighted_composite",
            "rounds": rounds,
        }
        _write_json(history_path, history)
        self.write_csv(output_dir / self.csv_name, rounds)
        return {
            "portfolio_history_json": str(history_path),
            "portfolio_history_csv": str(output_dir / self.csv_name),
            "round_count": len(rounds),
        }

    @staticmethod
    def read_history(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"rounds": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"rounds": []}
        return payload if isinstance(payload, dict) else {"rounds": []}

    @staticmethod
    def round_record(round_summary: dict[str, Any]) -> dict[str, Any]:
        portfolio_result = round_summary.get("factor_library_portfolio")
        if not isinstance(portfolio_result, dict):
            portfolio_result = {}
        portfolio = portfolio_result.get("portfolio")
        if not isinstance(portfolio, dict):
            portfolio = {}
        admission = round_summary.get("factor_library_admission")
        if not isinstance(admission, dict):
            admission = {}
        admitted_factors = [
            str(item.get("factor_name", "") or "")
            for item in admission.get("accepted_candidates", [])
            if isinstance(item, dict) and str(item.get("factor_name", "") or "")
        ]
        if not admitted_factors and admission.get("factor_name"):
            admitted_factors = [str(admission.get("factor_name"))]
        round_ic = round_summary.get("round_ic")
        if not isinstance(round_ic, dict):
            round_ic = {}
        return {
            "round": round_summary.get("round"),
            "status": portfolio_result.get("status"),
            "component_count": portfolio_result.get("component_count"),
            "score_rows": portfolio_result.get("score_rows"),
            "composite_rank_ic": portfolio_result.get("composite_rank_ic"),
            "composite_rank_ic_name": portfolio_result.get("composite_rank_ic_name"),
            "round_best_rank_ic": round_ic.get("best_ic"),
            "round_best_module": round_ic.get("best_module"),
            "round_best_improved": round_ic.get("improved"),
            "composite_analysis_status": (
                portfolio_result.get("composite_analysis", {}).get("status")
                if isinstance(portfolio_result.get("composite_analysis"), dict)
                else None
            ),
            "composite_analysis_json": portfolio_result.get("composite_analysis_json"),
            "admission_status": admission.get("status"),
            "admitted_factor": admission.get("factor_name"),
            "admitted_factors": ",".join(admitted_factors),
            "composite_factor_csv": portfolio_result.get("composite_factor_csv"),
            "weights_json": portfolio_result.get("weights_json"),
            "portfolio_json": portfolio_result.get("portfolio_json"),
            "cumulative_return": portfolio.get("cumulative_return"),
            "cumulative_return_after_cost": portfolio.get(
                "cumulative_return_after_cost"
            ),
            "cumulative_benchmark_return": portfolio.get(
                "cumulative_benchmark_return"
            ),
            "cumulative_excess_return_after_cost": portfolio.get(
                "cumulative_excess_return_after_cost"
            ),
            "latest_return_after_cost": portfolio.get("latest_return_after_cost"),
            "topk": portfolio.get("topk"),
            "n_drop": portfolio.get("n_drop"),
            "prediction_rows": portfolio.get("prediction_rows"),
            "prediction_start": portfolio.get("prediction_start"),
            "prediction_end": portfolio.get("prediction_end"),
            "error_type": portfolio_result.get("error_type"),
            "error": portfolio_result.get("error"),
        }

    @staticmethod
    def write_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
        fieldnames = [
            "round",
            "status",
            "component_count",
            "score_rows",
            "composite_rank_ic",
            "composite_rank_ic_name",
            "round_best_rank_ic",
            "round_best_module",
            "round_best_improved",
            "composite_analysis_status",
            "composite_analysis_json",
            "admission_status",
            "admitted_factor",
            "admitted_factors",
            "cumulative_return",
            "cumulative_return_after_cost",
            "cumulative_benchmark_return",
            "cumulative_excess_return_after_cost",
            "latest_return_after_cost",
            "topk",
            "n_drop",
            "prediction_rows",
            "prediction_start",
            "prediction_end",
            "composite_factor_csv",
            "weights_json",
            "portfolio_json",
            "error_type",
            "error",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in rounds:
                writer.writerow({key: item.get(key) for key in fieldnames})
