from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OOS_END,
    DEFAULT_OOS_START,
    DEFAULT_PROVIDER_URI,
    DEFAULT_RLM_MODEL,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
)
from .models import ParallelReflexionConfig
from .runner import run_parallel_reflexion

class ParallelReflexionCLI:
    """CLI adapter for constructing config and invoking the OOP runner."""

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        return self.build_parser().parse_args(argv)

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Run ProcessPool parallel RLM factor generation with reflexion memory."
            ),
        )
        parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
        parser.add_argument("--provider-uri", default=str(DEFAULT_PROVIDER_URI))
        parser.add_argument(
            "--factor-library-path",
            default=None,
            help="Default: <output-dir>/factor_library for an isolated run library.",
        )
        parser.add_argument("--instruments", default="csi500")
        parser.add_argument("--start", default=DEFAULT_TRAIN_START)
        parser.add_argument("--end", default=DEFAULT_TRAIN_END)
        parser.add_argument("--benchmark", default="SH000905")
        parser.add_argument("--topk", type=int, default=50)
        parser.add_argument("--n-drop", type=int, default=5)
        parser.add_argument("--horizon", type=int, default=1)
        parser.add_argument("--factor-shift", type=int, default=1)
        parser.add_argument("--candidates", type=int, default=6)
        parser.add_argument("--rounds", type=int, default=2)
        parser.add_argument("--max-workers", type=int, default=3)
        parser.add_argument(
            "--memory-size",
            type=int,
            default=5,
            help=(
                "Entries to keep per structured memory section in memory.json "
                "(state recency, P_succ, P_fail, and insights)."
            ),
        )
        parser.add_argument(
            "--run-portfolio",
            action="store_true",
            help="Also save per-candidate portfolio artifacts.",
        )
        parser.add_argument(
            "--skip-library-portfolio",
            action="store_true",
            help=(
                "Skip the per-round IC-weighted factor-library composite portfolio "
                "backtest."
            ),
        )
        parser.add_argument(
            "--skip-marginal-contribution-gate",
            action="store_true",
            help=(
                "Do not require each admitted factor to improve the training-window "
                "factor-library composite rank IC."
            ),
        )
        parser.add_argument(
            "--marginal-contribution-min-delta",
            type=float,
            default=0.0,
            help=(
                "Minimum required delta in composite rank IC when adding or replacing "
                "a library factor. Default: 0.0."
            ),
        )
        parser.add_argument(
            "--skip-oos-test",
            action="store_true",
            help="Skip the final out-of-sample composite test.",
        )
        parser.add_argument(
            "--oos-start",
            default=DEFAULT_OOS_START,
            help=f"Final OOS start date. Default: {DEFAULT_OOS_START}.",
        )
        parser.add_argument(
            "--oos-end",
            default=DEFAULT_OOS_END,
            help=f"Final OOS end date. Default: {DEFAULT_OOS_END}.",
        )
        parser.add_argument(
            "--oos-warmup-start",
            default=None,
            help="Start date used to warm up OOS factor computation.",
        )
        parser.add_argument("--model", default=DEFAULT_RLM_MODEL)
        parser.add_argument("--recursive-model", default=DEFAULT_RLM_MODEL)
        parser.add_argument("--max-iterations", type=int, default=5)
        logging_group = parser.add_mutually_exclusive_group()
        logging_group.add_argument(
            "--enable-rlm-logging",
            dest="enable_rlm_logging",
            action="store_true",
            default=True,
            help=(
                "Enable verbose RLM console logging and visualization. "
                "Enabled by default."
            ),
        )
        logging_group.add_argument(
            "--disable-rlm-logging",
            dest="enable_rlm_logging",
            action="store_false",
            help=(
                "Disable verbose RLM console logging and visualization for quiet "
                "ProcessPool runs."
            ),
        )
        return parser

    def config_from_args(self, args: argparse.Namespace) -> ParallelReflexionConfig:
        output_dir = Path(args.output_dir)
        factor_library_path = (
            Path(args.factor_library_path)
            if args.factor_library_path
            else output_dir / "factor_library"
        )
        return ParallelReflexionConfig(
            output_dir=output_dir,
            provider_uri=Path(args.provider_uri),
            factor_library_path=factor_library_path,
            instruments=args.instruments,
            start=args.start,
            end=args.end,
            benchmark=args.benchmark,
            topk=args.topk,
            n_drop=args.n_drop,
            horizon=args.horizon,
            factor_shift=args.factor_shift,
            candidates=args.candidates,
            rounds=args.rounds,
            max_workers=args.max_workers,
            memory_size=args.memory_size,
            run_portfolio=args.run_portfolio,
            run_library_portfolio=not args.skip_library_portfolio,
            marginal_contribution_gate=not args.skip_marginal_contribution_gate,
            marginal_contribution_min_delta=args.marginal_contribution_min_delta,
            run_oos_test=not args.skip_oos_test,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            oos_warmup_start=args.oos_warmup_start,
            model=args.model,
            recursive_model=args.recursive_model,
            max_iterations=args.max_iterations,
            enable_rlm_logging=args.enable_rlm_logging,
        )

    def main(self, argv: list[str] | None = None) -> None:
        config = self.config_from_args(self.parse_args(argv))
        summary = run_parallel_reflexion(config)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return ParallelReflexionCLI().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ParallelReflexionConfig:
    return ParallelReflexionCLI().config_from_args(args)


def main(argv: list[str] | None = None) -> None:
    ParallelReflexionCLI().main(argv)


if __name__ == "__main__":
    main()
