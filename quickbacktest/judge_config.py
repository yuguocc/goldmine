
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class StrategyJudgeConfig:
    min_relative_improvement: float = 0.05
    plateau_rounds: int = 5
    success_plateau_rounds: int = 3
    eps: float = 1e-8


@dataclass
class StrategyJudgeState:
    best_excess_return: Optional[float] = None
    best_information_ratio: Optional[float] = None
    best_risk_to_reward_ratio: Optional[float] = None

    best_signal_combinations: List[str] = field(default_factory=list)
    best_strategy_name: Optional[str] = None

    plateau_count: int = 0
    success_plateau_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


class StrategyJudge:
    """
    Expected input:
    {
        "original_hypothesis": "...",
        "hypothesis_true": true,
        "recommended_hypothesis": "...",
        "best_signal_combinations": ["signal_1+signal_3"],
        "best_strategy_name": "...",
        "metrics": {
            "excess_return": 0.061,
            "information_ratio": 0.836,
            "risk_to_reward_ratio": 0.62
        },
        "hypothesis_evidence": {
            "summary": "..."
        }
    }
    """

    def __init__(self, config: Optional[StrategyJudgeConfig] = None) -> None:
        self.config = config or StrategyJudgeConfig()
        self.state = StrategyJudgeState()

    def step(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        metrics = evaluation["metrics"]

        curr_excess_return = float(metrics["excess_return"])
        curr_information_ratio = float(metrics["information_ratio"])
        curr_risk_to_reward_ratio = float(metrics["risk_to_reward_ratio"])

        hypothesis_true = bool(evaluation["hypothesis_true"])
        recommended_hypothesis = str(evaluation["recommended_hypothesis"])
        best_signal_combinations = evaluation["best_signal_combinations"]
        best_strategy_name = str(evaluation["best_strategy_name"])
        evidence_summary = str(evaluation["hypothesis_evidence"]["summary"])

        # ===== first round =====
        if self.state.best_excess_return is None:
            self.state.best_excess_return = curr_excess_return
            self.state.best_information_ratio = curr_information_ratio
            self.state.best_risk_to_reward_ratio = curr_risk_to_reward_ratio
            self.state.best_signal_combinations = list(best_signal_combinations)
            self.state.best_strategy_name = best_strategy_name
            self.state.history.append(evaluation)

            return {
                "decision": "continue",
                "reason": "first_round",
                "next_step_path": {
                    "next_hypothesis": recommended_hypothesis,
                    "best_signal_combinations": self.state.best_signal_combinations,
                    "best_strategy_name": self.state.best_strategy_name,
                    "path_reason": evidence_summary,
                },
                # "extra": {
                #     "current_excess_return": curr_excess_return,
                #     "current_information_ratio": curr_information_ratio,
                #     "current_risk_to_reward_ratio": curr_risk_to_reward_ratio,
                #     "best_excess_return": self.state.best_excess_return,
                #     "best_information_ratio": self.state.best_information_ratio,
                #     "best_risk_to_reward_ratio": self.state.best_risk_to_reward_ratio,
                #     "relative_excess_return_improvement": 0.0,
                #     "relative_information_ratio_improvement": 0.0,
                #     "relative_risk_to_reward_improvement": 0.0,
                #     "plateau_count": 0,
                #     "success_plateau_count": 0,
                # },
            }

        # ===== relative improvements =====
        excess_return_improve = self._relative_improvement(
            curr=curr_excess_return,
            best=self.state.best_excess_return,
            direction="higher_better",
        )

        information_ratio_improve = self._relative_improvement(
            curr=curr_information_ratio,
            best=self.state.best_information_ratio,
            direction="higher_better",
        )

        risk_to_reward_improve = self._relative_improvement(
            curr=curr_risk_to_reward_ratio,
            best=self.state.best_risk_to_reward_ratio,
            direction="lower_better",
        )

        excess_return_good = excess_return_improve >= self.config.min_relative_improvement
        information_ratio_good = information_ratio_improve >= self.config.min_relative_improvement
        risk_to_reward_good = risk_to_reward_improve >= self.config.min_relative_improvement

        any_good = excess_return_good or information_ratio_good or risk_to_reward_good

        # ===== update best metrics =====
        if excess_return_good:
            self.state.best_excess_return = curr_excess_return

        if information_ratio_good:
            self.state.best_information_ratio = curr_information_ratio

        if risk_to_reward_good:
            self.state.best_risk_to_reward_ratio = curr_risk_to_reward_ratio

        # ===== update best path if any metric improved =====
        if any_good:
            self.state.best_signal_combinations = list(best_signal_combinations)
            self.state.best_strategy_name = best_strategy_name

        # ===== plateau logic =====
        if any_good:
            self.state.plateau_count = 0
        else:
            self.state.plateau_count += 1

        # ===== success plateau logic =====
        if hypothesis_true and (not excess_return_good) and (not information_ratio_good) and (not risk_to_reward_good):
            self.state.success_plateau_count += 1
        else:
            self.state.success_plateau_count = 0

        self.state.history.append(evaluation)

        next_step_path = {
            "next_hypothesis": recommended_hypothesis,
            "best_signal_combinations": self.state.best_signal_combinations,
            "best_strategy_name": self.state.best_strategy_name,
            "path_reason": evidence_summary,
        }

        extra = {
            "current_excess_return": curr_excess_return,
            "current_information_ratio": curr_information_ratio,
            "current_risk_to_reward_ratio": curr_risk_to_reward_ratio,
            "best_excess_return": self.state.best_excess_return,
            "best_information_ratio": self.state.best_information_ratio,
            "best_risk_to_reward_ratio": self.state.best_risk_to_reward_ratio,
            "relative_excess_return_improvement": excess_return_improve,
            "relative_information_ratio_improvement": information_ratio_improve,
            "relative_risk_to_reward_improvement": risk_to_reward_improve,
            "plateau_count": self.state.plateau_count,
            "success_plateau_count": self.state.success_plateau_count,
        }

        if self.state.plateau_count >= self.config.plateau_rounds:
            return {
                "decision": "end",
                "reason": f"strategy_plateau_{self.config.plateau_rounds}_rounds",
                "next_step_path": next_step_path,
                "extra": extra,
            }

        if self.state.success_plateau_count >= self.config.success_plateau_rounds:
            return {
                "decision": "end",
                "reason": "hypothesis_true_and_strategy_plateau",
                "next_step_path": next_step_path,
                "extra": extra,
            }

        return {
            "decision": "continue",
            "reason": "still_improving_or_not_yet_plateau",
            "next_step_path": next_step_path,
            # "extra": extra,
        }

    def _relative_improvement(self, curr: float, best: float, direction: str) -> float:
        if direction == "higher_better":
            return (curr - best) / max(abs(best), self.config.eps)
        if direction == "lower_better":
            return (best - curr) / max(abs(best), self.config.eps)
        raise ValueError(f"Unknown direction: {direction}")

@dataclass
class JudgeConfig:
    """
    Judge stopping rules.

    Args:
        min_relative_improvement:
            Minimum relative improvement required to count as meaningful progress.
            Example: 0.05 means 5%.
        plateau_rounds:
            End if combo rank_ic fails to improve meaningfully for this many consecutive rounds.
        success_plateau_rounds:
            End earlier if all hypotheses are true and combo rank_ic is already on plateau
            for this many consecutive rounds.
        eps:
            Small constant to avoid division by zero.
    """
    min_relative_improvement: float = 0.05
    plateau_rounds: int = 5
    success_plateau_rounds: int = 3
    min_avg_rank_ic_1h: float = 0.02
    min_avg_rank_ic_5h: float = 0.0
    rank_ic_early_stop_threshold: float = 0.05
    eps: float = 1e-8


@dataclass
class JudgeState:
    """
    Persistent state across iterations.
    """
    best_hypothesis_passed: int = 0
    best_profitable_count: int = 0
    best_avg_rank_ic_1h: Optional[float] = None
    best_avg_rank_ic_5h: Optional[float] = None
    plateau_count: int = 0
    success_plateau_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_signal_combination = None
    best_combo_hypothesis:str = None


class SignalJudge:
    """
    Rule-based judge for signal iteration.

    Input JSON format:
    {
      "signals": [
        {
          "name": "...",
          "original_hypothesis": "...",
          "hypothesis_true": true/false,
          "recommended_hypothesis": "...",
          "rank_ic": 0.024
        },
        ...
      ],
      "combo": {
        "original_hypothesis": "...",
        "hypothesis_true": true/false,
        "recommended_hypothesis": "...",
        "rank_ic": 0.019
      }
    }

    Output:
    {
      "decision": "continue" | "end",
      "reason": "...",
      "combo_rank_ic": float,
      "best_combo_rank_ic": float,
      "relative_improvement_vs_best": float,
      "all_hypothesis_true": bool,
      "plateau_count": int,
      "success_plateau_count": int,
      "suggested_focus": "combo" | "prune_weak_signal" | "refine_combo",
      "weak_signals": [...]
    }
    """

    def __init__(self, config: Optional[JudgeConfig] = None) -> None:
        self.config = config or JudgeConfig()
        self.state = JudgeState()

    def step(self, result_json: Dict[str, Any]) -> Dict[str, Any]:
        signals = result_json["signals"]
        all_signal_true = all(bool(s["hypothesis_true"]) for s in signals)
        all_profitable = all(bool(s["profitable"]) for s in signals)
        hypothesis_passed = sum(bool(s["hypothesis_true"]) for s in signals)
        profitable_count = sum(bool(s["profitable"]) for s in signals)

        rank_ic_1h_values = [float(s.get("rank_ic_1h", 0.0)) for s in signals]
        rank_ic_5h_values = [float(s.get("rank_ic_5h", 0.0)) for s in signals]
        avg_rank_ic_1h = (
            sum(rank_ic_1h_values) / len(rank_ic_1h_values) if rank_ic_1h_values else 0.0
        )
        avg_rank_ic_5h = (
            sum(rank_ic_5h_values) / len(rank_ic_5h_values) if rank_ic_5h_values else 0.0
        )
        ic_threshold_passed = (
            avg_rank_ic_1h >= self.config.min_avg_rank_ic_1h
            and avg_rank_ic_5h >= self.config.min_avg_rank_ic_5h
        )
        current_signals = [s["name"] for s in signals]

        quantile_return_analysis = {
            s["name"]: s["quantile_return_analysis"]
            for s in signals
        }

        all_hypothesis_true = all_signal_true

        quantile_return_analysis_feedback = {
            s["name"]: s["quantile_return_analysis"]
            for s in signals
            if not bool(s["profitable"])
        }


        # weak_signals = {
        #     s["name"]: s["enhance_ic_advise"]
        #     for s in signals
        #     if (not bool(s["hypothesis_true"])) or float(s["rank_ic"]) <= 0.02
        # }

        recommended_hypotheses = {
            s["name"]: s["recommended_hypothesis"]
            for s in signals
            if not bool(s["hypothesis_true"])
        }

        prev_best_hypothesis = self.state.best_hypothesis_passed
        prev_best_profitable = self.state.best_profitable_count
        prev_best_avg_rank_ic_1h = self.state.best_avg_rank_ic_1h
        prev_best_avg_rank_ic_5h = self.state.best_avg_rank_ic_5h

        if self.state.best_avg_rank_ic_1h is None:
            self.state.best_hypothesis_passed = hypothesis_passed
            self.state.best_profitable_count = profitable_count
            self.state.best_avg_rank_ic_1h = avg_rank_ic_1h
            self.state.best_avg_rank_ic_5h = avg_rank_ic_5h
            self.state.best_signal_combination = current_signals
            self.state.history.append(result_json)
            return {
                "decision": "continue",
                "reason": "first_round",
                "all_hypothesis_true": all_hypothesis_true,
                "all_profitable": all_profitable,
                "ic_threshold_passed": ic_threshold_passed,
                "best_signal_combination": self.state.best_signal_combination,
                "quantile_return_analysis": quantile_return_analysis,
                "return_analysis_for_unprofitable_signal": quantile_return_analysis_feedback,
                "recommended_improvements": recommended_hypotheses,
                "extra": {
                    "hypothesis_passed": hypothesis_passed,
                    "profitable_count": profitable_count,
                    "avg_rank_ic_1h": avg_rank_ic_1h,
                    "avg_rank_ic_5h": avg_rank_ic_5h,
                    "best_hypothesis_passed": self.state.best_hypothesis_passed,
                    "best_profitable_count": self.state.best_profitable_count,
                    "best_avg_rank_ic_1h": self.state.best_avg_rank_ic_1h,
                    "best_avg_rank_ic_5h": self.state.best_avg_rank_ic_5h,
                    "is_new_best": True,
                },
            }

        ic_1h_improve = self._relative_improvement(
            curr=avg_rank_ic_1h,
            best=self.state.best_avg_rank_ic_1h,
            direction="higher_better",
        )
        ic_5h_improve = self._relative_improvement(
            curr=avg_rank_ic_5h,
            best=self.state.best_avg_rank_ic_5h,
            direction="higher_better",
        )
        ic_improved = (
            ic_1h_improve >= self.config.min_relative_improvement
            or ic_5h_improve >= self.config.min_relative_improvement
        )
        hypothesis_improved = hypothesis_passed > self.state.best_hypothesis_passed
        profitability_improved = profitable_count > self.state.best_profitable_count
        is_new_best = hypothesis_improved or profitability_improved or ic_improved

        if is_new_best:
            self.state.best_hypothesis_passed = max(self.state.best_hypothesis_passed, hypothesis_passed)
            self.state.best_profitable_count = max(self.state.best_profitable_count, profitable_count)
            if avg_rank_ic_1h >= self.state.best_avg_rank_ic_1h:
                self.state.best_avg_rank_ic_1h = avg_rank_ic_1h
            if avg_rank_ic_5h >= self.state.best_avg_rank_ic_5h:
                self.state.best_avg_rank_ic_5h = avg_rank_ic_5h
            self.state.best_signal_combination = current_signals
            self.state.plateau_count = 0
        else:
            self.state.plateau_count += 1

        if all_hypothesis_true and all_profitable and ic_threshold_passed:
            self.state.success_plateau_count += 1
        else:
            self.state.success_plateau_count = 0

        self.state.history.append(result_json)

        extra = {
            "hypothesis_passed": hypothesis_passed,
            "profitable_count": profitable_count,
            "avg_rank_ic_1h": avg_rank_ic_1h,
            "avg_rank_ic_5h": avg_rank_ic_5h,
            "best_hypothesis_passed": self.state.best_hypothesis_passed,
            "best_profitable_count": self.state.best_profitable_count,
            "best_avg_rank_ic_1h": self.state.best_avg_rank_ic_1h,
            "best_avg_rank_ic_5h": self.state.best_avg_rank_ic_5h,
            "relative_rank_ic_1h_improvement": ic_1h_improve,
            "relative_rank_ic_5h_improvement": ic_5h_improve,
            "plateau_count": self.state.plateau_count,
            "success_plateau_count": self.state.success_plateau_count,
            "is_new_best": is_new_best,
            "previous_best_hypothesis_passed": prev_best_hypothesis,
            "previous_best_profitable_count": prev_best_profitable,
            "previous_best_avg_rank_ic_1h": prev_best_avg_rank_ic_1h,
            "previous_best_avg_rank_ic_5h": prev_best_avg_rank_ic_5h,
        }

        if all_hypothesis_true and all_profitable and ic_threshold_passed:
            return {
                "decision": "end",
                "reason": "all_hypothesis_true_profitable_and_ic_passed",
                "all_hypothesis_true": all_hypothesis_true,
                "all_profitable": all_profitable,
                "ic_threshold_passed": ic_threshold_passed,
                "best_signal_combination": self.state.best_signal_combination,
                "quantile_return_analysis": quantile_return_analysis,
                "return_analysis_for_unprofitable_signal": quantile_return_analysis_feedback,
                "recommended_improvements": recommended_hypotheses,
                "extra": extra,
            }

        if self.state.plateau_count >= self.config.plateau_rounds:
            return {
                "decision": "end",
                "reason": f"hybrid_plateau_{self.config.plateau_rounds}_rounds",
                "all_hypothesis_true": all_hypothesis_true,
                "all_profitable": all_profitable,
                "ic_threshold_passed": ic_threshold_passed,
                "best_signal_combination": self.state.best_signal_combination,
                "quantile_return_analysis": quantile_return_analysis,
                "return_analysis_for_unprofitable_signal": quantile_return_analysis_feedback,
                "recommended_improvements": recommended_hypotheses,
                "extra": extra,
            }

        return {
            "decision": "continue",
            "reason": "refine_hypothesis_profitability_or_ic",
            "all_hypothesis_true": all_hypothesis_true,
            "all_profitable": all_profitable,
            "ic_threshold_passed": ic_threshold_passed,
            "best_signal_combination": self.state.best_signal_combination,
            "quantile_return_analysis": quantile_return_analysis,
            "return_analysis_for_unprofitable_signal": quantile_return_analysis_feedback,
            "recommended_improvements": recommended_hypotheses,
            "extra": extra,
        }

    def _relative_improvement(self, curr: float, best: float, direction: str) -> float:
        if best is None:
            return 0.0
        if direction == "higher_better":
            return (curr - best) / max(abs(best), self.config.eps)
        if direction == "lower_better":
            return (best - curr) / max(abs(best), self.config.eps)
        raise ValueError(f"Unknown direction: {direction}")
