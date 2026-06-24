from __future__ import annotations

from .models import ResearchBranch


PRICE_VOLUME_BRANCHES: tuple[ResearchBranch, ...] = (
    ResearchBranch(
        name="Momentum",
        goal="Capture trend continuation using past returns, breakouts, or slope-like price movement.",
        must_use=("self.close", "pct_change/diff/rolling high-low breakout/ewm slope"),
        must_avoid=("plain one-window return only", "volume-free duplicate of generic momentum"),
        examples=("medium-horizon return gated by liquidity", "breakout strength normalized by recent range"),
    ),
    ResearchBranch(
        name="Reversal",
        goal="Capture mean reversion after short-term overreaction or extreme recent moves.",
        must_use=("self.close", "short-horizon return/deviation/z-score"),
        must_avoid=("same sign as momentum", "long-only trend continuation"),
        examples=("negative short return after extreme move", "distance from rolling mean with reversal sign"),
    ),
    ResearchBranch(
        name="Liquidity",
        goal="Use amount or volume to identify tradability, crowding, or liquidity-adjusted opportunity.",
        must_use=("self.volume or self.amount", "rolling mean/rank/relative liquidity"),
        must_avoid=("pure price return without liquidity term", "raw volume level without normalization"),
        examples=("liquidity-adjusted momentum", "amount stability or liquidity shock rank"),
    ),
    ResearchBranch(
        name="Volatility",
        goal="Use risk state, range, or return volatility to condition expected returns.",
        must_use=("rolling std/abs return/high-low range", "risk normalization or volatility change"),
        must_avoid=("unscaled return only", "constant volatility denominator without finite guards"),
        examples=("return divided by recent volatility", "volatility contraction followed by breakout"),
    ),
    ResearchBranch(
        name="Volume-price",
        goal="Combine price direction with volume or amount confirmation/divergence.",
        must_use=("price movement", "self.volume or self.amount"),
        must_avoid=("simple return * simple volume ratio if memory says it duplicated", "price-only factor"),
        examples=("volume-confirmed breakout", "price-volume divergence or volume shock persistence"),
    ),
    ResearchBranch(
        name="Cross-sectional anomaly",
        goal="Use cross-sectional rank or relative strength to compare stocks on each date.",
        must_use=("rank(axis=1, pct=True)", "relative price/volume/volatility feature"),
        must_avoid=("time-series signal without cross-sectional normalization", "all-constant rank"),
        examples=("ranked relative strength", "cross-sectional rank of liquidity-adjusted return"),
    ),
)


def branch_for_candidate(round_number: int, candidate_index: int) -> ResearchBranch:
    """Deterministically rotate style constraints across candidates and rounds."""
    if not PRICE_VOLUME_BRANCHES:
        raise RuntimeError("PRICE_VOLUME_BRANCHES must not be empty")
    offset = max(0, int(round_number) - 1)
    index = (offset + max(1, int(candidate_index)) - 1) % len(PRICE_VOLUME_BRANCHES)
    return PRICE_VOLUME_BRANCHES[index]
