from quickbacktest.base_types import BaseSignal
import pandas as pd
import talib as ta


class AgentSignal(BaseSignal):
    """Template for one cross-sectional factor signal.

    Use the requested class name and keep `name` identical to it.
    For mean reversion, choose the sign so higher scores imply better returns.
    """

    name = "momentum"

    def compute(self, **kwargs) -> pd.DataFrame:
        """Return one wide signal matrix.

        Available fields: `self.open`, `self.high`, `self.low`, `self.close`,
        `self.volume`, `self.amount`; each is indexed by trade_time with codes
        as columns.

        Rules: no lookahead, no reshape/merge, no extra output columns. Return
        only the raw signal matrix; quickbacktest applies factor_shift later.
        """

        window = kwargs.get("window", 20)

        # Example: trailing momentum.
        signal = self.close.pct_change(window)

        signal.index.name = "trade_time"

        return signal
