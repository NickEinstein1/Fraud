"""Sliding window buffer for real-time feature streams."""

from __future__ import annotations

from collections import deque

import pandas as pd


class SlidingWindowBuffer:
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._rows: deque[dict] = deque(maxlen=max_size)

    def append(self, df: pd.DataFrame) -> None:
        for row in df.to_dict(orient="records"):
            self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(list(self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()
