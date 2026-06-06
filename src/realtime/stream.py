"""Transaction stream sources for real-time simulation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Iterator

import pandas as pd


class TransactionStream:
    """Replay CSV or cycle rows as a live transaction stream."""

    def __init__(
        self,
        source_path: str | Path,
        batch_size: int = 10,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.source_path = Path(source_path)
        self.batch_size = batch_size
        self.df = pd.read_csv(self.source_path)
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self._cursor = 0

    def _next_slice(self) -> pd.DataFrame:
        n = len(self.df)
        if n == 0:
            return pd.DataFrame()
        start = self._cursor
        end = min(start + self.batch_size, n)
        batch = self.df.iloc[start:end].copy()
        self._cursor = 0 if end >= n else end
        return batch

    def batches(self) -> Iterator[pd.DataFrame]:
        while True:
            yield self._next_slice()

    async def async_batches(self, tps: float) -> AsyncIterator[pd.DataFrame]:
        delay = self.batch_size / max(tps, 0.1)
        while True:
            yield self._next_slice()
            await asyncio.sleep(delay)
