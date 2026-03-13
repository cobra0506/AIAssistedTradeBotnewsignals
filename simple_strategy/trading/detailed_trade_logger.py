"""
Detailed trade session logger.
Uses buffered pandas writes to keep logging overhead low during symbol scans.
"""

import os
import threading
import time
from datetime import datetime

import pandas as pd


class DetailedTradeLogger:
    COLUMNS = [
        "Timestamp",
        "Event",
        "Symbol",
        "Direction (Long/Short)",
        "Qty",
        "Entry Price",
        "Exit Price",
        "Closed P&L",
        "Entry Reason",
        "Exit Reason",
        "ATR Value at Entry",
        "SL Level Set",
        "TP Level Set",
        "Trend Filter Applied (Yes/No + why)",
        "Volatility Rank (e.g., Top 15%)",
        "Simulated Balance Before/After",
        "API AvailableBalance",
        "API Margin",
        "Any Errors/Notes",
    ]

    def __init__(self, log_dir, filename_prefix="trade_log", flush_interval_sec=10, batch_size=50):
        self.log_dir = str(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"{filename_prefix}_{timestamp}.csv")

        self.flush_interval_sec = max(1, int(flush_interval_sec))
        self.batch_size = max(1, int(batch_size))
        self._buffer = []
        self._last_flush_ts = time.time()
        self._lock = threading.Lock()
        self._initialized = False

        self._ensure_file()

    def _ensure_file(self):
        if self._initialized:
            return
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.csv_path, index=False)
        self._initialized = True

    def _normalize_row(self, row):
        normalized = {col: "" for col in self.COLUMNS}
        if isinstance(row, dict):
            for key, value in row.items():
                if key in normalized:
                    normalized[key] = value
        if not normalized["Timestamp"]:
            normalized["Timestamp"] = datetime.now().isoformat()
        return normalized

    def log_event(self, row):
        now_ts = time.time()
        with self._lock:
            self._buffer.append(self._normalize_row(row))
            should_flush = (
                len(self._buffer) >= self.batch_size
                or (now_ts - self._last_flush_ts) >= self.flush_interval_sec
            )
            if should_flush:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def close(self):
        self.flush()

    def _flush_locked(self):
        if not self._buffer:
            self._last_flush_ts = time.time()
            return
        frame = pd.DataFrame(self._buffer, columns=self.COLUMNS)
        frame.to_csv(self.csv_path, mode="a", index=False, header=False)
        self._buffer = []
        self._last_flush_ts = time.time()
