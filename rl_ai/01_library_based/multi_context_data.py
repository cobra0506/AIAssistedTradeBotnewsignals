import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ContextDataset:
    symbol: str
    timeframe: str
    data: pd.DataFrame
    source_path: str


def _normalize_timeframe(value: str) -> str:
    return str(value).strip().rstrip("m")


def _timeframe_label(value: str) -> str:
    return f"{_normalize_timeframe(value)}m"


def discover_context_files(
    data_dir: str,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str, str]]:
    path = Path(data_dir)
    if not path.exists():
        return []

    allowed_symbols = {str(s).upper() for s in symbols} if symbols else None
    allowed_timeframes = {_normalize_timeframe(tf) for tf in timeframes} if timeframes else None

    discovered: List[Tuple[str, str, str]] = []
    for csv_file in sorted(path.glob("*.csv")):
        stem = csv_file.stem
        if "_" not in stem:
            continue
        symbol, tf = stem.rsplit("_", 1)
        symbol = symbol.upper()
        tf_norm = _normalize_timeframe(tf)

        if allowed_symbols and symbol not in allowed_symbols:
            continue
        if allowed_timeframes and tf_norm not in allowed_timeframes:
            continue

        discovered.append((symbol, _timeframe_label(tf_norm), str(csv_file)))
    return discovered


def _extract_period(name: str, default: int = 14) -> int:
    match = re.search(r"(\d+)", str(name))
    if not match:
        return int(default)
    try:
        return max(1, int(match.group(1)))
    except ValueError:
        return int(default)


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(window=period, min_periods=period).mean()
    roll_down = down.rolling(window=period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _ensure_indicator_column(df: pd.DataFrame, column_name: str) -> None:
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        return

    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    period = _extract_period(column_name, default=14)

    if column_name.startswith("ema_"):
        df[column_name] = close.ewm(span=period, adjust=False).mean()
        return

    if column_name.startswith("sma_"):
        df[column_name] = close.rolling(window=period, min_periods=1).mean()
        return

    if column_name.startswith("rsi_"):
        df[column_name] = _compute_rsi(close, period=period)
        return

    if column_name.startswith("volume_sma_"):
        df[column_name] = volume.rolling(window=period, min_periods=1).mean()
        return

    if column_name.startswith("return_"):
        steps = _extract_period(column_name, default=1)
        df[column_name] = close.pct_change(periods=steps)
        return

    df[column_name] = 0.0


def prepare_multi_context_datasets(
    data_dir: str,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    indicator_columns: Optional[Iterable[str]] = None,
    min_rows: int = 200,
    add_context_features: bool = True,
) -> Tuple[List[ContextDataset], Dict[str, object]]:
    discovered = discover_context_files(data_dir, symbols=symbols, timeframes=timeframes)
    indicator_list = [str(col).strip() for col in (indicator_columns or []) if str(col).strip()]

    if not discovered:
        return [], {"contexts_found": 0, "contexts_loaded": 0}

    symbol_order = sorted({symbol for symbol, _, _ in discovered})
    timeframe_order = sorted({_normalize_timeframe(tf) for _, tf, _ in discovered}, key=lambda x: int(x))
    symbol_index = {symbol: idx for idx, symbol in enumerate(symbol_order)}
    timeframe_index = {tf: idx for idx, tf in enumerate(timeframe_order)}

    symbol_den = max(1, len(symbol_order) - 1)
    tf_den = max(1, len(timeframe_order) - 1)

    contexts: List[ContextDataset] = []
    for symbol, timeframe_label, file_path in discovered:
        df = pd.read_csv(file_path)
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            continue

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

        if add_context_features:
            df["ctx_symbol_id"] = float(symbol_index[symbol] / symbol_den)
            tf_norm = _normalize_timeframe(timeframe_label)
            df["ctx_timeframe_id"] = float(timeframe_index[tf_norm] / tf_den)

        for indicator in indicator_list:
            _ensure_indicator_column(df, indicator)

        df = df.ffill().bfill().fillna(0.0)
        if len(df) < int(min_rows):
            continue

        contexts.append(
            ContextDataset(
                symbol=symbol,
                timeframe=timeframe_label,
                data=df.reset_index(drop=True),
                source_path=file_path,
            )
        )

    metadata: Dict[str, object] = {
        "contexts_found": len(discovered),
        "contexts_loaded": len(contexts),
        "symbols": symbol_order,
        "timeframes": [_timeframe_label(tf) for tf in timeframe_order],
    }
    return contexts, metadata
