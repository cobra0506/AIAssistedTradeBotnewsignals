import csv
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _series_from_history(history: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        raw = item.get(key)
        if raw is None:
            continue
        values.append(_safe_float(raw))
    return values


def _svg_line_chart(
    values: List[float],
    title: str,
    stroke_color: str,
    width: int = 720,
    height: int = 220,
) -> str:
    if not values:
        return f"<div><h4>{escape(title)}</h4><p>No data</p></div>"

    min_val = min(values)
    max_val = max(values)
    span = max(max_val - min_val, 1e-9)

    left_pad = 30
    right_pad = 10
    top_pad = 18
    bottom_pad = 20

    plot_width = width - left_pad - right_pad
    plot_height = height - top_pad - bottom_pad

    points = []
    for idx, val in enumerate(values):
        x_ratio = 0.0 if len(values) == 1 else idx / float(len(values) - 1)
        y_ratio = (val - min_val) / span
        x = left_pad + x_ratio * plot_width
        y = top_pad + (1.0 - y_ratio) * plot_height
        points.append(f"{x:.2f},{y:.2f}")

    first_val = values[0]
    last_val = values[-1]
    polyline = " ".join(points)

    return f"""
<div style="margin: 14px 0;">
  <h4 style="margin: 0 0 6px 0;">{escape(title)}</h4>
  <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#d6d9de"/>
    <line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{height - bottom_pad}" stroke="#c8ccd2" stroke-width="1"/>
    <line x1="{left_pad}" y1="{height - bottom_pad}" x2="{width - right_pad}" y2="{height - bottom_pad}" stroke="#c8ccd2" stroke-width="1"/>
    <polyline points="{polyline}" fill="none" stroke="{stroke_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="{left_pad}" y="{top_pad - 4}" font-size="11" fill="#1f2937">{max_val:.6f}</text>
    <text x="{left_pad}" y="{height - 4}" font-size="11" fill="#1f2937">{min_val:.6f}</text>
    <text x="{width - right_pad - 130}" y="{top_pad + 12}" font-size="11" fill="#1f2937">start={first_val:.6f}</text>
    <text x="{width - right_pad - 130}" y="{top_pad + 27}" font-size="11" fill="#1f2937">end={last_val:.6f}</text>
  </svg>
</div>
"""


def _write_history_csv(history: List[Dict[str, Any]], csv_path: Path) -> Optional[Path]:
    if not history:
        return None
    fieldnames = sorted({key for item in history for key in item.keys()})
    if not fieldnames:
        return None

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow(item)
    return csv_path


def save_training_report(summary: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "training_summary.json"
    history_csv_path = out_dir / "training_history.csv"
    report_html_path = out_dir / "training_report.html"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    history = summary.get("train_history") or summary.get("history") or []
    history_csv_written = _write_history_csv(history, history_csv_path)

    train_rewards = _series_from_history(history, "train_total_reward")
    validation_rewards = _series_from_history(history, "validation_total_reward")
    train_equity = _series_from_history(history, "train_final_equity")

    test_metrics = summary.get("test_metrics", {})
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>RL Training Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 18px; color: #111827; background: #f7fafc; }}
    h1, h2, h3 {{ margin: 8px 0; }}
    .card {{ background: #fff; border: 1px solid #dde2e8; border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    .muted {{ color: #475569; font-size: 12px; }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>RL Training Report</h1>
  <div class="muted">Generated: {escape(generated_utc)}</div>

  <div class="card">
    <h3>Run Summary</h3>
    <table>
      <tr><th>Model Path</th><td><code>{escape(str(summary.get("model_path", "")))}</code></td></tr>
      <tr><th>Best Score</th><td>{_safe_float(summary.get("best_score")):.6f}</td></tr>
      <tr><th>Train Rows</th><td>{int(summary.get("train_rows", 0))}</td></tr>
      <tr><th>Validation Rows</th><td>{int(summary.get("validation_rows", 0))}</td></tr>
      <tr><th>Test Rows</th><td>{int(summary.get("test_rows", 0))}</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>Test Metrics</h3>
    <table>
      <tr><th>Total Reward</th><td>{_safe_float(test_metrics.get("total_reward")):.6f}</td></tr>
      <tr><th>Final Equity</th><td>{_safe_float(test_metrics.get("final_equity")):.2f}</td></tr>
      <tr><th>Max Drawdown %</th><td>{_safe_float(test_metrics.get("max_drawdown_pct")) * 100.0:.2f}%</td></tr>
      <tr><th>Trades</th><td>{int(test_metrics.get("trades", 0))}</td></tr>
      <tr><th>Steps</th><td>{int(test_metrics.get("steps", 0))}</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>Visuals</h3>
    {_svg_line_chart(train_rewards, "Train Reward By Episode", "#2563eb")}
    {_svg_line_chart(validation_rewards, "Validation Reward By Episode", "#d97706")}
    {_svg_line_chart(train_equity, "Train Final Equity By Episode", "#059669")}
  </div>
</body>
</html>"""

    with report_html_path.open("w", encoding="utf-8") as handle:
        handle.write(html_body)

    result = {
        "summary_json": str(summary_path),
        "training_report_html": str(report_html_path),
    }
    if history_csv_written is not None:
        result["history_csv"] = str(history_csv_written)
    return result


def save_evaluation_report(metrics: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "evaluation_metrics.json"
    report_html_path = out_dir / "evaluation_report.html"
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    signal_counts = metrics.get("signal_counts", {}) if isinstance(metrics, dict) else {}
    rows = ""
    for key in ("OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT", "HOLD"):
        rows += f"<tr><th>{escape(key)}</th><td>{int(signal_counts.get(key, 0))}</td></tr>"

    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>RL Evaluation Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 18px; color: #111827; background: #f7fafc; }}
    h1, h2, h3 {{ margin: 8px 0; }}
    .card {{ background: #fff; border: 1px solid #dde2e8; border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    .muted {{ color: #475569; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>RL Evaluation Report</h1>
  <div class="muted">Generated: {escape(generated_utc)}</div>

  <div class="card">
    <h3>Metrics</h3>
    <table>
      <tr><th>Total Reward</th><td>{_safe_float(metrics.get("total_reward")):.6f}</td></tr>
      <tr><th>Final Equity</th><td>{_safe_float(metrics.get("final_equity")):.2f}</td></tr>
      <tr><th>Max Drawdown %</th><td>{_safe_float(metrics.get("max_drawdown_pct")) * 100.0:.2f}%</td></tr>
      <tr><th>Trades</th><td>{int(metrics.get("trades", 0))}</td></tr>
      <tr><th>Steps</th><td>{int(metrics.get("steps", 0))}</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>Signal Counts</h3>
    <table>
      {rows}
    </table>
  </div>
</body>
</html>"""

    with report_html_path.open("w", encoding="utf-8") as handle:
        handle.write(html_body)

    return {
        "evaluation_json": str(metrics_path),
        "evaluation_report_html": str(report_html_path),
    }
