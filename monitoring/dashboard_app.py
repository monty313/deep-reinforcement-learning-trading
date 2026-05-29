"""
monitoring/dashboard_app.py
Streamlit dashboard for the FTMO RL trading system.
Run: streamlit run monitoring/dashboard_app.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import streamlit as st

from monitoring.metrics_calculator import calculate_sharpe, calculate_sortino, rolling_sharpe
from monitoring.regime_detector import annotate_regimes, separate_metrics_by_regime
from monitoring.tail_risk_analyzer import analyze_worst_days

# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = "config/training_config.yaml"
TRADE_LOG_DIR = "trade_logs"
SHAP_LOG_DIR  = "shap_logs"
METRICS_CSV   = "metrics.csv"
SUMMARY_JSON  = "metrics_summary.json"


@st.cache_data
def load_cfg():
    p = Path(CONFIG_PATH)
    if not p.exists():
        p = Path("config/ftmo_config.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_metrics_csv():
    p = Path(METRICS_CSV)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_summary_json():
    p = Path(SUMMARY_JSON)
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data
def load_trade_logs(run_id: str) -> pd.DataFrame:
    pattern = str(Path(TRADE_LOG_DIR) / run_id / "*.pkl")
    files   = glob.glob(pattern)
    if not files:
        # fall back to metrics_per_symbol.csv
        fb = Path("metrics_per_symbol.csv")
        if fb.exists():
            return pd.read_csv(fb)
        return pd.DataFrame()
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)


@st.cache_data
def load_forward_test(run_id: str) -> pd.DataFrame:
    for p in [
        Path(TRADE_LOG_DIR) / run_id / "forward_test.csv",
        Path("forward_test.csv"),
        Path(METRICS_CSV),
    ]:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FTMO RL Dashboard", layout="wide")
st.title("Self-Healing FTMO RL Trading System — Dashboard")

cfg = load_cfg()

with st.sidebar:
    st.header("Settings")
    daily_target_pct = st.number_input(
        "Daily Profit Target (%)",
        value=float(cfg["FTMO"]["daily_profit_target_pct"]) * 100,
        step=0.1
    )
    max_dd_pct = st.number_input(
        "Daily Max Drawdown (%)",
        value=float(cfg["FTMO"]["daily_max_drawdown_pct"]) * 100,
        step=0.1
    )
    run_id = st.text_input("Run ID", value="latest")
    regime_filter = st.selectbox("Regime Filter", ["All", "calm", "normal", "volatile"])
    st.markdown("---")
    st.caption("Reload page to refresh data.")

daily_target = daily_target_pct / 100.0
max_dd       = max_dd_pct / 100.0

# ── load data ─────────────────────────────────────────────────────────────────
daily_df = load_metrics_csv()
summary  = load_summary_json()
fwd_df   = load_forward_test(run_id)

# ── top-line summary cards ─────────────────────────────────────────────────────
st.subheader("Summary Statistics")
if summary:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sharpe",        f"{summary.get('sharpe', 0):.3f}",
              delta="vs 0.9 target" if summary.get("sharpe", 0) >= 0.9 else None)
    c2.metric("Sortino",       f"{summary.get('sortino', 0):.3f}")
    c3.metric("Max DD",        f"{summary.get('max_dd', 0)*100:.2f}%")
    c4.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}",
              delta="Good" if summary.get("profit_factor", 0) >= 1.5 else "Low")
    c5.metric("Hit Rate",      f"{summary.get('hit_rate', 0)*100:.1f}%")
    c6.metric("Total Trades",  summary.get("total_trades", 0))
else:
    st.info("Run test_run.py to generate metrics_summary.json")

# Kill switch status (read from any kill log)
kill_log = Path("kill_switch.log")
if kill_log.exists():
    lines = kill_log.read_text().strip().splitlines()
    if lines:
        st.error(f"KILL SWITCH ACTIVE — {lines[-1]}")

# ── equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")
if not fwd_df.empty and "end_equity" in fwd_df.columns:
    eq_series = fwd_df["end_equity"].reset_index(drop=True)
    eq_pct    = (eq_series / eq_series.iloc[0] - 1) * 100

    # Annotate regimes if we have return data
    if "daily_return_pct" in fwd_df.columns:
        fwd_annotated = annotate_regimes(fwd_df)
        color_map = {"calm": "#2196F3", "normal": "#4CAF50", "volatile": "#F44336"}
        st.line_chart(eq_pct, use_container_width=True)
        st.caption("Equity curve (% from start)")

        # Drawdown regions overlay
        rolling_peak = eq_series.cummax()
        dd_series    = (eq_series - rolling_peak) / rolling_peak * 100
        st.area_chart(dd_series, use_container_width=True)
        st.caption("Drawdown from peak (%)")
    else:
        st.line_chart(eq_pct)
elif not daily_df.empty:
    eq = daily_df["end_equity"].reset_index(drop=True)
    st.line_chart((eq / eq.iloc[0] - 1) * 100)
    st.caption("Equity curve from metrics.csv")
else:
    st.info("No equity data found.")

# ── daily metrics by regime ───────────────────────────────────────────────────
st.subheader("Daily FTMO Performance")
if not daily_df.empty:
    df_view = daily_df.copy()

    if "result" not in df_view.columns:
        df_view["result"] = df_view.apply(
            lambda r: ("pass" if r["daily_return_pct"] / 100 >= daily_target
                       and r["daily_max_dd_pct"] / 100 <= max_dd
                       else ("ok" if r["daily_return_pct"] >= 0
                             else "fail")),
            axis=1
        )

    # Regime annotation
    df_view = annotate_regimes(df_view)

    if regime_filter != "All":
        df_view = df_view[df_view["regime"] == regime_filter]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass Days",  int((df_view["result"] == "pass").sum()))
    c2.metric("OK Days",    int((df_view["result"] == "ok").sum()))
    c3.metric("Fail Days",  int((df_view["result"] == "fail").sum()))
    pass_rate = float((df_view["result"] == "pass").mean())
    c4.metric("Pass Rate",  f"{pass_rate*100:.1f}%",
              delta="Target 60%+" if pass_rate >= 0.6 else "Below target")

    # Rolling pass probability
    pass_flag = (df_view["result"] == "pass").astype(float)
    df_view["pass_30d"] = pass_flag.rolling(30, min_periods=1).mean()
    if "date" in df_view.columns:
        st.line_chart(df_view.set_index("date")["pass_30d"], use_container_width=True)
        st.caption("Rolling 30-day pass probability")

    # Daily returns bar chart
    st.bar_chart(df_view.set_index("date")["daily_return_pct"] if "date" in df_view.columns
                 else df_view["daily_return_pct"])
    st.caption("Daily return (%)")

    # Regime breakdown
    st.subheader("Performance by Market Regime")
    regime_stats = separate_metrics_by_regime(df_view, regime_col="regime")
    regime_rows  = [{"regime": r, **v} for r, v in regime_stats.items()]
    st.dataframe(pd.DataFrame(regime_rows))

    # Rolling Sharpe trend
    returns_frac = df_view["daily_return_pct"] / 100.0
    rs = rolling_sharpe(returns_frac, window=20)
    rs_plot = rs.reset_index(drop=True)
    st.line_chart(rs_plot, use_container_width=True)
    st.caption("Rolling 20-day Sharpe")

    # Worst 5 days
    st.subheader("Tail Risk — Worst 5 Days")
    worst = df_view.nsmallest(5, "daily_return_pct")[["date", "daily_return_pct", "daily_max_dd_pct", "result"]] \
            if "date" in df_view.columns \
            else df_view.nsmallest(5, "daily_return_pct")[["daily_return_pct", "daily_max_dd_pct", "result"]]
    st.dataframe(worst)

    tail_stats = analyze_worst_days(returns_frac)
    if tail_stats.get("flag_review"):
        st.warning(f"Worst single day: {tail_stats['worst_single_day']*100:.2f}% — REVIEW recommended")
    else:
        st.success(f"Worst single day: {tail_stats.get('worst_single_day',0)*100:.2f}% — within tolerance")

# ── current live risk indicators ──────────────────────────────────────────────
st.subheader("Live Risk Indicators")
if not daily_df.empty:
    last = daily_df.iloc[-1]
    curr_dd = last.get("daily_max_dd_pct", 0)
    c1, c2 = st.columns(2)

    dd_color = ("normal" if curr_dd < 2.0
                else ("off" if curr_dd < 4.0 else "inverse"))
    c1.metric("Last Day DD %", f"{curr_dd:.3f}%",
              delta="OK" if curr_dd < 2.0 else ("WARN" if curr_dd < 4.0 else "ALERT"))

    consecutive_loss = 0
    for r in reversed(daily_df.get("result", pd.Series()).tolist()):
        if r == "fail":
            consecutive_loss += 1
        else:
            break
    c2.metric("Consecutive Fail Days", consecutive_loss,
              delta="OK" if consecutive_loss < 3 else "WARN")

# ── trade analytics ───────────────────────────────────────────────────────────
st.subheader("Trade Analytics")
trades_df = load_trade_logs(run_id)
if not trades_df.empty and "pnl_pct" in trades_df.columns:
    wins   = (trades_df["pnl_pct"] > 0).sum()
    losses = (trades_df["pnl_pct"] <= 0).sum()
    wr     = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    avg_rr = float(trades_df["pnl_pct"].mean())
    gross_w = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].sum()
    gross_l = abs(trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].sum())
    pf = gross_w / gross_l if gross_l > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades",  len(trades_df))
    c2.metric("Win Rate",      f"{wr:.1f}%",
              delta="Above break-even" if wr > 50 else "Below break-even")
    c3.metric("Avg PnL",       f"{avg_rr*100:.3f}%")
    c4.metric("Profit Factor", f"{pf:.2f}",
              delta="Good" if pf >= 1.5 else ("OK" if pf >= 1.2 else "Reject"))

    if "symbol" in trades_df.columns:
        sym_summary = (
            trades_df.groupby("symbol")
            .agg(trades=("pnl_pct", "count"),
                 avg_pnl=("pnl_pct", "mean"),
                 total_pnl=("pnl_pct", "sum"))
            .reset_index()
        )
        st.dataframe(sym_summary)
else:
    st.info("No trade log data. Run an episode first.")

# ── walk-forward & seed robustness ────────────────────────────────────────────
st.subheader("Walk-Forward & Multi-Seed Results")
col1, col2 = st.columns(2)

with col1:
    wf_path = Path("walk_forward_report.csv")
    if wf_path.exists():
        wf_df = pd.read_csv(wf_path)
        st.write("Walk-Forward Windows")
        st.dataframe(wf_df)
        min_sharpe = wf_df["sharpe"].min()
        if min_sharpe >= 0.9:
            st.success(f"All windows pass (min Sharpe {min_sharpe:.3f})")
        else:
            st.error(f"Window failure detected (min Sharpe {min_sharpe:.3f} < 0.9)")
    else:
        st.info("Run training/walk_forward_trainer.py to generate this report.")

with col2:
    seed_path = Path("seed_robustness_report.csv")
    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        st.write("Multi-Seed Robustness")
        st.dataframe(seed_df[["seed", "sharpe", "max_dd", "pass_rate"]])
        std = seed_df["sharpe"].std()
        verdict_color = st.success if std < 0.3 else st.warning
        verdict_color(f"Sharpe std = {std:.3f} ({'Low variance — robust' if std < 0.3 else 'High variance — possible overfit'})")
    else:
        st.info("Run training/multi_seed_runner.py to generate this report.")

# ── slippage report ───────────────────────────────────────────────────────────
st.subheader("Slippage & Fee Impact")
slip_path = Path("slippage_impact_report.json")
if slip_path.exists():
    slip = json.loads(slip_path.read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("Backtest Sharpe", slip.get("backtest", {}).get("sharpe", "n/a"))
    c2.metric("Live Sharpe",     slip.get("live", {}).get("sharpe", "n/a"))
    c3.metric("Degradation",     f"{slip.get('pct_degradation', 0):.1f}%")
    if slip.get("alert"):
        st.error(slip.get("alert_message", "Check slippage"))
    else:
        st.success(slip.get("alert_message", "OK"))
else:
    st.info("Run test_run.py to generate slippage_impact_report.json")

# ── SHAP feature importance ───────────────────────────────────────────────────
st.subheader("Feature Importance (SHAP)")
shap_files = sorted(glob.glob(f"{SHAP_LOG_DIR}/shap_*.json"))
if shap_files:
    with open(shap_files[-1]) as f:
        shap_records = json.load(f)
    if shap_records:
        last = shap_records[-1]
        st.write(f"Last explained action: **{last.get('action_name', 'N/A')}**")
        feat_df = pd.DataFrame(last.get("top_features", []))
        if not feat_df.empty:
            feat_df = feat_df.set_index("name").sort_values("shap_value")
            st.bar_chart(feat_df["shap_value"])
else:
    st.info("No SHAP logs found.")

st.caption("FTMO RL Dashboard — config: " + CONFIG_PATH)
