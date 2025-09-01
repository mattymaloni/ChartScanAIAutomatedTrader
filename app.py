# app.py — ChartScanAI (Detect | Backtest | Signals)
from __future__ import annotations

import io
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from PIL import Image

import mplfinance as mpf
from ultralytics import YOLO

# ---- Optional backtest helpers (provided in backtest_yolo_events.py) ----
# Make sure backtest_yolo_events.py is in the same folder as this file.
try:
    from backtest_yolo_events import (
        Config as BTConfig,
        build_universe_candidates,
        screen_by_price_and_liquidity,
        download_prices_batched,
        single_test_multi_horizons,
    )
    HAVE_BACKTEST = True
except Exception:
    HAVE_BACKTEST = False


# ============================== Page setup ==============================
st.set_page_config(
    page_title="ChartScanAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_URL = "images/chartscan.png"
DEFAULT_MODEL_PATH = "weights/custom_yolov8.pt"

# ============================== Caching ==============================
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)

@st.cache_data(ttl=60 * 60, show_spinner=False)  # 1 hour cache
def yf_download(ticker: str, interval: str, period: Optional[str] = None,
                start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    return yf.download(ticker, interval=interval, period=period, start=start, end=end, progress=False, auto_adjust=False)

@st.cache_data(ttl=15 * 60, show_spinner=False)  # 15 min cache
def render_candles_png(df: pd.DataFrame, title: str = "", figsize=(18, 6.5), dpi: int = 100, axisoff: bool = True) -> BytesIO:
    fig, _ = mpf.plot(
        df,
        type="candle",
        style="yahoo",
        title=title,
        axisoff=axisoff,
        ylabel="",
        ylabel_lower="",
        volume=False,
        figsize=figsize,
        returnfig=True,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf

# ============================== Chart helpers ==============================
def generate_chart(ticker: str, interval: str = "1d", chunk_size: int = 180, figsize=(18, 6.5), dpi: int = 100) -> Tuple[Optional[BytesIO], Optional[pd.DatetimeIndex]]:
    if interval == "1h":
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        period = None
    else:
        start_date = None
        end_date = None
        period = "max"

    data = yf_download(ticker, interval=interval, start=start_date, end=end_date, period=period)
    if data is None or data.empty:
        return None, None

    data.index = pd.to_datetime(data.index)
    window = data.iloc[-chunk_size:].copy()
    if window.empty:
        return None, None

    buf = render_candles_png(window, title=f"{ticker} Latest {chunk_size} Candles", figsize=figsize, dpi=dpi)
    return buf, window.index

def yolo_on_png(png_buffer: BytesIO, model: YOLO, conf: float = 0.30, imgsz: int = 768):
    im = Image.open(png_buffer).convert("RGB")
    arr = np.array(im)
    res = model.predict(arr, conf=conf, imgsz=imgsz, verbose=False)
    return res[0]  # ultralytics Result

def _extract_signal_and_conf(box, names: Dict[int, str]) -> Tuple[Optional[str], Optional[float]]:
    cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else None
    label = str(names.get(cls_id, "")).lower().strip() if cls_id is not None else ""
    if "buy" in label:
        sig = "buy"
    elif "sell" in label or "seel" in label:
        sig = "sell"
    else:
        sig = None
    conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else None
    return sig, conf

def map_detections_to_last_bar(window_index: pd.DatetimeIndex, result, names: Dict[int, str]) -> List[dict]:
    """
    Approx map: normalize x centers across detections to 0..1, then map to candle index.
    Keep only detections that land on the **last bar** (right-edge only).
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(window_index) == 0:
        return []

    xs = []
    for b in boxes:
        try:
            xs.append(float(b.xywh[0][0]))
        except Exception:
            pass
    if not xs:
        return []

    xmin, xmax = min(xs), max(xs)
    span = max(xmax - xmin, 1e-6)
    last_bar = len(window_index) - 1

    rows = []
    for b in boxes:
        try:
            x, _, w, _ = b.xywh[0].tolist()
        except Exception:
            continue
        frac = (x - xmin) / span
        idx = int(np.clip(round(frac * (len(window_index) - 1)), 0, len(window_index) - 1))
        sig, conf = _extract_signal_and_conf(b, names)
        if sig and idx == last_bar:
            rows.append({
                "event_time": window_index[idx],
                "signal": sig,
                "confidence": float(conf) if conf is not None else None,
                "x_center_px": float(x),
                "width_px": float(w),
            })
    return rows


# ============================== Sidebar ==============================
with st.sidebar:
    st.image(LOGO_URL, use_column_width=True)
    st.markdown("**ChartScanAI** — Research & demo tool. Not financial advice.")

    model_path = st.text_input("YOLO model path", value=DEFAULT_MODEL_PATH)
    confidence = st.slider("Model confidence", 0.25, 0.99, 0.30, 0.01)

    page = st.radio("Mode", ["Detect", "Backtest", "Signals (today)"], index=0)


# ============================== Detect Page ==============================
if page == "Detect":
    st.title("📷 Detect patterns on a chart")

    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("Option A — Upload chart image")
        source_img = st.file_uploader("Upload chart image (png/jpg/webp)", type=("png", "jpg", "jpeg", "bmp", "webp"))

    with colR:
        st.subheader("Option B — Generate last 180 candles")
        ticker = st.text_input("Ticker (e.g., AAPL)", "")
        interval = st.selectbox("Interval", ["1d", "1h", "1wk"], index=0)
        if st.button("Generate chart"):
            if ticker:
                chart_buf, _ = generate_chart(ticker, interval=interval, chunk_size=180)
                if chart_buf:
                    st.success("Chart generated.")
                    st.image(chart_buf, caption=f"{ticker} Latest 180 Candles", use_column_width=True)
                    st.download_button(f"Download {ticker} chart", data=chart_buf, file_name=f"{ticker}_latest_180.png", mime="image/png")
                    # put buffer into session for quick detection
                    st.session_state["__last_generated_chart__"] = chart_buf.getvalue()
                else:
                    st.error("No data for that ticker/interval.")

    # Load model once
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model from {model_path}")
        st.exception(e)
        st.stop()

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        run_uploaded = st.button("Detect on uploaded image")
    with c2:
        run_generated = st.button("Detect on generated chart")

    if run_uploaded:
        if source_img is None:
            st.error("Please upload an image first.")
        else:
            img = Image.open(source_img).convert("RGB")
            res = model.predict(np.array(img), conf=confidence, verbose=False)
            boxes = res[0].boxes
            plotted = res[0].plot()[:, :, ::-1]  # BGR->RGB
            st.image(plotted, caption="Detections", use_column_width=True)
            with st.expander("Raw boxes"):
                for b in boxes:
                    st.write(b.xywh)

    if run_generated:
        if "__last_generated_chart__" not in st.session_state:
            st.error("Generate a chart first.")
        else:
            buf = BytesIO(st.session_state["__last_generated_chart__"])
            res = yolo_on_png(buf, model, conf=confidence)
            plotted = res.plot()[:, :, ::-1]
            st.image(plotted, caption="Detections on generated chart", use_column_width=True)
            with st.expander("Raw boxes"):
                for b in res.boxes:
                    st.write(b.xywh)


# ============================== Backtest Page ==============================
elif page == "Backtest":
    st.title("🧪 Backtest YOLO signals (no look-ahead)")
    if not HAVE_BACKTEST:
        st.warning("Backtest module not found. Add **backtest_yolo_events.py** next to this file.")
        st.stop()

    # Controls
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox("Interval", ["1d", "1h"], index=0)
    days = c2.slider("Windows (days)", 20, 120, 60, help="Number of sliding windows near the end of history.")
    holding = c3.multiselect("Holding bars", [1, 5, 10, 20], default=[1, 5, 10, 20])

    c4, c5, c6 = st.columns(3)
    min_px, max_px = c4.slider("Price range filter", 1.0, 2000.0, (5.0, 1500.0))
    min_dv = c5.number_input("Min avg $ volume (70d)", value=5_000_000, step=1_000_000)
    universe_source = c6.selectbox("Universe", ["Built-in sample", "CSV upload"], index=0)

    uploaded_csv = None
    if universe_source == "CSV upload":
        uploaded_csv = st.file_uploader("Upload CSV with a 'ticker' column (or first column is tickers)", type="csv")

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model from {model_path}")
        st.exception(e)
        st.stop()

    run_it = st.button("Run backtest")
    if run_it:
        with st.spinner("Building universe…"):
            if universe_source == "CSV upload" and uploaded_csv is not None:
                df = pd.read_csv(uploaded_csv)
                col = "ticker" if "ticker" in df.columns else df.columns[0]
                candidates = df[col].astype(str).str.upper().str.strip().tolist()
            else:
                # built-in helper list from the backtest module
                candidates = build_universe_candidates(BTConfig())

            # Screen
            tickers = screen_by_price_and_liquidity(
                candidates,
                min_price=min_px,
                max_price=max_px,
                min_dollar_vol=min_dv,
                lookback_days=BTConfig().dollar_vol_lookback,
            )

        if not tickers:
            st.error("No tickers passed the screen.")
            st.stop()

        cfg = BTConfig(
            interval=interval,
            days=days,
            holding_bars_list=tuple(holding),
        )

        with st.spinner(f"Downloading data for {len(tickers)} tickers…"):
            px = download_prices_batched(tickers, cfg)

        if not px:
            st.error("No price data downloaded.")
            st.stop()

        with st.spinner("Running simulation…"):
            res = single_test_multi_horizons(px, model, cfg)

        st.subheader("Results & KPIs")
        kpi_cols = st.columns(len(holding))
        for i, H in enumerate(holding):
            s = res["summaries"].get(H, {})
            with kpi_cols[i]:
                st.metric(f"H={H} bars • Trades", s.get("trades", 0))
                st.metric("Win-rate", f"{s['win_rate']*100:.1f}%" if s.get("win_rate") is not None else "NA")
                st.metric("Avg ret / trade", f"{s['avg_return']:.4f}" if s.get("avg_return") is not None else "NA")
                st.metric("Total return", f"{s.get('total_return', 0)*100:.2f}%")

        # Show trades + quick "equity" reconstruction per H
        for H in holding:
            df_tr = res["trades"].get(H, pd.DataFrame())
            st.markdown(f"### Trades (H={H})")
            if df_tr is None or df_tr.empty:
                st.info("No trades executed.")
                continue

            df_tr = df_tr.sort_values("exit_time")
            eq = [cfg.start_cash]
            for _, r in df_tr.iterrows():
                eq.append(eq[-1] + float(r["pnl"]))
            st.line_chart(pd.Series(eq, name=f"Equity H={H}"))

            st.dataframe(df_tr.tail(50), use_container_width=True)
            st.download_button(f"Download trades (H={H})", df_tr.to_csv(index=False), file_name=f"trades_H{H}.csv", mime="text/csv")

        with st.expander("Raw detection table"):
            det_all = res.get("all_detections", pd.DataFrame())
            if isinstance(det_all, pd.DataFrame) and not det_all.empty:
                st.dataframe(det_all.head(200), use_container_width=True)
                st.download_button("Download detections", det_all.to_csv(index=False), file_name="detections.csv", mime="text/csv")

        st.caption("Method: accept detections only on the right edge of the window, enter next bar, exit after H bars. Caps limit allocation realism.")


# ============================== Signals Page ==============================
elif page == "Signals (today)":
    st.title("🚨 Right-edge signals (today)")

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model from {model_path}")
        st.exception(e)
        st.stop()

    st.markdown("Provide a small universe (comma-separated) or upload a CSV. We'll scan the latest window for right-edge Buy/Sell detections and rank by confidence.")

    c1, c2 = st.columns([2, 1])
    raw = c1.text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA, TSLA, AMD, AMZN")
    uploaded_csv = c2.file_uploader("or upload CSV", type="csv")

    chunk = st.slider("Window size (candles)", 120, 240, 180, 10)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    topN = st.slider("Top N signals to show", 5, 50, 10, 1)

    run_scan = st.button("Scan now")
    if run_scan:
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            tickers = (
                df[col].astype(str).str.upper().str.strip().tolist()
            )
        else:
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

        if not tickers:
            st.error("No tickers provided.")
            st.stop()

        rows = []
        names = {int(k): str(v) for k, v in getattr(model, "names", {}).items()}

        prog = st.progress(0.0)
        for i, tkr in enumerate(tickers):
            prog.progress((i + 1) / max(len(tickers), 1))
            buf, idx = generate_chart(tkr, interval=interval, chunk_size=chunk)
            if buf is None or idx is None:
                continue
            res = yolo_on_png(buf, model, conf=confidence)
            dets = map_detections_to_last_bar(idx, res, names)
            for d in dets:
                rows.append({
                    "ticker": tkr,
                    "event_time": d["event_time"],
                    "signal": d["signal"],
                    "confidence": d["confidence"],
                })

        prog.empty()

        if not rows:
            st.info("No right-edge detections found.")
        else:
            sigs = pd.DataFrame(rows).sort_values(["confidence"], ascending=False)
            st.success(f"Found {len(sigs)} right-edge detections.")
            st.dataframe(sigs.head(topN), use_container_width=True)
            st.download_button("Download all signals", sigs.to_csv(index=False), file_name="signals.csv", mime="text/csv")

st.caption("Important: ChartScanAI is for research/education only. Not financial advice.")
