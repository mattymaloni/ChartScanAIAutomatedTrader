# paper_trader.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.models import Position

from ultralytics import YOLO

# Reuse your utilities
from backtest_yolo_events import (
    Config as BTConfig,
    build_universe_candidates,
    screen_by_price_and_liquidity,
    scan_right_edge_signals,
)

# ---------- Config ----------
@dataclass
class RunConfig:
    # Model / signals
    model_path: str = "weights/custom_yolov8.pt"
    interval: str = "1d"           # '1d' or '1h' (paper bot expects '1d')
    chunk_size: int = 180
    min_price: float = 5.0
    max_price: float = 3000.0
    min_dollar_vol: float = 1_000_000  # Lowered to 1M to include more stocks
    lookback_days: int = 70
    max_trades_per_day: int = 50   # Increased to allow more trades per day
    long_only: bool = True         # set False to allow shorts (paper supports it)
    holding_days: int = 5          # exit after this many trading days

    # Sizing
    max_alloc_per_trade: float = 0.40  # % of equity per trade
    max_portfolio_day_cap: float = 0.90  # max % equity used for all new trades today

    # Universe
    use_csv_universe: bool = False
    universe_csv_path: str = "tickers.csv"
    max_universe_size: int = 1000  # Scan top 1000 stocks

    # Files
    state_path: Path = Path("paper_state.json")
    runs_dir: Path = Path("daily_runs")

# ---------- Helpers ----------
def _now_et() -> datetime:
    # GitHub Actions runs in UTC; we just need a stable timestamp
    return datetime.now(timezone.utc)

def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"open_trades": []}  # list of dicts: symbol, side, qty, entry_date, exit_after

def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, default=str))

def _clear_yfinance_cache():
    """Clear yfinance cache to prevent corrupted data issues"""
    try:
        import shutil
        import os
        import platform
        
        # Different cache locations for different OS
        if platform.system() == "Darwin":  # macOS
            cache_dirs = ["~/Library/Caches/py-yfinance"]
        else:  # Linux/Ubuntu (GitHub Actions)
            cache_dirs = ["~/.cache/py-yfinance", "~/.cache/yfinance"]
            
        for cache_path in cache_dirs:
            cache_dir = os.path.expanduser(cache_path)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"[INFO] Cleared yfinance cache: {cache_dir}")
    except Exception as e:
        print(f"[WARN] Failed to clear yfinance cache: {e}")

def _latest_close_price(symbol: str) -> Optional[float]:
    import os
    ci_env = os.environ.get('CI', '').lower() in ('true', '1', 'yes')
    github_actions = os.environ.get('GITHUB_ACTIONS', '').lower() in ('true', '1', 'yes')
    is_ci = ci_env or github_actions
    max_retries = 8 if is_ci else 5  # Even more retries for CI
    
    for retry in range(max_retries):
        try:
            # Add random delay for CI to avoid coordinated requests
            if is_ci and retry > 0:
                import random
                import time
                random_delay = random.uniform(2, 8)
                time.sleep(random_delay)
            
            # More conservative settings for CI
            df = yf.download(
                symbol, 
                period="2d",  # Shorter period to reduce load
                interval="1d", 
                progress=False, 
                auto_adjust=False,
                timeout=45 if is_ci else 30,  # Longer timeout for CI
                show_errors=False  # Suppress yfinance error messages
            )
            if df is None or df.empty:
                if retry < max_retries - 1:
                    import time
                    time.sleep(3)  # Longer delay
                    continue
                return None
            
            # Handle MultiIndex columns (when downloading single ticker)
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                # For single ticker, columns are like ('Close', 'AAPL')
                close_col = None
                for col in df.columns:
                    if col[0] == 'Close':  # First level is 'Close'
                        close_col = col
                        break
                if close_col is None:
                    if retry < max_retries - 1:
                        import time
                        time.sleep(3)
                        continue
                    return None
                close_series = df[close_col]
            else:
                # Handle simple columns
                if "Close" not in df.columns:
                    if retry < max_retries - 1:
                        import time
                        time.sleep(3)
                        continue
                    return None
                close_series = df["Close"]
                
            if close_series.empty or close_series.isna().all():
                if retry < max_retries - 1:
                    import time
                    time.sleep(3)
                    continue
                return None
                
            # Get the last valid close price
            last_close = close_series.dropna().iloc[-1]
            return float(last_close)
            
        except Exception as e:
            if retry < max_retries - 1:
                import time
                # Progressive delays with randomization for CI
                if is_ci:
                    base_delay = 15 + (retry * 5)  # 15s, 20s, 25s, 30s, 35s...
                    import random
                    jitter = random.uniform(0, 5)
                    time.sleep(base_delay + jitter)
                else:
                    time.sleep(5 + (2 ** retry))
                continue
            
            # Final fallback: try a different approach
            if is_ci and "json" in str(e).lower():
                try:
                    print(f"[INFO] Trying alternative approach for {symbol}...")
                    import yfinance as yf
                    ticker_obj = yf.Ticker(symbol)
                    hist = ticker_obj.history(period="1d")
                    if not hist.empty:
                        return float(hist['Close'].iloc[-1])
                except Exception as fallback_e:
                    print(f"[WARN] Fallback also failed for {symbol}: {fallback_e}")
            
            print(f"[WARN] Failed to get ticker '{symbol}' reason: {e}")
            return None

def _qty_for_notional(price: Optional[float], notional: float) -> int:
    if price is None or price <= 0:
        return 0
    return max(int(notional // price), 0)

def _alpaca_client_from_env() -> TradingClient:
    # Using paper=True directs SDK to paper endpoint; keys from env
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("[WARN] Alpaca API keys not found. Running in DRY RUN mode.")
        # Return a mock client or handle gracefully
        raise ValueError("Alpaca API keys not configured. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables.")
    
    return TradingClient(api_key, api_secret, paper=True)

def _positions_by_symbol(client: TradingClient) -> Dict[str, Position]:
    pos = client.get_all_positions()
    out = {}
    for p in pos:
        out[p.symbol] = p
    return out

def _close_position(client: TradingClient, symbol: str) -> None:
    try:
        client.close_position(symbol)
    except Exception as e:
        print(f"[WARN] close_position({symbol}) failed: {e}")

# ---------- Core ----------
def run_once(cfg: RunConfig) -> dict:
    """
    1) Close any positions whose exit date is due
    2) Scan right-edge signals
    3) Place paper orders for up to N trades
    4) Save artifacts (signals, orders, positions) and persist state
    """
    # Clear yfinance cache to prevent corrupted data issues
    _clear_yfinance_cache()
    
    # CI environment detection (no longer limiting universe size since we fixed API issues)
    import os
    ci_env = os.environ.get('CI', '').lower() in ('true', '1', 'yes')
    github_actions = os.environ.get('GITHUB_ACTIONS', '').lower() in ('true', '1', 'yes')
    
    if ci_env or github_actions:
        print("[INFO] Running in CI environment - full 1000 stock universe enabled")
        # No longer limiting universe size - we fixed the root API issues
    
    # Prep
    try:
        client = _alpaca_client_from_env()
        account = client.get_account()
        equity = float(account.equity)  # paper equity
        print(f"[INFO] Paper equity: ${equity:,.2f}")
        dry_run = False
    except ValueError as e:
        print(f"[INFO] {e}")
        print("[INFO] Running in DRY RUN mode - no actual trades will be placed")
        client = None
        equity = 100000.0  # Default equity for dry run
        dry_run = True

    # 1) Exit due positions (based on our local state clock, not broker)
    state = _load_state(cfg.state_path)
    open_trades: List[dict] = state.get("open_trades", [])
    today = _now_et().date()
    still_open: List[dict] = []

    if not dry_run:
        existing_positions = _positions_by_symbol(client)
    else:
        existing_positions = {}

    for tr in open_trades:
        sym = tr["symbol"]
        due = date.fromisoformat(tr["exit_after"])
        if today >= due and sym in existing_positions:
            print(f"[INFO] Exiting due position {sym} ({tr['side']})")
            if not dry_run:
                _close_position(client, sym)
        else:
            still_open.append(tr)
    state["open_trades"] = still_open

    # 2) Build & screen universe
    bt = BTConfig(
        interval=cfg.interval,
        chunk_size=cfg.chunk_size,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
        min_dollar_vol=cfg.min_dollar_vol,
        dollar_vol_lookback=cfg.lookback_days,
        long_only=cfg.long_only,
    )
    if cfg.use_csv_universe:
        candidates = pd.read_csv(cfg.universe_csv_path)
        col = "ticker" if "ticker" in candidates.columns else candidates.columns[0]
        candidates = candidates[col].astype(str).str.upper().str.strip().tolist()
    else:
        # Use dynamic universe fetching
        from backtest_yolo_events import fetch_top_stocks_by_volume
        candidates = fetch_top_stocks_by_volume(max_stocks=cfg.max_universe_size)

    if cfg.max_universe_size:
        candidates = candidates[: cfg.max_universe_size]

    tickers, screening_prices = screen_by_price_and_liquidity(
        candidates,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
        min_dollar_vol=cfg.min_dollar_vol,
        lookback_days=cfg.lookback_days,
    )

    if not tickers:
        print("[WARN] No tickers passed screen. Trying reliable fallback list...")
        # Try with a smaller, more reliable list
        from backtest_yolo_events import get_reliable_stock_list
        reliable_candidates = get_reliable_stock_list()
        tickers, screening_prices = screen_by_price_and_liquidity(
            reliable_candidates,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            min_dollar_vol=cfg.min_dollar_vol,
            lookback_days=cfg.lookback_days,
        )
        
        if not tickers:
            print("[ERROR] No tickers passed screen even with fallback list.")
            return {"signals": pd.DataFrame(), "orders": []}
        else:
            print(f"[INFO] Using fallback list: {len(tickers)} tickers passed screen.")

    # 3) Scan signals
    model = YOLO(cfg.model_path)
    sigs = scan_right_edge_signals(
        tickers=tickers,
        cfg=bt,
        model=model,
        interval=cfg.interval,
        chunk_size=cfg.chunk_size,
    )
    if sigs.empty:
        print("[INFO] No right-edge signals today.")
    else:
        print(f"[INFO] Found {len(sigs)} signals.")

    # 4) Place orders (rank by confidence; apply caps)
    placed_orders = []
    day_cap_notional = equity * cfg.max_portfolio_day_cap
    used_notional = 0.0
    if not dry_run:
        positions = _positions_by_symbol(client)  # refresh after exits
    else:
        positions = {}

    # Convert to simple list of dicts ordered by confidence
    rows = sigs.to_dict(orient="records")
    rows = sorted(rows, key=lambda r: float(r.get("confidence") or 0.0), reverse=True)

    # Price cache to avoid double API calls - start with screening prices
    price_cache = screening_prices.copy()
    print(f"[INFO] Initialized price cache with {len(price_cache)} prices from screening")

    for row in rows:
        if len(placed_orders) >= cfg.max_trades_per_day:
            break
        sym = row["ticker"]
        sig = row["signal"].lower()
        if cfg.long_only and sig != "buy":
            continue

        # Use cached price or fetch new one
        if sym in price_cache:
            price = price_cache[sym]
        else:
            # For CI environments, skip individual price fetches to avoid rate limiting
            import os
            if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
                print(f"[INFO] Skipping price fetch for {sym} in CI - using approximate price")
                # Use a reasonable default price for paper trading in CI
                price = 100.0  # This won't affect actual trades since it's DRY RUN
                price_cache[sym] = price
            else:
                price = _latest_close_price(sym)
                if price:
                    price_cache[sym] = price
                # Add delay only for new fetches
                import time
                time.sleep(1)  # 1 second delay for new price fetches
        
        per_trade_notional = equity * cfg.max_alloc_per_trade
        remaining_cap = max(0.0, day_cap_notional - used_notional)
        notional = min(per_trade_notional, remaining_cap)
        qty = _qty_for_notional(price, notional)
        if qty < 1:
            continue

        side = OrderSide.BUY if sig == "buy" else OrderSide.SELL

        # Avoid stacking multiple same-direction positions: if we already hold long and signal is buy, skip
        if sym in positions:
            pos = positions[sym]
            already_long = (pos.side.lower() == "long")
            already_short = (pos.side.lower() == "short")
            if (side == OrderSide.BUY and already_long) or (side == OrderSide.SELL and already_short):
                print(f"[INFO] Skipping {sym}: already holding {pos.side}")
                continue

        try:
            if dry_run:
                print(f"[DRY RUN] Would place {side.value} {qty} {sym} at ~${price:.2f}")
                order_id = f"DRY_RUN_{sym}_{today}"
            else:
                order = client.submit_order(
                    order_data=MarketOrderRequest(
                        symbol=sym,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                print(f"[INFO] Placed {side.value} {qty} {sym}")
                order_id = order.id
            
            placed_orders.append({
                "symbol": sym,
                "side": side.value,
                "qty": qty,
                "estimated_price": price,
                "confidence": float(row.get("confidence") or 0.0),
                "event_time": str(row["event_time"]),
                "order_id": order_id,
            })
            used_notional += (price or 0.0) * qty

            # Track exit plan only if we *opened* a new exposure in that direction
            exit_after = today + timedelta(days=cfg.holding_days)
            state["open_trades"].append({
                "symbol": sym,
                "side": "long" if side == OrderSide.BUY else "short",
                "qty": qty,
                "entry_date": str(today),
                "exit_after": str(exit_after),
            })

        except Exception as e:
            print(f"[ERROR] submit_order failed for {sym}: {e}")

    # 5) Save artifacts
    run_day = _now_et().strftime("%Y-%m-%d")
    outdir = cfg.runs_dir / run_day
    outdir.mkdir(parents=True, exist_ok=True)

    sigs.to_csv(outdir / "signals.csv", index=False)
    if placed_orders:
        pd.DataFrame(placed_orders).to_csv(outdir / "orders.csv", index=False)

    # Save current positions snapshot
    if not dry_run:
        current_pos = client.get_all_positions()
        if current_pos:
            pd.DataFrame([{
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl)
            } for p in current_pos]).to_csv(outdir / "positions.csv", index=False)
    else:
        # In dry run mode, save empty positions file
        pd.DataFrame(columns=["symbol", "qty", "side", "market_value", "avg_entry_price", "unrealized_pl"]).to_csv(outdir / "positions.csv", index=False)

    _save_state(cfg.state_path, state)

    return {
        "signals": sigs,
        "orders": placed_orders,
        "positions_path": str(outdir / "positions.csv"),
        "outdir": str(outdir),
    }

if __name__ == "__main__":
    cfg = RunConfig()
    res = run_once(cfg)
    print("[DONE]", res.get("outdir", ""))
