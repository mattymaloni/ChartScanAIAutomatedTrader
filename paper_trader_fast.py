#!/usr/bin/env python3
"""
ChartScanAI Live Paper Trader - Matches backtest logic exactly
Scans daily, holds 5 trading days, enters at next open, exits at close
"""
import json
import os
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO

import pandas as pd
import yfinance as yf
from PIL import Image
import matplotlib.pyplot as plt
import mplfinance as mpf

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.models import Position

from ultralytics import YOLO
from trading_config import TradingConfig


# ========== Utilities ==========
def _now_et() -> datetime:
    """Get current time in UTC (GitHub Actions timezone)"""
    return datetime.now(timezone.utc)


def _load_state(path: Path) -> dict:
    """Load persisted state"""
    if path.exists():
        return json.loads(path.read_text())
    return {"open_trades": [], "cooldowns": []}


def _save_state(path: Path, state: dict) -> None:
    """Save state to disk"""
    path.write_text(json.dumps(state, indent=2, default=str))


def _clear_yfinance_cache():
    """Clear yfinance cache to prevent corrupted data"""
    try:
        import shutil
        import platform
        
        if platform.system() == "Darwin":
            cache_dirs = ["~/Library/Caches/py-yfinance"]
        else:
            cache_dirs = ["~/.cache/py-yfinance", "~/.cache/yfinance"]
            
        for cache_path in cache_dirs:
            cache_dir = os.path.expanduser(cache_path)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"[INFO] Cleared yfinance cache: {cache_dir}")
    except Exception as e:
        print(f"[WARN] Failed to clear cache: {e}")


def add_trading_days(start_date: date, days: int) -> date:
    """Add N trading days (Mon-Fri only)"""
    current = start_date
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Weekday
            remaining -= 1
    return current


def _latest_close_price(symbol: str) -> Optional[float]:
    """Get latest close price with retries"""
    import os
    ci_env = os.environ.get('CI', '').lower() in ('true', '1', 'yes')
    github_actions = os.environ.get('GITHUB_ACTIONS', '').lower() in ('true', '1', 'yes')
    is_ci = ci_env or github_actions
    max_retries = 8 if is_ci else 5
    
    for retry in range(max_retries):
        try:
            if is_ci and retry > 0:
                import random
                import time
                time.sleep(random.uniform(2, 8))
            
            df = yf.download(
                symbol, 
                period="2d",
                interval="1d", 
                progress=False, 
                auto_adjust=False,
                timeout=45 if is_ci else 30,
                show_errors=False
            )
            
            if df is None or df.empty:
                if retry < max_retries - 1:
                    import time
                    time.sleep(3)
                    continue
                return None
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                close_col = None
                for col in df.columns:
                    if col[0] == 'Close':
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
                
            last_close = close_series.dropna().iloc[-1]
            return float(last_close)
            
        except Exception as e:
            if retry < max_retries - 1:
                import time
                if is_ci:
                    base_delay = 15 + (retry * 5)
                    import random
                    jitter = random.uniform(0, 5)
                    time.sleep(base_delay + jitter)
                else:
                    time.sleep(5 + (2 ** retry))
                continue
            
            print(f"[WARN] Failed to get price for {symbol}: {e}")
            return None


def _alpaca_client_from_env() -> TradingClient:
    """Create Alpaca client from environment variables"""
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API keys not configured")
    
    return TradingClient(api_key, api_secret, paper=True)


def _positions_by_symbol(client: TradingClient) -> Dict[str, Position]:
    """Get all positions keyed by symbol"""
    positions = client.get_all_positions()
    return {p.symbol: p for p in positions}


def _submit_close_order(client: TradingClient, pos: Position) -> Optional[str]:
    """Submit market order to close position"""
    try:
        symbol = pos.symbol
        qty = int(abs(float(pos.qty)))
        
        if qty <= 0:
            print(f"[WARN] Position {symbol} has zero qty")
            return None
        
        side = OrderSide.SELL if getattr(pos, "side", "long").lower() == "long" else OrderSide.BUY
        
        order = client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )
        print(f"[INFO] Exit order: {symbol} {side.value} {qty} (order_id={order.id})")
        return order.id
    except Exception as e:
        print(f"[ERROR] Failed to close {pos.symbol}: {e}")
        return None


# ========== Chart & Signal Detection ==========
def create_chart(ticker: str, config: TradingConfig) -> Optional[BytesIO]:
    """Generate candlestick chart for signal detection"""
    try:
        df = yf.download(ticker, period="1y", interval=config.interval, 
                        progress=False, auto_adjust=False)
        
        if df.empty or len(df) < config.lookback_bars:
            return None
        
        chart_data = df.tail(config.lookback_bars).copy()
        
        for col in ['Open', 'High', 'Low', 'Close']:
            chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
        chart_data = chart_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(chart_data) < config.lookback_bars * 0.9:
            return None
        
        fig, axes = mpf.plot(
            chart_data,
            type='candle',
            style=config.chart_style,
            figsize=config.figsize,
            volume=False,
            returnfig=True,
            tight_layout=True,
            warn_too_much_data=config.lookback_bars + 100
        )
        
        ax = axes[0] if isinstance(axes, (list, tuple)) else axes
        ax.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=config.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except Exception as e:
        return None


def detect_signals(image_buffer: BytesIO, model: YOLO, config: TradingConfig, 
                  ticker: str) -> List[dict]:
    """Run YOLO detection on chart"""
    try:
        image = Image.open(image_buffer).convert('RGB')
        image_array = np.array(image)
        
        results = model.predict(
            image_array,
            conf=config.conf_threshold,
            verbose=False,
            save=False,
            show=False
        )
        
        signals = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                image_width = image_array.shape[1]
                right_edge_threshold = image_width * 0.8
                
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = model.names[cls_id].lower()
                        
                        if class_name not in config.valid_signals:
                            continue
                        
                        if confidence < config.min_confidence:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x_center = (x1 + x2) / 2
                        is_right_edge = x_center >= right_edge_threshold
                        
                        if config.right_edge_only and not is_right_edge:
                            continue
                        
                        signals.append({
                            'ticker': ticker,
                            'signal': class_name,
                            'confidence': confidence,
                            'right_edge': is_right_edge
                        })
                    except Exception:
                        continue
        
        return signals
        
    except Exception:
        return []


# ========== Main Trading Logic ==========
def run_once(config: TradingConfig) -> dict:
    """
    Daily trading cycle:
    1. Exit positions due today (at close of day 5)
    2. Scan for new signals
    3. Place orders for next day's open
    4. Track state
    """
    _clear_yfinance_cache()
    
    # Setup
    try:
        client = _alpaca_client_from_env()
        account = client.get_account()
        equity = float(account.equity)
        buying_power = float(getattr(account, "buying_power", equity))
        print(f"[INFO] Equity: ${equity:,.2f} | Buying power: ${buying_power:,.2f}")
        dry_run = False
    except ValueError as e:
        print(f"[INFO] {e} - Running in DRY RUN mode")
        client = None
        equity = 100000.0
        buying_power = equity
        dry_run = True
    
    # Load state
    state = _load_state(config.state_path)
    open_trades: List[dict] = state.get("open_trades", [])
    today = _now_et().date()
    
    # ========== Step 1: Exit Due Positions ==========
    still_open = []
    exit_orders = []
    exited_today = set()
    
    if not dry_run:
        existing_positions = _positions_by_symbol(client)
        
        # Backfill missing state from broker
        existing_keys = {(str(t.get("symbol")).upper(), str(t.get("side", "long")).lower()) 
                        for t in open_trades}
        for pos in existing_positions.values():
            symbol = str(getattr(pos, "symbol", "")).upper()
            side = str(getattr(pos, "side", "long")).lower()
            if (symbol, side) not in existing_keys:
                synth_entry = {
                    "symbol": symbol,
                    "side": side,
                    "qty": int(abs(float(getattr(pos, "qty", 0)))),
                    "entry_date": str(today),
                    "exit_after": str(add_trading_days(today, config.holding_days)),
                }
                print(f"[WARN] Backfilled missing state for {symbol}")
                open_trades.append(synth_entry)
                existing_keys.add((symbol, side))
    else:
        existing_positions = {}
    
    closed_count = 0
    for tr in open_trades:
        sym = tr["symbol"]
        due = date.fromisoformat(tr["exit_after"])
        
        if today >= due:
            print(f"[INFO] Position {sym} due for exit")
            if not dry_run and sym in existing_positions:
                pos = existing_positions[sym]
                order_id = _submit_close_order(client, pos)
                if order_id:
                    closed_count += 1
                    exit_orders.append({
                        "symbol": sym,
                        "side": "sell" if getattr(pos, "side", "long").lower() == "long" else "buy",
                        "qty": int(abs(float(pos.qty))),
                        "action": "exit",
                        "order_id": order_id,
                        "date": str(today)
                    })
                else:
                    still_open.append(tr)
                    continue
            elif dry_run:
                print(f"[DRY-RUN] Would close {sym}")
            
            exited_today.add(str(sym).upper())
        else:
            still_open.append(tr)
    
    print(f"[INFO] Closed {closed_count} positions")
    state["open_trades"] = still_open
    
    # Update cooldowns
    cooldowns: List[dict] = state.get("cooldowns", [])
    cooldowns = [c for c in cooldowns if date.fromisoformat(str(c.get("until_date"))) >= today]
    
    for sym in exited_today:
        until = add_trading_days(today, config.cooldown_days_after_exit)
        cooldowns.append({"symbol": sym, "until_date": str(until)})
    
    # Deduplicate cooldowns
    dedup = {}
    for c in cooldowns:
        sym = str(c.get("symbol", "")).upper()
        ud = str(c.get("until_date", str(today)))
        if sym and (sym not in dedup or date.fromisoformat(ud) > date.fromisoformat(dedup[sym])):
            dedup[sym] = ud
    state["cooldowns"] = [{"symbol": k, "until_date": v} for k, v in dedup.items()]
    
    # Blocked symbols
    open_symbols = {str(t.get("symbol", "")).upper() for t in state.get("open_trades", [])}
    cooldown_symbols = {str(c.get("symbol", "")).upper() for c in state.get("cooldowns", []) 
                       if date.fromisoformat(str(c.get("until_date"))) >= today}
    blocked_symbols = open_symbols | cooldown_symbols
    
    # ========== Step 2: Build Universe ==========
    print("[INFO] Building universe...")
    from simple_tester import build_universe_candidates, screen_by_price_and_liquidity
    
    candidates = build_universe_candidates(config)
    if config.max_universe_size:
        candidates = candidates[:config.max_universe_size]
    
    tickers, screening_prices = screen_by_price_and_liquidity(candidates, config)
    
    if not tickers:
        print("[WARN] No tickers passed screening")
        _save_state(config.state_path, state)
        return {"signals": [], "orders": exit_orders}
    
    print(f"[INFO] Screened: {len(tickers)} tickers")
    
    # ========== Step 3: Scan for Signals ==========
    print("[INFO] Scanning for signals...")
    model = YOLO(config.model_path)
    
    all_signals = []
    for ticker in tickers:
        if ticker.upper() in blocked_symbols:
            continue
        
        chart = create_chart(ticker, config)
        if chart is None:
            continue
        
        signals = detect_signals(chart, model, config, ticker)
        all_signals.extend(signals)
    
    if not all_signals:
        print("[INFO] No signals detected")
        _save_state(config.state_path, state)
        return {"signals": [], "orders": exit_orders}
    
    print(f"[INFO] Found {len(all_signals)} signals")
    
    # Check for conflicting signals
    if config.ignore_conflicting_signals:
        ticker_signals = {}
        for sig in all_signals:
            t = sig['ticker']
            ticker_signals.setdefault(t, []).append(sig)
        
        filtered_signals = []
        for ticker, sigs in ticker_signals.items():
            signal_types = {s['signal'] for s in sigs}
            if len(signal_types) == 1:
                filtered_signals.extend(sigs)
            else:
                print(f"[INFO] Ignoring conflicting signals for {ticker}")
        all_signals = filtered_signals
    
    # Sort by confidence
    all_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    # ========== Step 4: Place Orders ==========
    print("[INFO] Placing orders...")
    placed_orders = list(exit_orders)  # Include exits
    
    # Refresh account after exits
    if not dry_run:
        try:
            account = client.get_account()
            equity = float(account.equity)
            buying_power = float(getattr(account, "buying_power", equity))
            positions = _positions_by_symbol(client)
        except Exception as e:
            print(f"[WARN] Failed to refresh account: {e}")
            positions = {}
    else:
        positions = {}
    
    day_cap_notional = equity * config.max_portfolio_day_cap
    used_notional = 0.0
    price_cache = screening_prices.copy()
    
    for sig in all_signals:
        if len([o for o in placed_orders if o.get('action') != 'exit']) >= config.max_trades_per_day:
            break
        
        if len(state["open_trades"]) >= config.max_positions:
            print("[INFO] Max positions reached")
            break
        
        ticker = sig['ticker']
        signal_type = sig['signal']
        
        # Skip if already holding
        if ticker in positions:
            continue
        
        # Get price
        if ticker in price_cache:
            price = price_cache[ticker]
        else:
            price = _latest_close_price(ticker)
            if price:
                price_cache[ticker] = price
            import time
            time.sleep(1)
        
        if not price:
            continue
        
        # Calculate position size
        per_trade_notional = equity * config.max_alloc_per_trade
        remaining_day_cap = max(0.0, day_cap_notional - used_notional)
        available_notional = min(per_trade_notional, remaining_day_cap, buying_power)
        
        qty = int(available_notional / price)
        if qty < 1:
            continue
        
        side = OrderSide.BUY if signal_type == "buy" else OrderSide.SELL
        
        try:
            if dry_run:
                print(f"[DRY RUN] Would place {side.value} {qty} {ticker}")
                order_id = f"DRY_{ticker}_{today}"
            else:
                order = client.submit_order(
                    order_data=MarketOrderRequest(
                        symbol=ticker,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                print(f"[INFO] Placed {side.value} {qty} {ticker}")
                order_id = order.id
            
            placed_orders.append({
                "symbol": ticker,
                "side": side.value,
                "qty": qty,
                "estimated_price": price,
                "confidence": sig['confidence'],
                "order_id": order_id,
                "date": str(today),
                "action": "entry"
            })
            
            order_notional = price * qty
            used_notional += order_notional
            buying_power = max(0.0, buying_power - order_notional)
            
            # Schedule exit (5 TRADING days from tomorrow's entry)
            entry_date = add_trading_days(today, 1)  # Enters tomorrow
            exit_date = add_trading_days(entry_date, config.holding_days)
            
            state["open_trades"].append({
                "symbol": ticker,
                "side": "long" if side == OrderSide.BUY else "short",
                "qty": qty,
                "entry_date": str(entry_date),
                "exit_after": str(exit_date),
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to place order for {ticker}: {e}")
    
    # ========== Step 5: Save Artifacts ==========
    run_day = _now_et().strftime("%Y-%m-%d")
    outdir = config.runs_dir / run_day
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save signals
    if all_signals:
        pd.DataFrame(all_signals).to_csv(outdir / "signals.csv", index=False)
    
    # Save orders
    if placed_orders:
        pd.DataFrame(placed_orders).to_csv(outdir / "orders.csv", index=False)
    
    # Save positions snapshot
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
    
    _save_state(config.state_path, state)
    
    print(f"[INFO] Saved artifacts to {outdir}")
    print(f"[DONE] Signals: {len(all_signals)}, Orders: {len(placed_orders)}")
    
    return {
        "signals": all_signals,
        "orders": placed_orders,
        "outdir": str(outdir)
    }


if __name__ == "__main__":
    config = TradingConfig()
    config.validate()
    result = run_once(config)
    print(f"[COMPLETE] {result.get('outdir', '')}")