#!/usr/bin/env python3
"""
ChartScanAI Backtester - Simulates live trading with YOLO model
Matches paper_trader_fast.py logic exactly for accurate performance testing
"""
import os
import sys
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from io import BytesIO

import pandas as pd
import numpy as np
import yfinance as yf
from PIL import Image
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

warnings.filterwarnings('ignore')
plt.ioff()

try:
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: PyTorch/YOLO not available")
    TORCH_AVAILABLE = False
    sys.exit(1)

from trading_config import TradingConfig


# ========== Parallel Processing Helpers ==========
def process_ticker_for_signals(args):
    """Process a single ticker for signal detection (for parallel execution)"""
    ticker, df, current_date, config_dict, model_path = args
    
    try:
        # Recreate config from dict (can't pickle full config easily)
        config = TradingConfig()
        for key, val in config_dict.items():
            setattr(config, key, val)
        
        # Generate chart
        chart = create_chart_for_date(df, current_date, config)
        if chart is None:
            return []
        
        # Load model (each process needs its own)
        model = YOLO(model_path)
        
        # Detect signals
        signals = detect_signals(chart, model, config, ticker)
        return signals
    except Exception:
        return []


# ========== Universe Building (Self-Contained) ==========
def build_universe_candidates(config: TradingConfig) -> List[str]:
    """Build stock universe - simplified version"""
    # Curated list of liquid stocks
    major_stocks = """
    AAPL MSFT GOOGL GOOG AMZN META NVDA TSLA AVGO ORCL INTC AMD MU QCOM TXN IBM CRM NOW
    JPM BAC WFC C GS MS SCHW BLK BK STT PNC USB COF AXP
    XOM CVX COP SLB HAL EOG PXD MPC VLO
    UNH JNJ PFE MRK ABBV TMO DHR ABT BMY LLY
    PG KO PEP COST WMT TGT KMB CL
    HD LOW NKE SBUX MCD YUM CMG DPZ
    NEE DUK SO AEP EXC XEL SRE
    PLD AMT CCI EQIX DLR SPG O PSA
    CSCO ANET DELL HPE
    MA V PYPL
    NFLX DIS CMCSA FOX
    BA CAT DE GE HON MMM UPS FDX
    CVS WBA CI HUM
    ADBE SAP
    LMT RTX NOC GD
    AMGN GILD BIIB VRTX
    ISRG MDT SYK
    ACN CTSH
    ADSK INTU
    AMAT LRCX KLAC
    """
    
    tickers = set()
    for line in major_stocks.split('\n'):
        if line.strip():
            tickers.update(line.strip().split())
    
    ticker_list = sorted(list(tickers))
    
    if config.max_universe_size:
        ticker_list = ticker_list[:config.max_universe_size]
    
    return ticker_list


def screen_by_price_and_liquidity(
    tickers: List[str],
    config: TradingConfig
) -> Tuple[List[str], Dict[str, float]]:
    """Screen tickers by price and liquidity"""
    print(f"Screening {len(tickers)} tickers...")
    
    passed = []
    prices = {}
    batch_size = 10
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                tickers=batch,
                period=f"{config.dollar_vol_lookback + 5}d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker" if len(batch) > 1 else None,
                threads=False
            )
            
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df_t = data
                    else:
                        if ticker in data.columns.get_level_values(0):
                            df_t = data[ticker]
                        else:
                            continue
                    
                    if isinstance(df_t.columns, pd.MultiIndex):
                        df_t.columns = df_t.columns.droplevel(1)
                    
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df_t.columns for col in required_cols):
                        continue
                    
                    for col in ['Open', 'High', 'Low', 'Close']:
                        df_t[col] = pd.to_numeric(df_t[col], errors='coerce')
                    df_t['Volume'] = pd.to_numeric(df_t['Volume'], errors='coerce').fillna(0)
                    
                    df_t = df_t.dropna(subset=['Open', 'High', 'Low', 'Close'])
                    
                    if len(df_t) < config.dollar_vol_lookback:
                        continue
                    
                    last_price = float(df_t['Close'].iloc[-1])
                    if not (config.min_price <= last_price <= config.max_price):
                        continue
                    
                    recent_data = df_t.tail(config.dollar_vol_lookback)
                    dollar_vol = (recent_data['Close'] * recent_data['Volume']).mean()
                    
                    if dollar_vol >= config.min_dollar_vol:
                        passed.append(ticker)
                        prices[ticker] = last_price
                
                except Exception:
                    continue
        
        except Exception:
            pass
    
    print(f"Passed screening: {len(passed)} tickers")
    return sorted(passed), prices


# ========== Trading Day Utilities ==========
def get_trading_days(start_date: str, end_date: str) -> List[datetime]:
    """Get list of trading days (Mon-Fri) between dates"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_days = pd.date_range(start, end, freq='D')
    # Filter to weekdays only (Mon=0, Sun=6)
    trading_days = [d for d in all_days if d.weekday() < 5]
    return trading_days


def add_trading_days(date: datetime, days: int) -> datetime:
    """Add N trading days to a date (skipping weekends)"""
    current = date
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Weekday
            remaining -= 1
    return current


# ========== Chart Generation ==========
def create_chart_for_date(df: pd.DataFrame, as_of_date: datetime, config: TradingConfig) -> Optional[BytesIO]:
    """Create candlestick chart up to (and including) as_of_date"""
    try:
        # Get data up to as_of_date
        data_up_to = df[df.index <= as_of_date].copy()
        
        if len(data_up_to) < config.lookback_bars:
            return None
        
        # Take last N bars
        chart_data = data_up_to.tail(config.lookback_bars).copy()
        
        # Ensure numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
        chart_data = chart_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(chart_data) < config.lookback_bars * 0.9:
            return None
        
        # Create chart matching training format
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


def detect_signals(image_buffer: BytesIO, model: YOLO, config: TradingConfig, ticker: str) -> List[dict]:
    """Run YOLO detection on chart, return tradable signals"""
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
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x_center = (x1 + x2) / 2
                        is_right_edge = x_center >= right_edge_threshold
                        
                        # Filter by right edge if required
                        if config.right_edge_only and not is_right_edge:
                            continue
                        
                        # Filter by confidence
                        if confidence < config.min_confidence:
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


# ========== Position Tracking ==========
@dataclass
class Position:
    ticker: str
    side: str  # 'long' or 'short'
    qty: int
    entry_price: float
    entry_date: datetime
    exit_date: datetime  # Scheduled exit date
    
    def value_at_price(self, price: float) -> float:
        """Calculate position value"""
        if self.side == 'long':
            return self.qty * price
        else:  # short
            return self.qty * (2 * self.entry_price - price)
    
    def pnl_at_price(self, price: float) -> float:
        """Calculate P&L"""
        if self.side == 'long':
            return self.qty * (price - self.entry_price)
        else:  # short
            return self.qty * (self.entry_price - price)


@dataclass
class Trade:
    ticker: str
    side: str
    qty: int
    entry_price: float
    entry_date: datetime
    exit_price: float
    exit_date: datetime
    pnl: float
    pnl_pct: float
    holding_days: int
    signal_confidence: float


# ========== Backtester ==========
class Backtester:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.positions: List[Position] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        self.daily_stats = []
        
    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity"""
        position_value = sum(p.value_at_price(prices.get(p.ticker, p.entry_price)) 
                           for p in self.positions)
        return self.cash + position_value
    
    def get_available_cash(self) -> float:
        """Cash available for new trades"""
        return self.cash
    
    def can_trade(self, signal_type: str) -> bool:
        """Check if we can take more positions"""
        if len(self.positions) >= self.config.max_positions:
            return False
        return True
    
    def open_position(self, ticker: str, signal: str, price: float, 
                     entry_date: datetime, confidence: float) -> Optional[Position]:
        """Open a new position"""
        # Calculate position size
        equity = self.get_equity({ticker: price})
        max_notional = equity * self.config.max_alloc_per_trade
        available = min(max_notional, self.cash)
        
        if available < price:  # Can't afford even 1 share
            return None
        
        qty = int(available / price)
        if qty < 1:
            return None
        
        cost = qty * price
        self.cash -= cost
        
        side = 'long' if signal == 'buy' else 'short'
        exit_date = add_trading_days(entry_date, self.config.holding_days)
        
        pos = Position(
            ticker=ticker,
            side=side,
            qty=qty,
            entry_price=price,
            entry_date=entry_date,
            exit_date=exit_date
        )
        self.positions.append(pos)
        return pos
    
    def close_position(self, pos: Position, price: float, exit_date: datetime) -> Trade:
        """Close a position and record trade"""
        pnl = pos.pnl_at_price(price)
        proceeds = pos.qty * price if pos.side == 'long' else pos.qty * (2 * pos.entry_price - price)
        self.cash += proceeds
        
        holding_days = len(pd.bdate_range(pos.entry_date, exit_date))
        
        trade = Trade(
            ticker=pos.ticker,
            side=pos.side,
            qty=pos.qty,
            entry_price=pos.entry_price,
            entry_date=pos.entry_date,
            exit_price=price,
            exit_date=exit_date,
            pnl=pnl,
            pnl_pct=(pnl / (pos.qty * pos.entry_price)) * 100,
            holding_days=holding_days,
            signal_confidence=0.0  # Set by caller
        )
        self.closed_trades.append(trade)
        self.positions.remove(pos)
        return trade
    
    def run(self, stock_data: Dict[str, pd.DataFrame]) -> dict:
        """Run backtest"""
        # Get trading days
        end_date = self.config.backtest_end_date if self.config.backtest_end_date else datetime.now().strftime("%Y-%m-%d")
        trading_days = get_trading_days(self.config.backtest_start_date, end_date)
        
        print(f"\n{'='*60}")
        print(f"ChartScanAI Backtest: {self.config.backtest_start_date} to {end_date}")
        print(f"Universe: {len(stock_data)} tickers")
        print(f"Trading days: {len(trading_days)}")
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Position sizing: {self.config.max_alloc_per_trade:.1%} per trade")
        print(f"Holding period: {self.config.holding_days} trading days")
        print(f"{'='*60}\n")
        
        # Track cooldowns
        cooldowns: Dict[str, datetime] = {}  # ticker -> cooldown_until_date
        
        # Progress bar
        pbar = tqdm(trading_days, desc="Backtesting", unit="day")
        
        for current_date in pbar:
            # Get current prices for all tickers
            current_prices = {}
            for ticker, df in stock_data.items():
                try:
                    price_data = df[df.index <= current_date]
                    if not price_data.empty:
                        current_prices[ticker] = float(price_data['Close'].iloc[-1])
                except:
                    continue
            
            # Check for exits
            positions_to_close = [p for p in self.positions if current_date >= p.exit_date]
            for pos in positions_to_close:
                exit_price = current_prices.get(pos.ticker, pos.entry_price)
                trade = self.close_position(pos, exit_price, current_date)
                
                # Set cooldown
                cooldown_until = add_trading_days(current_date, self.config.cooldown_days_after_exit)
                cooldowns[pos.ticker] = cooldown_until
            
            # Clean expired cooldowns
            cooldowns = {t: d for t, d in cooldowns.items() if d > current_date}
            
            # Skip signal scanning if already at max positions
            if len(self.positions) >= self.config.max_positions:
                continue
            
            # Scan for new signals (OPTIMIZED WITH PARALLEL PROCESSING)
            signals_today = []
            
            # Filter tickers to scan (skip cooldowns and existing positions)
            tickers_to_scan = []
            for ticker, df in stock_data.items():
                # Skip if in cooldown
                if ticker in cooldowns:
                    continue
                
                # Skip if already holding
                if any(p.ticker == ticker for p in self.positions):
                    continue
                
                tickers_to_scan.append((ticker, df))
            
            # Parallel chart generation and signal detection
            if tickers_to_scan:
                # Prepare config as dict for pickling
                config_dict = {
                    'lookback_bars': self.config.lookback_bars,
                    'chart_style': self.config.chart_style,
                    'figsize': self.config.figsize,
                    'dpi': 50,  # Lower DPI for faster backtesting
                    'conf_threshold': self.config.conf_threshold,
                    'min_confidence': self.config.min_confidence,
                    'valid_signals': self.config.valid_signals,
                    'right_edge_only': self.config.right_edge_only,
                }
                
                # Process in parallel (use half CPU cores to avoid overload)
                max_workers = max(1, os.cpu_count() // 2)
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = []
                    for ticker, df in tickers_to_scan:
                        args = (ticker, df, current_date, config_dict, self.config.model_path)
                        futures.append(executor.submit(process_ticker_for_signals, args))
                    
                    # Collect results as they complete
                    for future in futures:
                        try:
                            signals = future.result(timeout=5)
                            signals_today.extend(signals)
                        except Exception:
                            continue
                
                # Add date and price info to signals
                for sig in signals_today:
                    sig['date'] = current_date
                    sig['price'] = current_prices.get(sig['ticker'], 0)
            
            # Check for conflicting signals
            if self.config.ignore_conflicting_signals:
                ticker_signals = {}
                for sig in signals_today:
                    t = sig['ticker']
                    ticker_signals.setdefault(t, []).append(sig)
                
                # Remove tickers with both buy and sell
                filtered_signals = []
                for ticker, sigs in ticker_signals.items():
                    signal_types = {s['signal'] for s in sigs}
                    if len(signal_types) == 1:  # Only buy OR sell, not both
                        filtered_signals.extend(sigs)
                signals_today = filtered_signals
            
            # Sort by confidence and take top signals
            signals_today.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Enter new positions (at next day's open)
            next_day = add_trading_days(current_date, 1)
            new_positions_today = 0
            day_cap_used = 0.0
            
            for sig in signals_today:
                if not self.can_trade(sig['signal']):
                    break
                
                # Check day cap
                equity = self.get_equity(current_prices)
                if day_cap_used >= self.config.max_portfolio_day_cap * equity:
                    break
                
                # Get next day open price (approximate with close)
                ticker = sig['ticker']
                if ticker not in stock_data:
                    continue
                
                next_day_data = stock_data[ticker][stock_data[ticker].index > current_date]
                if next_day_data.empty:
                    continue
                
                entry_price = float(next_day_data['Open'].iloc[0])
                
                # Open position
                pos = self.open_position(ticker, sig['signal'], entry_price, next_day, sig['confidence'])
                if pos:
                    new_positions_today += 1
                    day_cap_used += pos.qty * entry_price
                
                if new_positions_today >= self.config.max_trades_per_day:
                    break
            
            # Record daily stats
            equity = self.get_equity(current_prices)
            self.equity_curve.append({
                'date': current_date,
                'equity': equity,
                'cash': self.cash,
                'positions': len(self.positions),
                'trades_today': new_positions_today
            })
            
            # Update progress bar
            pbar.set_postfix({
                'Equity': f'${equity:,.0f}',
                'Positions': len(self.positions),
                'Trades': len(self.closed_trades)
            })
        
        pbar.close()
        
        # Close any remaining positions at end
        if self.positions:
            print(f"\nClosing {len(self.positions)} remaining positions at backtest end...")
            for pos in list(self.positions):
                final_price = current_prices.get(pos.ticker, pos.entry_price)
                self.close_position(pos, final_price, trading_days[-1])
        
        return self.generate_report()
    
    def generate_report(self) -> dict:
        """Generate performance report"""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate metrics
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        trades_df = pd.DataFrame([{
            'ticker': t.ticker,
            'side': t.side,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'qty': t.qty,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'holding_days': t.holding_days
        } for t in self.closed_trades])
        
        if len(trades_df) > 0:
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() if (trades_df['pnl'] > 0).any() else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if (trades_df['pnl'] < 0).any() else 0
            avg_trade_pnl = trades_df['pnl'].mean()
            total_pnl = trades_df['pnl'].sum()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_trade_pnl = 0
            total_pnl = 0
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        report = {
            'config': {
                'start_date': self.config.backtest_start_date,
                'end_date': self.config.backtest_end_date or datetime.now().strftime("%Y-%m-%d"),
                'initial_capital': self.initial_capital,
                'position_size': f"{self.config.max_alloc_per_trade:.1%}",
                'holding_days': self.config.holding_days,
                'max_positions': self.config.max_positions
            },
            'performance': {
                'final_equity': final_equity,
                'total_return_pct': total_return,
                'total_pnl': total_pnl,
                'max_drawdown_pct': max_drawdown,
                'total_trades': len(self.closed_trades),
                'win_rate_pct': win_rate,
                'avg_trade_pnl': avg_trade_pnl,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss
            },
            'equity_curve': equity_df,
            'trades': trades_df
        }
        
        return report


# ========== Main ==========
def main():
    # Load config
    config = TradingConfig()
    config.validate()
    
    # Set backtest end to today if not specified
    if not config.backtest_end_date:
        config.backtest_end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Verify date range is valid
    start = pd.to_datetime(config.backtest_start_date)
    end = pd.to_datetime(config.backtest_end_date)
    today = pd.to_datetime(datetime.now())
    
    print(f"\n{'='*60}")
    print(f"Date Configuration:")
    print(f"  Backtest Start: {config.backtest_start_date}")
    print(f"  Backtest End: {config.backtest_end_date}")
    print(f"  Today: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    
    if start > today:
        print(f"ERROR: Start date {config.backtest_start_date} is in the future!")
        print(f"Please set backtest_start_date to a past date.")
        return
    
    if end > today:
        print(f"WARNING: End date {config.backtest_end_date} is in the future. Using today instead.")
        config.backtest_end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start >= end:
        print(f"ERROR: Start date must be before end date!")
        return
    
    days_diff = (end - start).days
    if days_diff < 30:
        print(f"WARNING: Backtest period is only {days_diff} days. Recommend at least 60 days.")
    
    print("Loading model...")
    if not Path(config.model_path).exists():
        print(f"ERROR: Model not found at {config.model_path}")
        return
    
    # Build universe
    print("Building universe...")
    candidates = build_universe_candidates(config)
    screened, _ = screen_by_price_and_liquidity(candidates, config)
    
    if not screened:
        print("ERROR: No tickers passed screening")
        return
    
    print(f"Screened universe: {len(screened)} tickers")
    
    # Download data
    print("Downloading historical data...")
    
    # Add buffer to download period to ensure we have enough lookback data
    download_start = (pd.to_datetime(config.backtest_start_date) - timedelta(days=365)).strftime("%Y-%m-%d")
    download_end = config.backtest_end_date
    
    print(f"Date range: {download_start} to {download_end}")
    stock_data = {}
    failed_count = 0
    
    for ticker in tqdm(screened, desc="Downloading"):
        try:
            df = yf.download(
                ticker, 
                start=download_start, 
                end=download_end, 
                interval="1d",
                progress=False,
                auto_adjust=False
            )
            
            if df is not None and not df.empty:
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Ensure we have required columns
                required_cols = ['Open', 'High', 'Low', 'Close']
                if all(col in df.columns for col in required_cols):
                    # Clean data
                    for col in required_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=required_cols)
                    
                    # Only keep if we have enough data
                    if len(df) >= config.lookback_bars:
                        stock_data[ticker] = df
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Downloaded data for {len(stock_data)} tickers (failed: {failed_count})")
    
    if not stock_data:
        print("ERROR: No data downloaded")
        return
    
    # Run backtest
    backtester = Backtester(config)
    results = backtester.run(stock_data)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Period: {results['config']['start_date']} to {results['config']['end_date']}")
    print(f"Initial capital: ${results['config']['initial_capital']:,.0f}")
    print(f"Final equity: ${results['performance']['final_equity']:,.0f}")
    print(f"Total return: {results['performance']['total_return_pct']:.2f}%")
    print(f"Max drawdown: {results['performance']['max_drawdown_pct']:.2f}%")
    print(f"\nTrades: {results['performance']['total_trades']}")
    print(f"Win rate: {results['performance']['win_rate_pct']:.1f}%")
    print(f"Avg trade P&L: ${results['performance']['avg_trade_pnl']:.2f}")
    print(f"Avg win: {results['performance']['avg_win_pct']:.2f}%")
    print(f"Avg loss: {results['performance']['avg_loss_pct']:.2f}%")
    
    # Save results
    output_dir = config.backtest_results_dir
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['trades'].to_csv(output_dir / f"trades_{timestamp}.csv", index=False)
    results['equity_curve'].to_csv(output_dir / f"equity_curve_{timestamp}.csv")
    
    # Save summary
    summary = {
        'config': results['config'],
        'performance': results['performance']
    }
    with open(output_dir / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    print("✅ Backtest complete!")


if __name__ == "__main__":
    main()