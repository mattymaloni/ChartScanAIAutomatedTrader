"""
Shared configuration for ChartScanAI trading system.
Used by both live trader and backtester to ensure consistency.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TradingConfig:
    """Unified configuration for live trading and backtesting"""
    
    # ========== Model Settings ==========
    model_path: str = "weights/custom_yolov8.pt"
    conf_threshold: float = 0.25
    img_size: int = 640
    
    # ========== Chart Settings ==========
    interval: str = "1d"  # Daily candles
    chunk_size: int = 180  # Bars per chart
    lookback_bars: int = 180  # Must match chunk_size
    chart_style: str = 'charles'
    figsize: tuple = (16, 6)
    dpi: int = 100
    
    # ========== Universe Settings ==========
    use_csv_universe: bool = False
    universe_csv_path: str = "tickers.csv"
    max_universe_size: int = 1000
    
    # Universe filters
    min_price: float = 5.0
    max_price: float = 3000.0
    min_dollar_vol: float = 1_000_000
    dollar_vol_lookback: int = 20
    
    # ========== Trading Rules ==========
    # Position sizing (MODERATE - tune these!)
    max_alloc_per_trade: float = 0.15  # 15% per position
    max_portfolio_day_cap: float = 0.60  # Max 60% used for new trades per day
    max_positions: int = 10  # Max concurrent positions
    
    # Holding period
    holding_days: int = 5  # TRADING days (Mon-Fri only)
    cooldown_days_after_exit: int = 1  # Days before re-entering same symbol
    
    # Entry/Exit timing
    entry_timing: str = "next_open"  # Enter at next day's open
    exit_timing: str = "close"  # Exit at close of day 5
    
    # Signal filtering
    min_confidence: float = 0.25
    valid_signals: list = None  # ['buy', 'sell']
    long_only: bool = False
    ignore_conflicting_signals: bool = True  # Skip if both buy & sell
    right_edge_only: bool = True  # Only trade right-edge signals
    
    # Risk management
    max_trades_per_day: int = 50
    
    # ========== Backtesting Settings ==========
    backtest_start_date: str = "2025-07-01"  # Updated to 2025
    backtest_end_date: str = "2025-10-01"    # Will default to today if not set
    initial_capital: float = 100000.0
    
    # Costs (set to 0 for now, can enable later)
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.0  # % of price
    
    # ========== File Paths ==========
    state_path: Path = Path("paper_state.json")
    runs_dir: Path = Path("daily_runs")
    backtest_results_dir: Path = Path("backtest_results")
    cache_dir: Path = Path("data_cache")
    
    # Caching
    enable_caching: bool = True
    cache_expiry_hours: int = 6
    
    def __post_init__(self):
        """Initialize default values"""
        if self.valid_signals is None:
            self.valid_signals = ['buy', 'sell']
    
    def validate(self):
        """Validate configuration"""
        assert 0 < self.max_alloc_per_trade <= 1.0, "max_alloc_per_trade must be between 0 and 1"
        assert 0 < self.max_portfolio_day_cap <= 1.0, "max_portfolio_day_cap must be between 0 and 1"
        assert self.holding_days > 0, "holding_days must be positive"
        assert self.conf_threshold >= 0, "conf_threshold must be non-negative"
        assert self.entry_timing in ["next_open", "same_close"], "Invalid entry_timing"
        assert self.exit_timing in ["open", "close"], "Invalid exit_timing"
        
        # Check if position sizing is reasonable
        max_possible_positions = int(1.0 / self.max_alloc_per_trade)
        if self.max_positions > max_possible_positions:
            print(f"WARNING: max_positions ({self.max_positions}) > theoretical max "
                  f"({max_possible_positions}) given {self.max_alloc_per_trade:.1%} per trade")


# ========== Preset Configurations ==========

def get_conservative_config() -> TradingConfig:
    """Conservative: 5-10% per position, max 10-20 positions"""
    config = TradingConfig()
    config.max_alloc_per_trade = 0.08  # 8% per position
    config.max_portfolio_day_cap = 0.40  # 40% max new trades per day
    config.max_positions = 12
    config.holding_days = 7  # Longer holds
    config.cooldown_days_after_exit = 2
    return config


def get_moderate_config() -> TradingConfig:
    """Moderate: 10-20% per position, max 5-10 positions (DEFAULT)"""
    return TradingConfig()  # Uses defaults


def get_aggressive_config() -> TradingConfig:
    """Aggressive: 20-30% per position, max 3-5 positions"""
    config = TradingConfig()
    config.max_alloc_per_trade = 0.25  # 25% per position
    config.max_portfolio_day_cap = 0.80  # 80% max new trades per day
    config.max_positions = 4
    config.holding_days = 3  # Shorter holds
    config.min_confidence = 0.35  # Higher confidence required
    return config


# ========== Usage Example ==========
if __name__ == "__main__":
    # Default moderate config
    config = TradingConfig()
    config.validate()
    
    print("=== ChartScanAI Trading Configuration ===")
    print(f"Position sizing: {config.max_alloc_per_trade:.1%} per trade")
    print(f"Max positions: {config.max_positions}")
    print(f"Holding period: {config.holding_days} trading days")
    print(f"Entry: {config.entry_timing}, Exit: {config.exit_timing}")
    print(f"Min confidence: {config.min_confidence:.1%}")
    print(f"Backtest period: {config.backtest_start_date} to {config.backtest_end_date}")
    
    # Show all preset configs
    print("\n=== Available Presets ===")
    for preset_name, preset_func in [
        ("Conservative", get_conservative_config),
        ("Moderate", get_moderate_config),
        ("Aggressive", get_aggressive_config)
    ]:
        cfg = preset_func()
        max_pos = int(1.0 / cfg.max_alloc_per_trade)
        print(f"{preset_name:12}: {cfg.max_alloc_per_trade:.1%}/trade, "
              f"~{max_pos} positions, {cfg.holding_days}d hold")