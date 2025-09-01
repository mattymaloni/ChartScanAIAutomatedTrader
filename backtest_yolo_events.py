# backtest_yolo_events.py
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image
from tqdm.auto import tqdm

# plotting / image
import matplotlib.pyplot as plt
import mplfinance as mpf

# accel / model
import torch
from ultralytics import YOLO


# ============================== logging / utils ==============================

def get_logger(name: str = "yolo_backtest", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
    return logger


def seed_all(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================== config ==============================

@dataclass(frozen=True)
class Config:
    # --- runtime / device ---
    seed: int = 42
    device: str = field(default_factory=lambda:
                        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- model ---
    model_path: str = "custom_yolov8.pt"
    yolo_conf: float = 0.30
    yolo_imgsz: int = 768

    # --- universe ---
    universe_source: str = "dynamic"     # "static" | "csv" | "dynamic"
    universe_csv_path: str = "tickers.csv"
    max_universe_size: int = 1000

    # --- universe filters ---
    max_price: float = 3000.0
    min_price: float = 5.0
    min_dollar_vol: float = 1e6  # Lowered to 1M to include more stocks
    dollar_vol_lookback: int = 70

    # --- test selection ---
    interval: str = "1d"                 # "1d" or "1h" ("1h" -> "60m")
    chunk_size: int = 180
    days: int = 60
    stride: int = 1
    holding_bars_list: Tuple[int, ...] = (1, 5, 10, 20)
    auto_adjust: bool = False

    # --- chart rendering (YOLO input) ---
    figsize: Tuple[float, float] = (18, 6.5)
    dpi: int = 100
    mpf_style: str = "yahoo"
    axis_off: bool = True
    show_volume: bool = False
    mpf_kwargs: dict = field(default_factory=lambda: {
        "figratio": (16, 9),
        "figscale": 1.5,
        "update_width_config": {"candle_width": 0.6, "candle_linewidth": 0.4},
        "warn_too_much_data": 181,
    })

    # --- trading controls ---
    start_cash: float = 100_000.0
    long_only: bool = False                 # False = allow shorts on "sell"
    max_alloc_per_trade: float = 0.30
    portfolio_day_cap: float = 1.00         # fraction of equity usable across ALL tickers per day
    per_ticker_day_cap: Optional[float] = 0.30
    one_position_per_ticker: bool = True

    # --- persistence ---
    save_dir: str = "runs"
    save_trades: bool = True
    save_summary: bool = True
    save_detections: bool = False


# ============================== data helpers ==============================

FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def _normalize_yf_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Accepts a yfinance multi-index or flat dataframe and returns a flat OHLCV frame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # If the outermost level contains field names (Open, High, ...) pull a column per field for the chosen ticker
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        fields_set = set(FIELDS)

        # Case A: fields in level 0 (Open, High, ...) / tickers in level 1
        if fields_set.issubset(set(lvl0)):
            tickers = list(dict.fromkeys(lvl1))
            chosen = ticker if ticker in lvl1 else tickers[0]
            df = df.xs(chosen, axis=1, level=1, drop_level=True)
        # Case B: fields in level 1 / tickers in level 0
        elif fields_set.issubset(set(lvl1)):
            tickers = list(dict.fromkeys(lvl0))
            chosen = ticker if ticker in lvl0 else tickers[0]
            df = df.xs(chosen, axis=1, level=0, drop_level=True)
        else:
            # Fallback: flatten columns
            df = df.copy()
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns]

    # Standardize column names
    rename = {}
    for c in df.columns:
        s = str(c).lower()
        if s.startswith("adj close") or s == "close":
            rename[c] = "Close"
        elif s.startswith("open"):
            rename[c] = "Open"
        elif s.startswith("high"):
            rename[c] = "High"
        elif s.startswith("low"):
            rename[c] = "Low"
        elif s.startswith("volume"):
            rename[c] = "Volume"

    if rename:
        df = df.rename(columns=rename)

    # Drop dup columns and keep only needed ones
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if not keep:
        return pd.DataFrame()

    out = df[keep].copy()

    # Coerce numeric + drop NaNs in OHLC
    for c in ("Open", "High", "Low", "Close"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[c for c in ("Open", "High", "Low", "Close") if c in out.columns])

    # Index to tz-naive datetime and sort
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index()


def _compute_fetch_period_days(cfg: Config) -> int:
    """
    History needed = chunk_size + days windows + max holding horizon (+ small headroom).
    Convert trading bars to calendar days (~252 trading days/year) + 10% cushion.
    """
    h_max = max(cfg.holding_bars_list) if cfg.holding_bars_list else 1
    required_trading_bars = cfg.chunk_size + cfg.days + h_max + 2
    calendar_days = math.ceil(required_trading_bars * (365.0 / 252.0))
    return max(math.ceil(calendar_days * 1.10), 30)


def load_candidates_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tickers = df[col].astype(str).str.upper().str.strip()
    return sorted({t for t in tickers if t and t.isalnum()})


def load_candidates_static() -> List[str]:
    """Fallback static list for when dynamic fetching fails."""
    static = """
AAPL MSFT GOOGL AMZN META NVDA TSLA AVGO ORCL INTC AMD MU QCOM TXN IBM CRM NOW
JPM BAC WFC C GS MS SCHW BLK BK STT PNC USB
XOM CVX COP SLB HAL
UNH JNJ PFE MRK ABBV TMO DHR ABT
PG KO PEP COST WMT TGT
HD LOW NKE SBUX MCD YUM
NEE DUK SO AEP EXC
PLD AMT CCI EQIX DLR SPG O
CSCO ANET DELL
MA V PYPL
NFLX DIS CMCSA
SPY QQQ IWM VTI VOO
"""
    return sorted(set(static.split()))


def get_reliable_stock_list() -> List[str]:
    """Get a smaller, more reliable list of liquid stocks for when API issues occur."""
    reliable = """
AAPL MSFT GOOGL AMZN META NVDA TSLA AVGO ORCL INTC AMD
JPM BAC WFC C GS MS
XOM CVX COP
UNH JNJ PFE MRK ABBV
PG KO PEP COST WMT
HD LOW NKE
CSCO
MA V
NFLX DIS
SPY QQQ IWM VTI VOO
"""
    return sorted(set(reliable.split()))


def fetch_top_stocks_by_volume(max_stocks: int = 1000, cache_days: int = 7) -> List[str]:
    """
    Fetch top stocks by dollar volume from multiple sources.
    Caches the result for cache_days to avoid repeated API calls.
    """
    from datetime import datetime, timedelta
    import json
    
    cache_file = Path("top_stocks_cache.json")
    now = datetime.now()
    
    # Check if we have a recent cache
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_date = datetime.fromisoformat(cache_data['date'])
            if (now - cache_date).days < cache_days:
                print(f"[INFO] Using cached top stocks from {cache_date.strftime('%Y-%m-%d')}")
                return cache_data['tickers'][:max_stocks]
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")
    
    print(f"[INFO] Fetching top {max_stocks} stocks by volume...")
    
    try:
        tickers = set()
        
        # Method 1: Comprehensive static list of major stocks
        major_stocks = """
        # S&P 500 Major Companies
        AAPL MSFT GOOGL GOOG AMZN META NVDA TSLA AVGO ORCL INTC AMD MU QCOM TXN IBM CRM NOW
        JPM BAC WFC C GS MS SCHW BLK BK STT PNC USB COF AXP
        XOM CVX COP SLB HAL EOG PXD MPC VLO
        UNH JNJ PFE MRK ABBV TMO DHR ABT BMY LLY
        PG KO PEP COST WMT TGT KMB CL
        HD LOW NKE SBUX MCD YUM CMG DPZ PZZA
        NEE DUK SO AEP EXC XEL SRE
        PLD AMT CCI EQIX DLR SPG O PSA
        CSCO ANET DELL HPE
        MA V PYPL
        NFLX DIS CMCSA FOX
        BA CAT DE GE HON MMM UPS FDX
        CVS WBA CI HUM ANTM
        ADBE CRM ORCL SAP
        LMT RTX NOC GD
        TMO DHR ABT BMY LLY
        AMGN GILD BIIB VRTX
        ISRG MDT SYK
        ACN CTSH
        ADSK INTU
        AMAT LRCX KLAC
        AVGO QCOM TXN
        BKR HAL SLB
        CME ICE
        COO DHR
        CPB GIS K
        D DUK SO
        EBAY ETSY
        EL F
        EXC NEE
        FIS FISV
        GE HON
        GPN FISV
        HCA UHS
        HSY K
        ICE CME
        ILMN
        ISRG
        JCI
        KDP PEP
        KHC K
        LH
        LMT
        LRCX
        MCD
        MDLZ K
        MDT
        MMM
        MO
        MRK
        MSI
        MTB
        NCLH
        NEE
        NOC
        NTRS
        NUE
        NVDA
        NXPI
        O
        ODFL
        OGN
        OMC
        ORLY
        OXY
        PAYX
        PCAR
        PEG
        PEP
        PFE
        PG
        PGR
        PH
        PHM
        PKG
        PLD
        PM
        PNC
        PPG
        PPL
        PRU
        PSA
        PSX
        PWR
        PXD
        QCOM
        QRVO
        RCL
        REG
        REGN
        RF
        RHI
        RJF
        RL
        RMD
        ROK
        ROP
        ROST
        RSG
        RTX
        SBAC
        SBUX
        SCHW
        SEDG
        SHW
        SIVB
        SJM
        SLB
        SNA
        SNPS
        SO
        SPG
        SRE
        STE
        STT
        STX
        STZ
        SWK
        SWKS
        SYF
        SYK
        SYY
        T
        TAP
        TDG
        TDY
        TEL
        TER
        TFC
        TFX
        TGT
        TJX
        TMO
        TPR
        TRMB
        TROW
        TRV
        TSCO
        TSLA
        TSN
        TT
        TTD
        TTWO
        TWTR
        TXN
        TXT
        TYL
        UAA
        UAL
        UDR
        UHS
        ULTA
        UNH
        UNP
        UPS
        URI
        USB
        V
        VFC
        VICI
        VLO
        VMC
        VRSK
        VRSN
        VRTX
        VTI
        VTRS
        VZ
        WAB
        WAT
        WBA
        WEC
        WELL
        WFC
        WLTW
        WM
        WMB
        WMT
        WRB
        WST
        WTW
        WY
        WYNN
        XEL
        XOM
        XRAY
        XYL
        YUM
        ZBH
        ZBRA
        ZION
        ZTS
        
        # Popular ETFs
        SPY QQQ IWM VTI VOO VEA VWO EFA EEM AGG TLT LQD HYG
        VUG VTV VYM VNQ VGT VHT VDE VFH VOX VCR VDC VDY
        ARKK ARKW ARKQ ARKG ARKF
        TQQQ SQQQ UPRO SPXU
        TMF TBT
        GLD SLV
        XLE XLF XLI XLK XLV XLY XLP XLU XLB XLC XLY XLP XLU XLB XLC
        
        # Chinese ADRs
        BABA JD PDD NIO XPEV LI BIDU TME VIPS YMM BILI TAL EDU
        ZTO YMM VIPS TME BIDU BABA JD PDD NIO XPEV LI
        
        # Crypto-related
        COIN MSTR
        
        # Meme stocks
        GME AMC BBBY
        
        # Other popular stocks
        ROKU ZM PTON
        
        # Additional S&P 500 companies
        A ADI ADM AES AFL AIV ALGN ALK ALLE ALXN AMCR AMD AME AMG AMP AMT ANSS ANTM AOS APA APD APH APTV ARE ARNC ATR AVB AVY AWK AXP AZO BA BAX BDX BEN BF.B BIIB BK BKNG BLK BLL BMY BR BRK.B BSX BWA BXP C CAG CAH CAT CB CBOE CBRE CCI CCL CDNS CDW CE CERN CF CHD CHRW CHTR CI CINF CL CLX CMA CMCSA CME CMG CMCSA CMS CNP CNX COF COG COO COP COST CPB CPG CPRI CPRT CPT CRL CRM CSCO CSGP CSX CTAS CTLT CTSH CTVA CTXS CVS CVX CXO D DAL DD DE DFS DG DGX DHI DHR DIS DISCA DISCK DISH DLTR DOV DOW DRE DRI DTE DUK DVA DVN DXC EA EBAY ECL ED EFX EIX EL EMN EMR ENPH EOG EPAM EQIX EQR ES ESS ETN ETR ETSY EVRG EW EXC EXPD EXPE EXR F FANG FAST FB FBHS FCX FDX FE FFIV FIS FISV FITB FLT FMC FOX FOXA FRC FRT FTI FTV GD GE GILD GIS GL GLW GM GOOG GOOGL GPC GPN GPS GRMN GS GWW HAL HAS HBAN HBI HCA HCSG HD HES HIG HII HLT HOG HOLX HON HPE HPQ HRL HSIC HST HSY HUM HWM IBM ICE IDXX IEX IFF ILMN INCY INFO INTC INTU INVH IP IPG IPGP IQV IR IRM ISRG IT ITW IVZ JBHT JCI JKHY JLL JNJ JNPR JPM JWN K KBWR KDP KEM KEX KEY KEYS KHC KIM KKR KLAC KMB KMI KMX KO KR KRC KSU L LAD LH LKQ LLY LMT LNC LNT LOW LRCX LUMN LUV LVS LW LYB LYV MA MAA MAR MAS MCD MCHP MCK MCO MDLZ MDT MDU MEC MGM MHK MKC MKTX MLM MMC MNST MO MOH MOS MPC MPWR MRK MRO MS MSFT MSI MSM MTB MTD MU MXIM MYL NCLH NDAQ NEE NEM NFLX NKE NLOK NLSN NOC NOV NOW NRG NSC NTAP NTRS NUE NVDA NVR NWL NWS NWSA NXPI NXST O ODFL OGN OKE OMC ORCL ORLY OXY PAYX PCAR PEAK PEG PEP PFE PFG PG PGR PH PHM PKG PKI PLD PM PNC PNR PNW PPG PPL PRGO PRU PSA PSX PTC PWR PXD PYPL QCOM QRVO RCL REG REGN RF RHI RJF RL RMD ROK ROP ROST RSG RTX SBAC SBUX SCHW SEDG SHW SIVB SJM SLB SNA SNPS SO SPG SRE STE STT STX STZ SWK SWKS SYF SYK SYY T TAP TDG TDY TEL TER TFC TFX TGT TJX TMO TPR TRMB TROW TRV TSCO TSLA TSN TT TTD TTWO TWTR TXN TXT TYL UAA UAL UDR UHS ULTA UNH UNP UPS URI USB V VFC VICI VLO VMC VRSK VRSN VRTX VTI VTRS VZ WAB WAT WBA WEC WELL WFC WLTW WM WMB WMT WRB WST WTW WY WYNN XEL XOM XRAY XYL YUM ZBH ZBRA ZION ZM ZTO ZTS
        """
        
        # Parse the major stocks list
        for line in major_stocks.split('\n'):
            if line.strip() and not line.strip().startswith('#'):
                tickers.update(line.strip().split())
        
        # Method 2: Try to get additional stocks from web sources
        try:
            import requests
            # Try to get S&P 500 from Wikipedia
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(sp500_url, timeout=15)
            if response.status_code == 200:
                import re
                # More comprehensive regex patterns
                patterns = [
                    r'<td><a[^>]*>([A-Z]{1,5})</a></td>',
                    r'<td>([A-Z]{1,5})</td>',
                    r'<a[^>]*>([A-Z]{1,5})</a>',
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, response.text)
                    tickers.update(matches)
                print(f"[INFO] Found {len(matches)} additional tickers from S&P 500")
        except Exception as e:
            print(f"[WARN] Failed to fetch additional tickers: {e}")
        
        # Convert to sorted list and limit
        ticker_list = sorted(list(tickers))[:max_stocks]
        
        print(f"[INFO] Total unique tickers collected: {len(ticker_list)}")
        
        # Cache the result
        cache_data = {
            'date': now.isoformat(),
            'tickers': ticker_list,
            'count': len(ticker_list)
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"[INFO] Cached {len(ticker_list)} tickers")
        except Exception as e:
            print(f"[WARN] Failed to cache tickers: {e}")
        
        return ticker_list
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch top stocks: {e}")
        print("[INFO] Falling back to static list")
        return load_candidates_static()


def build_universe_candidates(cfg: Config) -> List[str]:
    if cfg.universe_source == "csv":
        tickers = load_candidates_from_csv(cfg.universe_csv_path)
    elif cfg.universe_source == "dynamic":
        tickers = fetch_top_stocks_by_volume(max_stocks=cfg.max_universe_size)
    else:
        tickers = load_candidates_static()
    
    return tickers[: cfg.max_universe_size] if cfg.max_universe_size else tickers


# ============================== screening / download ==============================

def _extract_close_vol(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if df is None or df.empty:
        return None, None
    close = df.get("Close")
    vol = df.get("Volume")
    if close is None or vol is None:
        return None, None
    close = pd.to_numeric(close, errors="coerce").dropna()
    vol = pd.to_numeric(vol.reindex(close.index), errors="coerce").fillna(0)
    return close, vol


def screen_by_price_and_liquidity(
    tickers: List[str],
    *,
    min_price: float,
    max_price: float,
    min_dollar_vol: float,
    lookback_days: int,
    batch_size: int = 5,  # Very small batch size to avoid rate limits
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    if not tickers:
        return []
    log = logger or get_logger()

    passed: List[str] = []
    failed_downloads = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        # Add retry logic for failed downloads
        max_retries = 3
        for retry in range(max_retries):
            try:
                data = yf.download(
                    tickers=batch,
                    period=f"{lookback_days + 10}d",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,  # Disable threading to avoid database locks
                )
                break  # Success, exit retry loop
            except Exception as e:
                if retry == max_retries - 1:
                    log.warning(f"Failed to download batch after {max_retries} retries: {e}")
                    failed_downloads.extend(batch)
                    continue
                else:
                    import time
                    time.sleep(5 + (2 ** retry))  # Longer delays: 6s, 7s, 9s
                    continue
        
        # Add delay between batches to avoid rate limiting
        if i + batch_size < len(tickers):  # Don't delay after the last batch
            import time
            time.sleep(5)  # 5 second delay between batches
        
        if data is None or data.empty:
            failed_downloads.extend(batch)
            continue
            
        for t in batch:
            try:
                df_t = data[t] if isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0) \
                    else data.xs(t, axis=1, level=0, drop_level=False)
            except Exception:
                failed_downloads.append(t)
                continue
            df_t = _normalize_yf_frame(df_t, t)
            close, vol = _extract_close_vol(df_t)
            if close is None or vol is None or len(close) < lookback_days:
                failed_downloads.append(t)
                continue
            last_px = float(close.iloc[-1])
            if not (min_price <= last_px <= max_price):
                continue
            dv = (close * vol).tail(lookback_days).mean()
            if float(dv) >= min_dollar_vol:
                passed.append(t)

    if failed_downloads:
        log.warning(f"Failed downloads: {len(failed_downloads)} tickers")
        if len(failed_downloads) <= 10:
            log.warning(f"Failed tickers: {failed_downloads}")

    out = sorted(set(passed))
    log.info("Screened %d/%d tickers (price %.2f–%.2f, avg $%s/day over %dd).",
             len(out), len(tickers), min_price, max_price, f"{min_dollar_vol:,.0f}", lookback_days)
    return out


def download_prices_batched(
    tickers: List[str],
    cfg: Config,
    batch_size: int = 180,
) -> Dict[str, pd.DataFrame]:
    """
    Download only the needed history based on chunk_size + days (+ cushion).
    """
    iv = "60m" if cfg.interval == "1h" else cfg.interval
    period_days = _compute_fetch_period_days(cfg)
    # yfinance restricts very short intervals to ~60 days; for our intervals this mapping is OK.
    period = f"{period_days}d"

    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        raw = yf.download(
            tickers=batch,
            period=period,
            interval=iv,
            auto_adjust=cfg.auto_adjust,
            actions=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        for t in batch:
            try:
                df_t = raw[t] if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0) \
                    else raw.xs(t, axis=1, level=0, drop_level=False)
            except Exception:
                continue
            clean = _normalize_yf_frame(df_t, t)
            if not clean.empty:
                out[t] = clean
    return out


# ============================== chart & model ==============================

def render_window_png(price_df: pd.DataFrame, start: int, cfg: Config) -> Optional[dict]:
    """
    Render a chunk_size-bar candlestick window starting at `start` from `price_df`.
    Returns a dict with PNG buffer and the window index for mapping.
    """
    end = start + cfg.chunk_size
    window_df = price_df.iloc[start:end]
    if window_df.empty or len(window_df) < cfg.chunk_size:
        return None

    fig, _ = mpf.plot(
        window_df,
        type="candle",
        style=cfg.mpf_style,
        axisoff=cfg.axis_off,
        figsize=cfg.figsize,
        returnfig=True,
        **cfg.mpf_kwargs,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=cfg.dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return {"buffer": buf, "window_index": window_df.index}


def yolo_predict_png(png_buffer: BytesIO, model: YOLO, cfg: Config):
    im = Image.open(png_buffer).convert("RGB")
    arr = np.array(im)
    res = model.predict(arr, conf=cfg.yolo_conf, imgsz=cfg.yolo_imgsz, verbose=False, device=cfg.device)
    return res[0]


def _extract_signal_and_conf(box, names: Dict[int, str]) -> Tuple[Optional[str], Optional[float]]:
    cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else None
    name_l = str(names.get(cls_id, "")).strip().lower() if cls_id is not None else ""
    if "buy" in name_l:
        sig = "buy"
    elif "sell" in name_l or "seel" in name_l:  # tolerate occasional misspelling
        sig = "sell"
    else:
        sig = None
    conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else None
    return sig, conf


def map_detections_to_bars(
    ticker: str,
    window_index: pd.DatetimeIndex,
    result,
    names: Dict[int, str],
) -> List[dict]:
    """
    Map YOLO boxes -> candle index -> timestamp. We normalize x centers across detected boxes
    (avoid margins). Only mapping; filtering to right edge happens later.
    """
    boxes = getattr(result, "boxes", None)
    n = len(window_index)
    if boxes is None or n == 0:
        return []

    xs: List[float] = []
    for b in boxes:
        try:
            xs.append(float(b.xywh[0][0]))
        except Exception:
            pass
    if not xs:
        return []

    xmin, xmax = min(xs), max(xs)
    span = max(xmax - xmin, 1e-6)

    rows = []
    for b in boxes:
        try:
            x, _, w, _ = b.xywh[0].tolist()
        except Exception:
            continue

        frac = (x - xmin) / span  # 0..1 across actual plotted candles
        idx = int(np.clip(round(frac * (n - 1)), 0, n - 1))

        sig, conf = _extract_signal_and_conf(b, names)
        rows.append({
            "ticker": ticker,
            "event_time": window_index[idx],
            "bar_index": idx,
            "signal": sig,
            "confidence": float(conf) if conf is not None else None,
            "x_center_px": float(x),
            "width_px": float(w),
        })
    return rows


# ============================== backtest core ==============================

def _build_window_starts(prices_by_ticker: Dict[str, pd.DataFrame], cfg: Config, log: logging.Logger) -> Dict[str, List[int]]:
    starts_by_ticker: Dict[str, List[int]] = {}
    for t, px in prices_by_ticker.items():
        n = len(px)
        need = cfg.chunk_size + (max(cfg.holding_bars_list) if cfg.holding_bars_list else 1) + 1
        if n < need:
            log.warning("Skipping %s: not enough bars (%d < %d).", t, n, need)
            continue
        last_start = n - cfg.chunk_size
        num = min(cfg.days, last_start + 1)
        first_start = max(0, last_start - (num - 1))
        starts_by_ticker[t] = list(range(first_start, last_start + 1, cfg.stride))
    return starts_by_ticker


def _dedupe_detections(det_df: pd.DataFrame, long_only: bool) -> pd.DataFrame:
    if det_df.empty:
        return det_df
    if long_only:
        det_df = det_df[det_df["signal"] == "buy"].copy()

    det_df["event_time"] = pd.to_datetime(det_df["event_time"])
    det_df = (
        det_df.sort_values(["confidence"], ascending=False)
              .drop_duplicates(subset=["ticker", "event_time", "signal"], keep="first")
              .sort_values(["event_time", "ticker"])
              .reset_index(drop=True)
    )
    return det_df


def _prepare_orders_for_horizon(
    det_df: pd.DataFrame, prices_by_ticker: Dict[str, pd.DataFrame], H: int
) -> List[dict]:
    orders: List[dict] = []
    for _, r in det_df.iterrows():
        tkr = r["ticker"]
        sig = r["signal"]
        conf = float(r.get("confidence", 0.0) or 0.0)
        px = prices_by_ticker.get(tkr)
        if px is None or px.empty or "Close" not in px.columns:
            continue
        t = pd.Timestamp(r["event_time"])

        if t in px.index:
            pos = px.index.get_loc(t)
            if isinstance(pos, slice):
                pos = pos.start
        else:
            pos = px.index.searchsorted(t, side="right") - 1
            if pos < 0:
                continue

        entry_pos = pos + 1
        exit_pos = entry_pos + H
        if exit_pos >= len(px.index):
            continue

        entry_time = px.index[entry_pos]
        exit_time = px.index[exit_pos]
        entry_px = float(px.iloc[entry_pos]["Close"])
        exit_px = float(px.iloc[exit_pos]["Close"])

        orders.append({
            "ticker": tkr,
            "signal": sig,
            "confidence": conf,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_px,
            "exit_price": exit_px,
        })
    return orders


def _run_timeline(
    orders: List[dict],
    prices_by_ticker: Dict[str, pd.DataFrame],
    cfg: Config,
) -> Tuple[float, pd.DataFrame]:
    """
    Simulate chronologically: close exits due at t, then open entries at t.
    Enforce caps + optional one position per ticker.
    """
    equity = float(cfg.start_cash)
    executed: List[dict] = []

    # Build event timeline
    from collections import defaultdict

    entries_by_t, exits_by_t = {}, {}
    for o in orders:
        entries_by_t.setdefault(o["entry_time"], []).append(o)
        exits_by_t.setdefault(o["exit_time"], []).append(o)

    day_base_equity: Dict[pd.Timestamp, float] = {}
    day_port_notional = defaultdict(float)
    day_tkr_notional = defaultdict(float)
    open_pos_keys = set()  # track (ticker, entry_time)

    all_times = sorted(set(entries_by_t.keys()) | set(exits_by_t.keys()))

    for t in all_times:
        day_key = pd.Timestamp(t).normalize()

        # 1) Close exits due at t
        for o in exits_by_t.get(t, []):
            key = (o["ticker"], o["entry_time"])
            if cfg.one_position_per_ticker and key not in open_pos_keys:
                continue  # never opened (budget may have blocked it)
            if o["signal"] == "buy":
                pnl = (o["exit_price"] - o["entry_price"]) * o["_shares"]
            else:
                pnl = (o["entry_price"] - o["exit_price"]) * o["_shares"]
            equity += o["_notional"] + pnl
            open_pos_keys.discard(key)

        # 2) Open entries at t (greedy by confidence)
        if t in entries_by_t:
            if day_key not in day_base_equity:
                day_base_equity[day_key] = equity

            for o in sorted(entries_by_t[t], key=lambda z: z["confidence"], reverse=True):
                tkr = o["ticker"]
                if cfg.one_position_per_ticker and any(k[0] == tkr for k in open_pos_keys):
                    continue  # already holding this ticker

                # caps
                per_trade_cash_cap = equity * cfg.max_alloc_per_trade
                remaining_port_cap = float("inf")
                if cfg.portfolio_day_cap is not None:
                    day_cap = cfg.portfolio_day_cap * day_base_equity[day_key]
                    used = day_port_notional[day_key]
                    remaining_port_cap = max(0.0, day_cap - used)

                remaining_tkr_cap = float("inf")
                if cfg.per_ticker_day_cap is not None:
                    tkr_cap = cfg.per_ticker_day_cap * day_base_equity[day_key]
                    used_tkr = day_tkr_notional[(day_key, tkr)]
                    remaining_tkr_cap = max(0.0, tkr_cap - used_tkr)

                spend_cap = min(per_trade_cash_cap, remaining_port_cap, remaining_tkr_cap)
                px = max(o["entry_price"], 1e-9)
                shares = math.floor(min(spend_cap, equity) / px)
                if shares < 1:
                    continue

                notional = shares * px
                if notional > equity:
                    continue

                equity -= notional
                o["_shares"] = shares
                o["_notional"] = notional
                open_pos_keys.add((tkr, o["entry_time"]))

                if cfg.portfolio_day_cap is not None:
                    day_port_notional[day_key] += notional
                if cfg.per_ticker_day_cap is not None:
                    day_tkr_notional[(day_key, tkr)] += notional

                if o["signal"] == "buy":
                    pnl = (o["exit_price"] - o["entry_price"]) * shares
                    ret = o["exit_price"] / o["entry_price"] - 1.0
                    side = "long"
                else:
                    pnl = (o["entry_price"] - o["exit_price"]) * shares
                    ret = o["entry_price"] / o["exit_price"] - 1.0
                    side = "short"

                executed.append({
                    "ticker": tkr,
                    "side": side,
                    "entry_time": o["entry_time"],
                    "exit_time": o["exit_time"],
                    "entry_price": o["entry_price"],
                    "exit_price": o["exit_price"],
                    "shares": shares,
                    "pnl": pnl,
                    "return": ret,
                    "signal_conf": float(o.get("confidence", 0.0) or 0.0),
                    "holding_bars": (o.get("holding_bars") or 0),  # optional; may be set by caller
                })

    trades_df = pd.DataFrame(executed).sort_values("entry_time").reset_index(drop=True)
    return equity, trades_df


def single_test_multi_horizons(
    prices_by_ticker: Dict[str, pd.DataFrame],
    model: YOLO,
    cfg: Config,
    logger: Optional[logging.Logger] = None,
) -> dict:
    log = logger or get_logger()
    names = {int(k): str(v) for k, v in getattr(model, "names", {}).items()}

    # ---- Build window starts ----
    starts_by_ticker = _build_window_starts(prices_by_ticker, cfg, log)
    total_windows = sum(len(v) for v in starts_by_ticker.values())
    if total_windows == 0:
        return {"summaries": {}, "trades": {}, "detections": {"total": 0, "eligible": {h: 0 for h in cfg.holding_bars_list}}, "all_detections": pd.DataFrame()}

    # ---- Collect right-edge detections ----
    det_rows: List[dict] = []
    with tqdm(total=total_windows, desc="Windows", unit="win", dynamic_ncols=True, mininterval=0.5) as pbar:
        for tkr, px in starts_by_ticker.items():
            pxt = prices_by_ticker[tkr]
            for start in px:
                chart = render_window_png(pxt, start, cfg)
                if chart is None:
                    pbar.update(1)
                    continue
                r0 = yolo_predict_png(chart["buffer"], model, cfg)
                rows = map_detections_to_bars(tkr, chart["window_index"], r0, names)
                if rows:
                    last_bar = len(chart["window_index"]) - 1
                    for r in rows:
                        if r.get("signal") and int(r["bar_index"]) == last_bar:
                            det_rows.append(r)
                pbar.update(1)

    det_df_all = pd.DataFrame(det_rows)
    total_dets = int(len(det_df_all))
    if det_df_all.empty:
        empty = {
            "summaries": {h: {"trades": 0, "final_cash": cfg.start_cash, "total_return": 0.0, "win_rate": None, "avg_return": None} for h in cfg.holding_bars_list},
            "trades": {h: pd.DataFrame() for h in cfg.holding_bars_list},
            "detections": {"total": total_dets, "eligible": {h: 0 for h in cfg.holding_bars_list}},
            "all_detections": det_df_all,
        }
        return empty

    det_df_all = _dedupe_detections(det_df_all, cfg.long_only)

    out_summaries, out_trades, eligible_counts = {}, {}, {}

    # ---- For each horizon, simulate chronologically ----
    for H in cfg.holding_bars_list:
        orders = _prepare_orders_for_horizon(det_df_all, prices_by_ticker, H)
        if not orders:
            out_summaries[H] = {"trades": 0, "final_cash": cfg.start_cash, "total_return": 0.0, "win_rate": None, "avg_return": None}
            out_trades[H] = pd.DataFrame()
            eligible_counts[H] = 0
            continue

        # tag horizon for reporting
        for o in orders:
            o["holding_bars"] = H

        eligible_counts[H] = len(orders)
        final_equity, trades_df = _run_timeline(orders, prices_by_ticker, cfg)

        summary = {
            "trades": int(len(trades_df)),
            "final_cash": float(final_equity),
            "total_return": float(final_equity / cfg.start_cash - 1.0) if cfg.start_cash else 0.0,
            "win_rate": float((trades_df["return"] > 0).mean()) if not trades_df.empty else None,
            "avg_return": float(trades_df["return"].mean()) if not trades_df.empty else None,
        }
        out_summaries[H] = summary
        out_trades[H] = trades_df

    return {
        "summaries": out_summaries,
        "trades": out_trades,
        "detections": {"total": total_dets, "eligible": {h: len(out_trades[h]) for h in cfg.holding_bars_list}},
        "all_detections": det_df_all,
    }


# ============================== signal scanning ==============================

def scan_right_edge_signals(
    tickers: List[str],
    cfg: Config,
    model: YOLO,
    interval: str = "1d",
    chunk_size: int = 180,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Scan a list of tickers for right-edge signals (detections on the last bar of the window).
    Returns a DataFrame with columns: ticker, event_time, signal, confidence
    """
    log = logger or get_logger()
    names = {int(k): str(v) for k, v in getattr(model, "names", {}).items()}
    
    rows = []
    failed_scans = []
    
    for ticker in tqdm(tickers, desc="Scanning signals", unit="ticker"):
        max_retries = 2
        for retry in range(max_retries):
            try:
                # Download recent data
                iv = "60m" if interval == "1h" else interval
                if interval == "1h":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=730)
                    data = yf.download(ticker, interval=iv, start=start_date, end=end_date, progress=False, auto_adjust=False)
                else:
                    data = yf.download(ticker, interval=iv, period="max", progress=False, auto_adjust=False)
                
                if data is None or data.empty:
                    if retry < max_retries - 1:
                        import time
                        time.sleep(3)  # Longer delay
                        continue
                    failed_scans.append(ticker)
                    break
                    
                data = _normalize_yf_frame(data, ticker)
                if data.empty or len(data) < chunk_size:
                    failed_scans.append(ticker)
                    break
                    
                # Get the latest window
                window = data.iloc[-chunk_size:].copy()
                if window.empty:
                    failed_scans.append(ticker)
                    break
                    
                # Render chart
                chart = render_window_png(window, 0, cfg)
                if chart is None:
                    failed_scans.append(ticker)
                    break
                    
                # Run YOLO prediction
                result = yolo_predict_png(chart["buffer"], model, cfg)
                
                # Map detections to bars and filter for right-edge only
                detections = map_detections_to_bars(ticker, chart["window_index"], result, names)
                last_bar = len(chart["window_index"]) - 1
                
                for det in detections:
                    if det.get("signal") and int(det["bar_index"]) == last_bar:
                        rows.append({
                            "ticker": ticker,
                            "event_time": det["event_time"],
                            "signal": det["signal"],
                            "confidence": det["confidence"],
                        })
                break  # Success, exit retry loop
                        
            except Exception as e:
                if retry < max_retries - 1:
                    import time
                    time.sleep(5 + (2 ** retry))  # Longer delays: 6s, 7s, 9s
                    continue
                log.warning(f"Error scanning {ticker}: {e}")
                failed_scans.append(ticker)
                break
        
        # Add delay between stock scans to avoid rate limiting
        import time
        time.sleep(2)  # 2 second delay between stock scans
    
    if failed_scans:
        log.warning(f"Failed to scan {len(failed_scans)} tickers")
        if len(failed_scans) <= 10:
            log.warning(f"Failed tickers: {failed_scans}")
    
    return pd.DataFrame(rows)


# ============================== persistence ==============================

def save_results(res: dict, cfg: Config, log: Optional[logging.Logger] = None) -> Path:
    """
    Save trades per horizon, combined trades, detections (optional), summaries, and frozen config.
    Returns the output directory path.
    """
    log = log or get_logger()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(cfg.save_dir) / f"single_test_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save per-horizon & combined trades
    if cfg.save_trades:
        any_trades = False
        frames = []
        for H, df in (res.get("trades") or {}).items():
            path = outdir / f"trades_H{H}.csv"
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_csv(path, index=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                any_trades = True
                frames.append(df.assign(holding_bars=H))
        if any_trades:
            pd.concat(frames, ignore_index=True).to_csv(outdir / "trades_all.csv", index=False)

    if cfg.save_detections and "all_detections" in res:
        (res["all_detections"] or pd.DataFrame()).to_csv(outdir / "detections.csv", index=False)

    if cfg.save_summary:
        with open(outdir / "summary.json", "w") as f:
            json.dump(res.get("summaries", {}), f, indent=2)

    # freeze run config
    with open(outdir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # manifest (helps future reproducibility checks)
    manifest = {
        "timestamp": ts,
        "device": cfg.device,
        "interval": cfg.interval,
        "chunk_size": cfg.chunk_size,
        "days": cfg.days,
        "holding_bars_list": cfg.holding_bars_list,
        "long_only": cfg.long_only,
        "start_cash": cfg.start_cash,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Saved outputs to: %s", str(outdir))
    return outdir


# ============================== main ==============================

def main(cfg: Config = Config()) -> None:
    log = get_logger(level=logging.INFO)
    log.info("Using device: %s", cfg.device)
    seed_all(cfg.seed)

    # 0) Sanity: model exists
    if not os.path.exists(cfg.model_path):
        log.error("MODEL_PATH not found: %s", cfg.model_path)
        return

    # 1) Load model
    model = YOLO(cfg.model_path).to(cfg.device)

    # 2) Build & screen universe
    candidates = build_universe_candidates(cfg)
    tickers = screen_by_price_and_liquidity(
        candidates,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
        min_dollar_vol=cfg.min_dollar_vol,
        lookback_days=cfg.dollar_vol_lookback,
        logger=log,
    )
    if not tickers:
        log.error("No tickers passed the screen.")
        return

    # 3) Download just what's needed
    prices_by_ticker = download_prices_batched(tickers, cfg)
    if not prices_by_ticker:
        log.error("No price data downloaded.")
        return

    # 4) Run the single test
    res = single_test_multi_horizons(prices_by_ticker, model, cfg, logger=log)
    detc = res.get("detections", {"total": 0, "eligible": {h: 0 for h in cfg.holding_bars_list}})

    # 5) Report (console)
    log.info("=== Single Test — INTERVAL=%s, CHUNK=%d, last %d windows, start cash $%.0f, long_only=%s ===",
             cfg.interval, cfg.chunk_size, cfg.days, cfg.start_cash, cfg.long_only)
    log.info("Detections (right-edge only): %d total", detc["total"])
    for H in cfg.holding_bars_list:
        s = res["summaries"].get(H, {})
        log.info("--- Hold %d bar(s) ---", H)
        log.info("Executed trades: %s", s.get("trades", 0))
        wr, ar = s.get("win_rate"), s.get("avg_return")
        if wr is None:
            log.info("Win Rate: NA | Avg Ret/Trade: NA")
        else:
            log.info("Win Rate: %.2f%% | Avg Ret/Trade: %.4f", wr * 100, ar or 0.0)
        if "final_cash" in s and "total_return" in s:
            log.info("Final Cash: $%.2f  Total Return: %.2f%%", s["final_cash"], s["total_return"] * 100)

        td = res["trades"].get(H, pd.DataFrame())
        if isinstance(td, pd.DataFrame) and not td.empty:
            # Print last few without overwhelming logs
            tail = td.tail(10)[["ticker", "side", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "pnl", "signal_conf"]]
            log.info("Recent trades (tail 10):\n%s", tail.to_string(index=False))

    # 6) Persist
    if cfg.save_trades or cfg.save_summary or (cfg.save_detections and "all_detections" in res):
        save_results(res, cfg, log)


if __name__ == "__main__":
    main()
