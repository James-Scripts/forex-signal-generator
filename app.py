import os, logging, threading, requests, time, numpy as np, pandas as pd, ta, csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, StopLossDetails, TakeProfitDetails, TrailingStopLossDetails

# ================= CONFIGURATION =================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_JPY", "GBP_JPY", "EUR_AUD", "AUD_JPY", "CAD_JPY"
]

# Risk & Performance Settings
BASE_RISK = 0.0025         
MAX_DAILY_TRADES = 10       
MAX_OPEN_TRADES = 5         
DAILY_LOSS_LIMIT = 0.02    
COOLDOWN_MINUTES = 30      
MAX_SPREAD_PIPS = 2.2      

# Spike Logic Constants
SPIKE_SENSITIVITY = 1.6    
MIN_VOLATILITY_RATIO = 0.00015 

JOURNAL_FILE = "trade_journal.csv"
ENGINE_LOCK = threading.Lock()
PORTFOLIO_LOCK = threading.Lock()

# Global State
SESSION_START_NAV = None
LAST_TRADE_TIME = {} 
OPEN_POSITIONS = {}  
HIGH_IMPACT_EVENTS = [] 
DAILY_TRADE_COUNT = 0 
ENGINE_HALTED = False
PAIR_CACHE = {}       

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V16-PRO] | %(message)s")

# ==================== UTILITIES ====================

def compute_atr_from_candles(symbol, period=14):
    """Calculates True Range ATR to properly capture market gaps."""
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": period+1, "granularity": "M15"})
        client.request(r)
        candles = r.response["candles"]
        trs = []
        for i in range(1, len(candles)):
            high = float(candles[i]["mid"]["h"])
            low = float(candles[i]["mid"]["l"])
            prev_c = float(candles[i-1]["mid"]["c"])
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))
            trs.append(tr)
        return float(pd.Series(trs).mean())
    except Exception as e:
        logging.error(f"ATR Error {symbol}: {e}")
        return None

def reset_daily_metrics():
    global SESSION_START_NAV, DAILY_TRADE_COUNT, LAST_TRADE_TIME, ENGINE_HALTED
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        SESSION_START_NAV = float(r.response["account"]["NAV"])
        DAILY_TRADE_COUNT = 0
        ENGINE_HALTED = False
        LAST_TRADE_TIME = {}
        send_telegram(f"🌅 ENGINE RESET: NAV Anchor @ {SESSION_START_NAV}")
    except: pass

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except: pass

def log_to_journal(symbol, side, price, atr, rsi, ema, units):
    file_exists = os.path.isfile(JOURNAL_FILE)
    try:
        with open(JOURNAL_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Symbol", "Side", "Price", "ATR", "RSI/Mode", "EMA", "Units"])
            writer.writerow([datetime.now(timezone.utc), symbol, side, price, round(atr,5), rsi, round(ema,5), units])
    except: pass

# ==================== FILTERS ====================

def is_near_news(symbol, window_min=45):
    now = datetime.now(timezone.utc)
    base, quote = symbol.split("_")
    for curr, dt in HIGH_IMPACT_EVENTS:
        if curr in (base, quote) and abs((dt - now).total_seconds()) <= window_min * 60:
            return True
    return False

def is_spread_ok(symbol):
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": 1, "granularity": "M1", "price": "BA"})
        client.request(r)
        c = r.response["candles"][-1]
        spread = (float(c["ask"]["c"]) - float(c["bid"]["c"])) / (0.01 if "JPY" in symbol else 0.0001)
        return spread <= MAX_SPREAD_PIPS
    except: return False

def is_portfolio_safe(symbol):
    with PORTFOLIO_LOCK:
        if len(OPEN_POSITIONS) >= MAX_OPEN_TRADES: return False
        if symbol in OPEN_POSITIONS: return False
        # Currency exposure check
        open_ccys = [c for p in OPEN_POSITIONS.keys() for c in p.split('_')]
        for c in symbol.split('_'):
            if open_ccys.count(c) >= 3: return False
    return True

# ==================== EXECUTION ====================

def execute_trade(symbol, side, price, atr, mode_label, rsi_val, nav, drawdown_pct):
    global DAILY_TRADE_COUNT, LAST_TRADE_TIME
    try:
        risk_mod = max(0.25, 1 - (drawdown_pct / DAILY_LOSS_LIMIT))
        units = int((nav * (BASE_RISK * risk_mod)) / (atr * 3))
        if units < 1000: return
        
        units = min(units, 100000)
        if side == "SELL": units = -units

        prec = 3 if "JPY" in symbol else 5
        trail = 1.2 if mode_label == "SPIKE" else 1.5
        
        sl = round(price - atr*3 if side=="BUY" else price + atr*3, prec)
        tp = round(price + atr*2.5 if side=="BUY" else price - atr*2.5, prec)
        
        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(sl)).data,
            takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * trail, prec))).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        
        log_to_journal(symbol, side, price, atr, rsi_val if mode_label=="NORMAL" else "SPIKE", 0, units)
        LAST_TRADE_TIME[symbol] = datetime.now(timezone.utc)
        DAILY_TRADE_COUNT += 1
        send_telegram(f"✅ {mode_label} {side}: {symbol} @ {price}")
    except Exception as e: logging.error(f"Trade Error: {e}")

# ==================== ENGINE ====================

def run_apex_cycle(symbol):
    global ENGINE_HALTED
    with ENGINE_LOCK:
        if ENGINE_HALTED or SESSION_START_NAV is None: return
        if DAILY_TRADE_COUNT >= MAX_DAILY_TRADES: return
        
        # 1. Account Safety Check
        ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
        nav = float(ra.response["account"]["NAV"])
        drawdown = max(0, (SESSION_START_NAV - nav) / SESSION_START_NAV)
        
        if drawdown >= DAILY_LOSS_LIMIT:
            ENGINE_HALTED = True
            send_telegram("🛑 ENGINE HALTED: Daily loss limit hit.")
            return

        # 2. Volatility & Cooldown check
        last_t = LAST_TRADE_TIME.get(symbol)
        if last_t and (datetime.now(timezone.utc) - last_t).total_seconds() < COOLDOWN_MINUTES * 60: return

        atr = compute_atr_from_candles(symbol)
        r = instruments.InstrumentsCandles(symbol, {"count": 3, "granularity": "M15"})
        client.request(r); data = [float(c['mid']['c']) for c in r.response['candles']]
        price = data[-1]

        if not atr or (atr / price) < MIN_VOLATILITY_RATIO: return

        # --- MODE A: SPIKE CHASER (Momentum Continuation) ---
        recent_move = abs(data[-1] - data[-2])
        prev_move = abs(data[-2] - data[-3])
        
        if recent_move > (atr * SPIKE_SENSITIVITY) and recent_move > (prev_move * 1.3):
            if is_spread_ok(symbol) and is_portfolio_safe(symbol):
                side = "BUY" if data[-1] > data[-2] else "SELL"
                execute_trade(symbol, side, price, atr, "SPIKE", None, nav, drawdown)
                return

        # --- MODE B: NORMAL TREND ---
        if is_near_news(symbol) or not is_spread_ok(symbol): return
        
        df = pd.DataFrame({'c': get_historical_closes(symbol)})
        rsi = ta.momentum.rsi(df['c'], 14).iloc[-1]
        ema20 = df['c'].ewm(span=20).mean().iloc[-1]
        
        if rsi < 30 and price > ema20 and is_portfolio_safe(symbol):
            execute_trade(symbol, "BUY", price, atr, "NORMAL", rsi, nav, drawdown)
        elif rsi > 70 and price < ema20 and is_portfolio_safe(symbol):
            execute_trade(symbol, "SELL", price, atr, "NORMAL", rsi, nav, drawdown)

# ... (Include standard /run and /status Flask routes as discussed) ...
