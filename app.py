import os, logging, threading, requests, time, numpy as np, pandas as pd, csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
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

SYMBOLS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_JPY","GBP_JPY"]

# Risk & Performance
BASE_RISK = 0.0025
MAX_DAILY_TRADES = 12
MAX_OPEN_TRADES = 5
DAILY_LOSS_LIMIT = 0.02
COOLDOWN_MINUTES = 20
MAX_SPREAD_PIPS = 2.5
MIN_UNITS = 1000
MAX_UNIT_CAP = 100000

# Spike / News Hunter
SPIKE_SENSITIVITY = 1.6
HUNTER_WINDOW_SECONDS = 60

JOURNAL_FILE = "trade_journal.csv"
ENGINE_LOCK = threading.Lock()

# Global State
SESSION_START_NAV = None
LAST_TRADE_TIME = {} 
DAILY_TRADE_COUNT = 0 
ENGINE_HALTED = False

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-ELITE] | %(message)s")

# ==================== UTILITIES ====================

def compute_atr_true_range(symbol, period=14):
    """Institutional ATR using High/Low/Close for accuracy."""
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": period+1, "granularity": "M15"})
        client.request(r)
        candles = r.response["candles"]
        trs = []
        for i in range(1, len(candles)):
            h = float(candles[i]["mid"]["h"])
            l = float(candles[i]["mid"]["l"])
            pc = float(candles[i-1]["mid"]["c"])
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return float(pd.Series(trs).mean())
    except: return None

def reset_daily_metrics():
    global SESSION_START_NAV, DAILY_TRADE_COUNT, LAST_TRADE_TIME, ENGINE_HALTED
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        SESSION_START_NAV = float(r.response["account"]["NAV"])
        DAILY_TRADE_COUNT, ENGINE_HALTED, LAST_TRADE_TIME = 0, False, {}
        send_telegram(f"🌅 SESSION RESET: NAV @ {SESSION_START_NAV}")
    except: logging.error("Reset Failed")

def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except: pass

# ==================== FILTERS ====================

def get_open_positions_count():
    """Live check of OANDA to prevent duplicate/ghost trades."""
    try:
        r = positions.OpenPositions(OANDA_ACCOUNT_ID); client.request(r)
        return len(r.response.get("positions", []))
    except: return 99 # Safety block

def is_market_active():
    now = datetime.now(timezone.utc)
    # Allows trading 24/5 - adjust if you want to avoid specific sessions
    return now.weekday() < 5

# ==================== ENGINE ====================

def run_apex_cycle(symbol):
    global ENGINE_HALTED, SESSION_START_NAV, DAILY_TRADE_COUNT
    if ENGINE_HALTED or SESSION_START_NAV is None: return
    
    with ENGINE_LOCK:
        try:
            # 1. Kill-switch & Portfolio Checks
            ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
            nav = float(ra.response["account"]["NAV"])
            drawdown = max(0, (SESSION_START_NAV - nav) / SESSION_START_NAV)
            
            if drawdown >= DAILY_LOSS_LIMIT:
                ENGINE_HALTED = True
                send_telegram("🛑 KILL-SWITCH ACTIVATED. Engine Halted.")
                return

            if get_open_positions_count() >= MAX_OPEN_TRADES: return
            if DAILY_TRADE_COUNT >= MAX_DAILY_TRADES: return

            # 2. Data Acquisition
            atr = compute_atr_true_range(symbol)
            r = instruments.InstrumentsCandles(symbol, {"count": 50, "granularity": "M15"})
            client.request(r)
            data = [float(c['mid']['c']) for c in r.response['candles']]
            price = data[-1]
            
            if not atr: return

            # 3. GEAR 1: NEWS HUNTER (The Friend Strategy)
            hunter_side = get_hunter_signal(symbol) # Assumes your scrape function is active
            if hunter_side:
                execute_trade(symbol, hunter_side, price, atr, "HUNTER", nav, drawdown)
                return

            # 4. GEAR 2: SPIKE DETECTION
            recent_move = abs(data[-1] - data[-2])
            if recent_move > (atr * SPIKE_SENSITIVITY):
                execute_trade(symbol, "BUY" if data[-1] > data[-2] else "SELL", price, atr, "SPIKE", nav, drawdown)
                return

            # 5. GEAR 3: NORMAL RSI/EMA (When calm)
            # [Insert RSI/EMA logic from your code here]
            
        except Exception as e:
            logging.error(f"Cycle Error {symbol}: {e}")

# ... (Include execute_trade, scrape_high_impact_news, and Flask routes)
