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
BASE_RISK = 0.0030  # Slightly increased to help "put food on the table"
MAX_DAILY_TRADES = 20
MAX_OPEN_TRADES = 5
DAILY_LOSS_LIMIT = 0.025
COOLDOWN_MINUTES = 15
MAX_SPREAD_PIPS = 2.5

# Engine Tuning (Loosened for more frequency)
SPIKE_SENSITIVITY = 0.9      # Was 1.6 (Will catch many more momentum moves)
HUNTER_WINDOW_SECONDS = 300  # Was 60s (5-minute window for news moves)
EMA_FILTER_PERIOD = 200      # The "Bodyguard" trend line

JOURNAL_FILE = "trade_journal.csv"
ENGINE_LOCK = threading.Lock()

# Global State
SESSION_START_NAV = None
LAST_TRADE_TIME = {} 
DAILY_TRADE_COUNT = 0 
ENGINE_HALTED = False
HIGH_IMPACT_EVENTS = []

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-SURVIVOR] | %(message)s")

# ==================== UTILITIES ====================

def compute_indicators(symbol):
    try:
        # Fetch 250 candles to ensure enough data for 200 EMA and RSI
        r = instruments.InstrumentsCandles(symbol, {"count": 250, "granularity": "M5"})
        client.request(r)
        candles = r.response["candles"]
        prices = pd.Series([float(c['mid']['c']) for c in candles])
        
        # 1. EMA Bodyguard
        ema = prices.ewm(span=EMA_FILTER_PERIOD).mean().iloc[-1]
        
        # 2. RSI (14)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # 3. ATR (14)
        trs = []
        for i in range(len(candles)-14, len(candles)):
            h, l = float(candles[i]["mid"]["h"]), float(candles[i]["mid"]["l"])
            pc = float(candles[i-1]["mid"]["c"])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = float(pd.Series(trs).mean())
        
        return ema, rsi, atr
    except Exception as e:
        logging.error(f"Indicator Error {symbol}: {e}")
        return None, None, None

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

def log_to_journal(symbol, side, price, atr, mode, units):
    file_exists = os.path.isfile(JOURNAL_FILE)
    try:
        with open(JOURNAL_FILE,'a',newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["Timestamp","Symbol","Side","Price","ATR","Mode","Units"])
            writer.writerow([datetime.now(timezone.utc),symbol,side,price,round(atr,5),mode,units])
    except: pass

# ==================== NEWS / HUNTER LOGIC ====================

def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=15)
        tree = ET.fromstring(r.content)
        events = []
        for event in tree.findall('event'):
            if event.find('impact').text == 'High':
                curr = event.find('country').text
                dt_str = f"{event.find('date').text} {event.find('time').text}"
                dt = datetime.strptime(dt_str,"%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                events.append((curr, dt))
        HIGH_IMPACT_EVENTS = events
        logging.info(f"News Scraped: {len(HIGH_IMPACT_EVENTS)} high-impact events loaded.")
    except: logging.error("News Scrape Failed")

def get_hunter_signal(symbol):
    now = datetime.now(timezone.utc)
    base, quote = symbol.split("_")
    for curr, dt in HIGH_IMPACT_EVENTS:
        if curr in (base, quote):
            diff = (dt - now).total_seconds()
            if 0 < diff <= HUNTER_WINDOW_SECONDS:
                r = instruments.InstrumentsCandles(symbol, {"count": 2, "granularity": "M1"})
                client.request(r); c = r.response['candles']
                if float(c[-1]['mid']['c']) > float(c[-1]['mid']['o']): return "BUY"
                if float(c[-1]['mid']['c']) < float(c[-1]['mid']['o']): return "SELL"
    return None

def get_open_positions_count():
    try:
        r = positions.OpenPositions(OANDA_ACCOUNT_ID); client.request(r)
        return len(r.response.get("positions", []))
    except: return 99

# ==================== EXECUTION ====================

def execute_trade(symbol, side, price, atr, mode_label, nav, drawdown_pct):
    global DAILY_TRADE_COUNT, LAST_TRADE_TIME
    # Cooldown Check
    if symbol in LAST_TRADE_TIME:
        elapsed = (datetime.now(timezone.utc) - LAST_TRADE_TIME[symbol]).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES: return

    try:
        risk_mod = max(0.25, 1 - (drawdown_pct / DAILY_LOSS_LIMIT))
        units = int((nav * (BASE_RISK * risk_mod)) / (atr * 3))
        if units < 100: return # Micro-lot safety
        
        if side == "SELL": units = -units
        prec = 3 if "JPY" in symbol else 5
        
        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(round(price - atr*3 if side=="BUY" else price + atr*3, prec))).data,
            takeProfitOnFill=TakeProfitDetails(price=str(round(price + (atr*4 if mode_label=="HUNTER" else atr*2.5) if side=="BUY" else price - (atr*4 if mode_label=="HUNTER" else atr*2.5), prec))).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * 1.2, prec))).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        
        log_to_journal(symbol, side, price, atr, mode_label, units)
        LAST_TRADE_TIME[symbol] = datetime.now(timezone.utc)
        DAILY_TRADE_COUNT += 1
        send_telegram(f"🚀 {mode_label} {side}: {symbol} @ {price}")
    except Exception as e: logging.error(f"Trade Error: {e}")

# ==================== ENGINE CYCLE ====================

def run_apex_cycle(symbol):
    global ENGINE_HALTED, SESSION_START_NAV

    if ENGINE_HALTED or SESSION_START_NAV is None: return

    with ENGINE_LOCK:
        try:
            # 1. Health & NAV Check
            ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
            nav = float(ra.response["account"]["NAV"])
            drawdown = max(0, (SESSION_START_NAV - nav) / SESSION_START_NAV)
            
            if drawdown >= DAILY_LOSS_LIMIT:
                ENGINE_HALTED = True
                send_telegram("🛑 KILL-SWITCH: Daily Loss Limit Reached.")
                return

            if get_open_positions_count() >= MAX_OPEN_TRADES: return

            # 2. Indicators Fetch (Bodyguard logic)
            ema, rsi, atr = compute_indicators(symbol)
            if not ema or not atr: return

            r = instruments.InstrumentsCandles(symbol, {"count": 5, "granularity": "M5"})
            client.request(r); data = [float(c['mid']['c']) for c in r.response['candles']]
            price = data[-1]
            trend = "UP" if price > ema else "DOWN"

            # 3. GEAR 1: HUNTER (News Window)
            h_side = get_hunter_signal(symbol)
            if h_side and h_side == ("BUY" if trend == "UP" else "SELL"):
                execute_trade(symbol, h_side, price, atr, "HUNTER", nav, drawdown)
                return

            # 4. GEAR 2: SPIKE (Momentum with Bodyguard)
            move = data[-1] - data[-2]
            if abs(move) > (atr * SPIKE_SENSITIVITY):
                side = "BUY" if move > 0 else "SELL"
                if side == ("BUY" if trend == "UP" else "SELL"): # Only follow the 200 EMA
                    execute_trade(symbol, side, price, atr, "SPIKE", nav, drawdown)
                    return

            # 5. GEAR 3: FLOW (The Survival Mode - Catching the trend)
            # If market is trending up and RSI is strong but not overbought
            if trend == "UP" and 55 < rsi < 75:
                execute_trade(symbol, "BUY", price, atr, "FLOW", nav, drawdown)
            elif trend == "DOWN" and 25 < rsi < 45:
                execute_trade(symbol, "SELL", price, atr, "FLOW", nav, drawdown)

        except Exception as e: logging.error(f"Cycle Error {symbol}: {e}")

# ==================== SERVER & SCHEDULER ====================

scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scrape_high_impact_news, 'interval', minutes=30)
scheduler.add_job(reset_daily_metrics, 'cron', hour=0, minute=0)
scheduler.start()

@app.route('/')
def home():
    return jsonify({"status": "APEX-SURVIVOR ONLINE", "trades_today": DAILY_TRADE_COUNT})

@app.route('/run')
def trigger():
    threading.Thread(target=lambda: [run_apex_cycle(s) for s in SYMBOLS]).start()
    return "Scan Initiated...", 200

if __name__ == "__main__":
    reset_daily_metrics()
    scrape_high_impact_news()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
