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
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CHF", "EUR_CAD", "GBP_JPY", "GBP_CHF",
    "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY", "AUD_CAD", "AUD_NZD"
]

# Deployment Risk Settings (Safe Start)
BASE_RISK = 0.0025         # Start at 0.25%
MAX_DAILY_TRADES = 5       # Strict trade cap
COOLDOWN_MINUTES = 45      # Patient entries
DAILY_LOSS_LIMIT = 0.02
MAX_UNIT_CAP = 100000     
MIN_UNITS = 1000           
CORRELATION_THRESHOLD = 0.7 
MAX_CURRENCY_EXPOSURE = 3 
MAX_SPREAD_PIPS = 2.5      

COUNTRY_TO_CCY = {
    "USA": "USD", "United States": "USD", "EUR": "EUR", "Germany": "EUR", 
    "United Kingdom": "GBP", "Japan": "JPY", "Switzerland": "CHF", 
    "Australia": "AUD", "Canada": "CAD", "New Zealand": "NZD"
}

JOURNAL_FILE = "trade_journal.csv"
PORTFOLIO_LOCK = threading.Lock()

# Global tracking
SESSION_START_NAV = None
LAST_TRADE_TIME = {} 
OPEN_POSITIONS = {}  
HIGH_IMPACT_EVENTS = [] 
DAILY_TRADE_COUNT = 0 
PAIR_CACHE = {}       

# CHANGE "practice" to "live" when ready for real funds
client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V16-FINAL] | %(message)s")

# ==================== UTILITIES ====================

def compute_atr(data, period=14):
    if len(data) < period + 1: return None
    trs = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
    atr = pd.Series(trs).rolling(period).mean().iloc[-1]
    return float(atr) if atr and atr > 0 else None

def reset_daily_metrics():
    global SESSION_START_NAV, DAILY_TRADE_COUNT, LAST_TRADE_TIME
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        SESSION_START_NAV = float(r.response["account"]["NAV"])
        DAILY_TRADE_COUNT = 0
        LAST_TRADE_TIME = {}
        send_telegram(f"🌅 SESSION START: NAV @ {SESSION_START_NAV}")
    except Exception as e: logging.error(f"Reset Error: {e}")

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
                writer.writerow(["Timestamp", "Symbol", "Side", "Price", "ATR", "RSI", "EMA", "Units"])
            writer.writerow([datetime.now(timezone.utc), symbol, side, price, round(atr,5), round(rsi,2), round(ema,5), units])
    except: pass

# ==================== FILTERS ====================

def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=15)
        tree = ET.fromstring(r.content)
        events = []
        for event in tree.findall('event'):
            if event.find('impact').text == 'High':
                curr = COUNTRY_TO_CCY.get(event.find('country').text, event.find('country').text)
                dt_str = f"{event.find('date').text} {event.find('time').text}"
                try:
                    dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                    events.append((curr, dt))
                except: continue
        HIGH_IMPACT_EVENTS = events
    except: logging.error("News Fetch Failed.")

def is_near_news(symbol, window_min=45):
    now = datetime.now(timezone.utc)
    base, quote = symbol.split("_")
    for curr, dt in HIGH_IMPACT_EVENTS:
        if curr in (base, quote) and abs((dt - now).total_seconds()) <= window_min * 60:
            return True
    return False

def is_market_active():
    now = datetime.now(timezone.utc)
    return now.weekday() < 5 and 6 <= now.hour <= 20

def is_spread_ok(symbol):
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": 1, "granularity": "M1", "price": "BA"})
        client.request(r)
        c = r.response["candles"][-1]
        spread_pips = (float(c["ask"]["c"]) - float(c["bid"]["c"])) / (0.01 if "JPY" in symbol else 0.0001)
        return spread_pips <= MAX_SPREAD_PIPS
    except: return False

def is_portfolio_safe(symbol):
    with PORTFOLIO_LOCK:
        if symbol in OPEN_POSITIONS: return False
        open_ccys = [c for p in OPEN_POSITIONS.keys() for c in p.split('_')]
        for c in symbol.split('_'):
            if open_ccys.count(c) >= MAX_CURRENCY_EXPOSURE: return False
        
        new_data = get_historical_closes(symbol)
        if not new_data: return False
        for op in OPEN_POSITIONS.keys():
            op_data = get_historical_closes(op)
            if not op_data: continue
            min_l = min(len(new_data), len(op_data))
            corr = np.corrcoef(new_data[-min_l:], op_data[-min_l:])[0, 1]
            if not np.isnan(corr) and corr > CORRELATION_THRESHOLD: return False
    return True

def get_historical_closes(symbol):
    cache_key = f"{symbol}_M15"
    if cache_key in PAIR_CACHE: return PAIR_CACHE[cache_key]
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": 100, "granularity": "M15"})
        client.request(r)
        data = [float(c['mid']['c']) for c in r.response['candles']]
        PAIR_CACHE[cache_key] = data
        return data
    except: return []

# ==================== EXECUTION ====================

def execute_trade(symbol, side, price, atr, rsi, ema, nav, drawdown_pct):
    global DAILY_TRADE_COUNT, LAST_TRADE_TIME
    try:
        risk_mod = max(0.25, 1 - (drawdown_pct / DAILY_LOSS_LIMIT))
        
        # [FIX 1] Corrected Unit Logic
        raw_units = int((nav * (BASE_RISK * risk_mod)) / (atr * 3))
        if raw_units < MIN_UNITS: return
        
        units = min(raw_units, MAX_UNIT_CAP)
        if side == "SELL":
            units = -units

        prec = 3 if "JPY" in symbol else 5
        sl = round(price - atr*3 if side=="BUY" else price + atr*3, prec)
        tp = round(price + atr*2.5 if side=="BUY" else price - atr*2.5, prec)
        
        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(sl)).data,
            takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr*1.5, prec))).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        
        log_to_journal(symbol, side, price, atr, rsi, ema, units)
        LAST_TRADE_TIME[symbol] = datetime.now(timezone.utc)
        DAILY_TRADE_COUNT += 1
        send_telegram(f"✅ {symbol} {side} Executed.")
    except Exception as e: logging.error(f"Trade Error: {e}")

# ==================== ENGINE ====================

def run_apex_cycle(symbol):
    global SESSION_START_NAV
    try:
        # [FIX 2] Missing NAV Guard
        if SESSION_START_NAV is None:
            reset_daily_metrics()
            return

        if DAILY_TRADE_COUNT >= MAX_DAILY_TRADES or not is_market_active(): return
        
        last_t = LAST_TRADE_TIME.get(symbol)
        if last_t and (datetime.now(timezone.utc) - last_t).total_seconds() < COOLDOWN_MINUTES * 60: return

        ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
        nav = float(ra.response["account"]["NAV"])
        drawdown = max(0, (SESSION_START_NAV - nav) / SESSION_START_NAV)
        if drawdown >= DAILY_LOSS_LIMIT: return

        if is_near_news(symbol) or not is_spread_ok(symbol): return

        data = get_historical_closes(symbol)
        if len(data) < 50: return
        df = pd.DataFrame({'c': data})
        rsi = ta.momentum.rsi(df['c'], 14).iloc[-1]
        ema20 = df['c'].ewm(span=20).mean().iloc[-1]
        ema50 = df['c'].ewm(span=50).mean().iloc[-1]
        price = data[-1]
        atr = compute_atr(data)
        if not atr or abs(ema20 - ema50) / price < 0.0004: return
        
        trend = "UP" if price > ema20 else "DOWN"

        if rsi < 30 and trend == "UP":
            if is_portfolio_safe(symbol):
                execute_trade(symbol, "BUY", price, atr, rsi, ema20, nav, drawdown)
        elif rsi > 70 and trend == "DOWN":
            if is_portfolio_safe(symbol):
                execute_trade(symbol, "SELL", price, atr, rsi, ema20, nav, drawdown)

    except Exception as e: logging.error(f"Cycle Error: {e}")

# ==================== DEPLOYMENT ====================

scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scrape_high_impact_news, 'interval', minutes=30)
scheduler.add_job(reset_daily_metrics, 'cron', hour=0, minute=0)
scheduler.start()

@app.route('/run')
def trigger():
    global PAIR_CACHE
    PAIR_CACHE = {} 
    # Sync positions
    try:
        r = positions.OpenPositions(OANDA_ACCOUNT_ID); client.request(r)
        global OPEN_POSITIONS
        OPEN_POSITIONS = {p['instrument']: "ACTIVE" for p in r.response.get('positions', []) if int(p['long']['units']) != 0 or int(p['short']['units']) != 0}
    except: pass
    
    threading.Thread(target=lambda: [run_apex_cycle(s) for s in SYMBOLS]).start()
    return jsonify({"status": "APEX-V16-FINAL Scanning"}), 200

@app.route('/')
def health(): return "APEX ENGINE ACTIVE", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
