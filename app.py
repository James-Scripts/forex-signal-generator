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

BASE_RISK = 0.005 
DAILY_LOSS_LIMIT = 0.02
CORRELATION_THRESHOLD = 0.7  
MAX_CURRENCY_EXPOSURE = 3 
COOLDOWN_MINUTES = 30 
JOURNAL_FILE = "trade_journal.csv"
PORTFOLIO_LOCK = threading.Lock()

# Global tracking
SESSION_START_NAV = None
LAST_TRADE_TIME = {} 
OPEN_POSITIONS = {}  
HIGH_IMPACT_EVENTS = [] 

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V12] | %(message)s")

# ==================== UTILITIES ====================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def log_to_journal(symbol, side, price, atr, rsi, ema, t_type, units):
    """Saves trade data to CSV for performance auditing."""
    file_exists = os.path.isfile(JOURNAL_FILE)
    try:
        with open(JOURNAL_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Symbol", "Side", "Price", "ATR", "RSI", "EMA_Trend", "Strategy", "Units"])
            
            writer.writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side, price, round(atr, 5), round(rsi, 2), round(ema, 5), t_type, units
            ])
        logging.info(f"📝 Journal Updated: {symbol} {side}")
    except Exception as e:
        logging.error(f"Journaling Error: {e}")

def sync_open_positions():
    global OPEN_POSITIONS
    try:
        r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
        client.request(r)
        pos_data = r.response.get('positions', [])
        new_positions = {}
        with PORTFOLIO_LOCK:
            for p in pos_data:
                symbol = p['instrument']
                if int(p['long']['units']) != 0: new_positions[symbol] = "BUY"
                elif int(p['short']['units']) != 0: new_positions[symbol] = "SELL"
            OPEN_POSITIONS = new_positions
    except Exception as e:
        logging.error(f"Sync Open Positions Failed: {e}")

def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        r = requests.get(url, timeout=15)
        tree = ET.fromstring(r.content)
        events = []
        for event in tree.findall('event'):
            if event.find('impact').text == 'High':
                curr = event.find('country').text
                dt_str = f"{event.find('date').text} {event.find('time').text}"
                try:
                    dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                    events.append((curr, dt))
                except: continue
        HIGH_IMPACT_EVENTS = events
    except Exception as e:
        logging.error(f"News Scrape Failed: {e}")

# ==================== FILTERS ====================
def is_news_safe(symbol):
    now = datetime.now(timezone.utc)
    currencies = symbol.split('_')
    for curr, ts in HIGH_IMPACT_EVENTS:
        if curr in currencies and abs((ts - now).total_seconds()) < 3600:
            return False
    return True

def is_spread_safe(symbol, max_pips=3.0):
    try:
        params = {"count": 1, "granularity": "M1", "price": "BA"}
        r = instruments.InstrumentsCandles(symbol, params=params)
        client.request(r)
        candle = r.response['candles'][0]
        spread = float(candle['ask']['c']) - float(candle['bid']['c'])
        pip = 0.01 if "JPY" in symbol else 0.0001
        return (spread / pip) <= max_pips
    except: return False

def get_historical_closes(symbol, count=100, gran="M15"):
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": count, "granularity": gran})
        client.request(r)
        return [float(c['mid']['c']) for c in r.response['candles']]
    except: return []

def is_portfolio_safe(symbol):
    with PORTFOLIO_LOCK:
        if symbol in OPEN_POSITIONS: return False
        new_currencies = symbol.split('_')
        open_currencies = []
        for op in OPEN_POSITIONS.keys():
            open_currencies.extend(op.split('_'))
        
        for curr in new_currencies:
            if open_currencies.count(curr) >= MAX_CURRENCY_EXPOSURE:
                logging.info(f"🚫 Exposure Block: {curr}")
                return False

        new_pair_data = get_historical_closes(symbol)
        if len(new_pair_data) < 50: return False

        for open_symbol in OPEN_POSITIONS.keys():
            open_pair_data = get_historical_closes(open_symbol)
            if len(open_pair_data) < 50: continue
            min_len = min(len(new_pair_data), len(open_pair_data))
            corr = np.corrcoef(new_pair_data[-min_len:], open_pair_data[-min_len:])[0, 1]
            if abs(corr) > CORRELATION_THRESHOLD:
                logging.info(f"🚫 Correlation Block: {symbol} vs {open_symbol}")
                return False
    return True

# ==================== EXECUTION ====================
def execute_trade(symbol, side, price, atr, rsi, ema, nav, t_type):
    global LAST_TRADE_TIME
    try:
        units = max(1, int((nav * BASE_RISK) / (atr * 3)))
        if side == "SELL": units *= -1

        prec = 3 if "JPY" in symbol else 5
        sl = round(price - atr*3 if side=="BUY" else price + atr*3, prec)
        tp = round(price + atr*2.5 if side=="BUY" else price - atr*2.5, prec)
        tsl = round(atr * 1.5, prec)

        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(sl)).data,
            takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(tsl)).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        
        # LOG TO JOURNAL
        log_to_journal(symbol, side, price, atr, rsi, ema, t_type, units)
        
        LAST_TRADE_TIME[symbol] = datetime.now(timezone.utc)
        send_telegram(f"✅ {t_type}\n{symbol} {side} @ {price}\nSL: {sl} | TP: {tp}")
    except Exception as e:
        logging.error(f"Trade Fail: {e}")

# ==================== ENGINE ====================
def run_apex_cycle(symbol):
    try:
        ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
        nav = float(ra.response["account"]["NAV"])
        drawdown_pct = (SESSION_START_NAV - nav) / SESSION_START_NAV
        if drawdown_pct >= DAILY_LOSS_LIMIT:
            return

        last_t = LAST_TRADE_TIME.get(symbol)
        if last_t and (datetime.now(timezone.utc) - last_t).total_seconds() < COOLDOWN_MINUTES * 60:
            return

        if not is_news_safe(symbol) or not is_portfolio_safe(symbol) or not is_spread_safe(symbol):
            return

        r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 50})
        client.request(r)
        df = pd.DataFrame([{"h": float(c["mid"]["h"]), "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"])} for c in r.response["candles"]])
        
        rsi = ta.momentum.rsi(df['c'], 14).iloc[-1]
        atr = ta.volatility.average_true_range(df['h'], df['l'], df['c']).iloc[-1]
        price = df['c'].iloc[-1]
        
        rh = instruments.InstrumentsCandles(symbol, {"granularity": "H1", "count": 50})
        client.request(rh)
        h1_closes = pd.Series([float(c["mid"]["c"]) for c in rh.response["candles"]])
        ema20 = h1_closes.ewm(span=20).mean().iloc[-1]
        trend = "UP" if price > ema20 else "DOWN"

        if rsi < 35 and trend == "UP":
            execute_trade(symbol, "BUY", price, atr, rsi, ema20, nav, "APEX-STRATEGY")
        elif rsi > 65 and trend == "DOWN":
            execute_trade(symbol, "SELL", price, atr, rsi, ema20, nav, "APEX-STRATEGY")

    except Exception as e:
        logging.error(f"Cycle error for {symbol}: {e}")

# ==================== FLASK & START ====================
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scrape_high_impact_news, 'interval', minutes=10)
scheduler.start()

@app.route('/run')
def trigger():
    sync_open_positions()
    threading.Thread(target=lambda: [run_apex_cycle(s) for s in SYMBOLS]).start()
    return jsonify({"status": "Scanning", "active": list(OPEN_POSITIONS.keys())}), 200

@app.route('/journal')
def view_journal():
    """Allows you to view the last 20 entries of your journal via browser."""
    if not os.path.exists(JOURNAL_FILE): return "No journal yet."
    with open(JOURNAL_FILE, 'r') as f:
        lines = f.readlines()
    return "<pre>" + "".join(lines[-21:]) + "</pre>"

@app.route('/')
def home():
    return f"APEX-V12 ONLINE. Monitoring {len(SYMBOLS)} pairs.", 200

if __name__ == "__main__":
    scrape_high_impact_news()
    sync_open_positions()
    r_init = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r_init)
    SESSION_START_NAV = float(r_init.response["account"]["NAV"])
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
