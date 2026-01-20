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
HIGH_IMPACT_EVENTS = []

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-ELITE] | %(message)s")

# ==================== UTILITIES ====================

def compute_atr_true_range(symbol, period=14):
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": period+1, "granularity": "M15"})
        client.request(r)
        candles = r.response["candles"]
        trs = []
        for i in range(1, len(candles)):
            h = float(candles[i]["mid"]["h"])
            l = float(candles[i]["mid"]["l"])
            pc = float(candles[i-1]["mid"]["c"])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
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
    try:
        risk_mod = max(0.25, 1 - (drawdown_pct / DAILY_LOSS_LIMIT))
        units = int((nav * (BASE_RISK * risk_mod)) / (atr * 3))
        if units < 1000: return
        units = min(units, 100000)
        if side == "SELL": units = -units
        
        prec = 3 if "JPY" in symbol else 5
        trail_dist = 1.1 if mode_label in ["SPIKE", "HUNTER"] else 1.5
        
        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(round(price - atr*3 if side=="BUY" else price + atr*3, prec))).data,
            takeProfitOnFill=TakeProfitDetails(price=str(round(price + (atr*4 if mode_label=="HUNTER" else atr*2.5) if side=="BUY" else price - (atr*4 if mode_label=="HUNTER" else atr*2.5), prec))).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * trail_dist, prec))).data
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


    if SESSION_START_NAV is None:
        logging.info("NAV anchor missing. Attempting emergency initialization...")
        reset_daily_metrics()
        if SESSION_START_NAV is None:
            logging.error("OANDA Connection Failed. Cannot initialize NAV.")
            return
    
    # Pre-Lock Check (Visual confirmation in logs)
    logging.info(f"--- 🔎 Scanning {symbol} ---")
    
    if ENGINE_HALTED:
        logging.warning(f"Aborted {symbol}: Engine is currently HALTED by Global Kill-Switch.")
        return

    if SESSION_START_NAV is None:
        logging.error(f"Aborted {symbol}: Session Start NAV not set. Check connection.")
        return

    with ENGINE_LOCK:
        try:
            # 1. ACCOUNT HEALTH CHECK
            ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
            nav = float(ra.response["account"]["NAV"])
            drawdown = max(0, (SESSION_START_NAV - nav) / SESSION_START_NAV)
            
            if drawdown >= DAILY_LOSS_LIMIT:
                ENGINE_HALTED = True
                send_telegram(f"🛑 KILL-SWITCH ACTIVATED. Drawdown at {round(drawdown*100, 2)}%. Engine Halted.")
                return

            # 2. CAPACITY CHECK
            open_count = get_open_positions_count()
            if open_count >= MAX_OPEN_TRADES:
                logging.info(f"[{symbol}] Max open trades reached ({open_count}). Skipping.")
                return
            
            # 3. TECHNICAL DATA FETCH
            atr = compute_atr_true_range(symbol)
            r = instruments.InstrumentsCandles(symbol, {"count": 20, "granularity": "M15"})
            client.request(r)
            data = [float(c['mid']['c']) for c in r.response['candles']]
            price = data[-1]

            if not atr:
                logging.warning(f"[{symbol}] Could not calculate ATR. Skipping.")
                return

            # 4. GEAR 1: HUNTER (The Friend's Strategy)
            h_side = get_hunter_signal(symbol)
            if h_side:
                logging.info(f"🎯 HUNTER SIGNAL DETECTED: {symbol} {h_side}!")
                execute_trade(symbol, h_side, price, atr, "HUNTER", nav, drawdown)
                return
            else:
                logging.info(f"[{symbol}] News Hunter: No pending high-impact events in window.")

            # 5. GEAR 2: SPIKE (Reactionary Momentum)
            move = abs(data[-1] - data[-2])
            threshold = atr * SPIKE_SENSITIVITY
            if move > threshold:
                side = "BUY" if data[-1] > data[-2] else "SELL"
                logging.info(f"⚡ SPIKE DETECTED: {symbol} {side} (Move: {round(move,5)} > Threshold: {round(threshold,5)})")
                execute_trade(symbol, side, price, atr, "SPIKE", nav, drawdown)
                return
            else:
                logging.info(f"[{symbol}] Volatility: Move {round(move,5)} < Threshold {round(threshold,5)}")

            # 6. GEAR 3: NORMAL (Optional RSI/EMA)
            # logging.info(f"[{symbol}] Quiet market. No valid technical signals found.")

        except Exception as e:
            logging.error(f"Cycle Error {symbol}: {e}")







# ==================== SERVER & SCHEDULER ====================

scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scrape_high_impact_news, 'interval', minutes=30)
scheduler.add_job(reset_daily_metrics, 'cron', hour=0, minute=0)
scheduler.start()

@app.route('/status')
def status():
    return jsonify({
        "engine": "APEX-ELITE", 
        "halted": ENGINE_HALTED, 
        "trades": DAILY_TRADE_COUNT, 
        "nav": SESSION_START_NAV,
        "news_events": len(HIGH_IMPACT_EVENTS)
    }), 200

@app.route('/run')
def trigger():
    threading.Thread(target=lambda: [run_apex_cycle(s) for s in SYMBOLS]).start()
    return "Scanning Symbols...", 200



if __name__ == "__main__":
    # 1. Force immediate data load
    logging.info("Initializing APEX-ELITE State...")
    reset_daily_metrics() # This sets the SESSION_START_NAV
    scrape_high_impact_news()
    
    # 2. Start the web server
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

