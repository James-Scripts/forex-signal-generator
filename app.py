import os, logging, threading, requests, time, numpy as np, pandas as pd, ta
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

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
BASE_RISK = 0.005  # 0.5% per trade
DAILY_LOSS_LIMIT = 0.02
PORTFOLIO_LOCK = threading.Lock()

# Global tracking
SESSION_START_NAV = None
DAILY_TRADE_COUNT = 0
OPEN_POSITIONS = {}  # symbol -> side
HIGH_IMPACT_EVENTS = [] 

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V10] | %(message)s")

# ==================== UTILITIES ====================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def sync_open_positions():
    """Correctly syncs open positions using the positions endpoint."""
    global OPEN_POSITIONS
    try:
        r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
        client.request(r)
        pos_data = r.response.get('positions', [])
        new_positions = {}
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
        logging.info(f"News Shield Updated: {len(events)} events.")
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

def is_spread_safe(symbol, max_pips=2.5):
    """Checks the real-time Bid/Ask spread."""
    try:
        params = {"count": 1, "granularity": "S5", "price": "BA"}
        r = instruments.InstrumentsCandles(symbol, params=params)
        client.request(r)
        candle = r.response['candles'][0]
        spread = float(candle['ask']['c']) - float(candle['bid']['c'])
        pip = 0.01 if "JPY" in symbol else 0.0001
        return (spread / pip) <= max_pips
    except: return False

def is_correlation_safe(symbol):
    corrs = {"EUR_USD": "GBP_USD", "GBP_USD": "EUR_USD", "USD_JPY": "AUD_USD", "AUD_USD": "USD_JPY"}
    correlated_pair = corrs.get(symbol)
    return correlated_pair not in OPEN_POSITIONS

# ==================== EXECUTION ====================
def execute_trade(symbol, side, price, atr, nav, t_type):
    global DAILY_TRADE_COUNT
    try:
        pip = 0.01 if "JPY" in symbol else 0.0001
        # Adaptive Risk Formula
        drawdown_pct = (SESSION_START_NAV - nav) / SESSION_START_NAV
        risk_mod = max(0.5, 1 - (drawdown_pct / DAILY_LOSS_LIMIT))
        
        units = max(1, int((nav * (BASE_RISK * risk_mod)) / (atr * 3 * pip)))
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
        DAILY_TRADE_COUNT += 1
        send_telegram(f"✅ {t_type}\n{symbol} {side} @ {price}\nSL: {sl} | TP: {tp}")
    except Exception as e:
        logging.error(f"Trade Fail: {e}")

# ==================== ENGINE ====================
def run_apex_cycle(symbol):
    with PORTFOLIO_LOCK:
        try:
            # 1. Validation Checks
            if symbol in OPEN_POSITIONS: return
            if not is_news_safe(symbol) or not is_correlation_safe(symbol) or not is_spread_safe(symbol):
                return

            # 2. Get Market Data (M15)
            r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 50})
            client.request(r)
            df = pd.DataFrame([{"o": float(c["mid"]["o"]), "h": float(c["mid"]["h"]), 
                                "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"])} for c in r.response["candles"]])
            
            rsi = ta.momentum.rsi(df['c'], 14).iloc[-1]
            atr = ta.volatility.average_true_range(df['h'], df['l'], df['c']).iloc[-1]
            price = df['c'].iloc[-1]
            
            # 3. H1 Trend Confirmation
            rh = instruments.InstrumentsCandles(symbol, {"granularity": "H1", "count": 20})
            client.request(rh)
            h1_avg = np.mean([float(c["mid"]["c"]) for c in rh.response["candles"]])
            trend = "UP" if price > h1_avg else "DOWN"

            # 4. NAV Check (Fixed request bug)
            ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
            nav = float(ra.response["account"]["NAV"])

            # 5. Signals
            if rsi < 35 and trend == "UP":
                execute_trade(symbol, "BUY", price, atr, nav, "APEX-STRATEGY")
            elif rsi > 65 and trend == "DOWN":
                execute_trade(symbol, "SELL", price, atr, nav, "APEX-STRATEGY")

        except Exception as e:
            logging.error(f"Cycle error for {symbol}: {e}")

# ==================== FLASK & START ====================
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scrape_high_impact_news, 'interval', minutes=60)
scheduler.start()

@app.route('/run')
def trigger():
    sync_open_positions()
    threading.Thread(target=lambda: [run_apex_cycle(s) for s in SYMBOLS]).start()
    return jsonify({"status": "APEX-V10 Scanning", "positions_active": list(OPEN_POSITIONS.keys())}), 200

@app.route('/')
def home():
    return f"APEX-V10 SYSTEM ONLINE. Active Pairs: {len(OPEN_POSITIONS)}", 200

if __name__ == "__main__":
    scrape_high_impact_news()
    sync_open_positions()
    r_init = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r_init)
    SESSION_START_NAV = float(r_init.response["account"]["NAV"])
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
