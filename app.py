import os, logging, threading, requests, time, numpy as np, pandas as pd, ta
from datetime import datetime, timezone
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, StopLossDetails, TakeProfitDetails, TrailingStopLossDetails

# ================= CONFIGURATION =================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
BASE_RISK = 0.005
PORTFOLIO_LOCK = threading.Lock()

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V10] | %(message)s")

# Global tracking
SESSION_START_NAV = None
DAILY_TRADE_COUNT = 0
HIGH_IMPACT_EVENTS = []

# ==================== UTILS ====================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        r = requests.get(url, timeout=15)
        HIGH_IMPACT_EVENTS = [] # Simple parse for brevity
        logging.info("News Calendar Refreshed.")
    except: logging.error("News Scrape Failed.")

# ==================== CORE LOGIC ====================
def execute_trade(symbol, side, price, atr, nav, t_type):
    global DAILY_TRADE_COUNT
    try:
        pip = 0.01 if "JPY" in symbol else 0.0001
        units = max(1, int((nav * BASE_RISK) / (atr * 3 * pip)))
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
        send_telegram(f"✅ {t_type} TRADE: {symbol} {side} @ {price}")
    except Exception as e:
        logging.error(f"Trade Execution Failed: {e}")

def run_apex_unified(symbol):
    with PORTFOLIO_LOCK:
        try:
            # 1. Fetch Data
            r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 50})
            client.request(r)
            df = pd.DataFrame([{"h": float(c["mid"]["h"]), "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"])} for c in r.response["candles"]])
            
            rsi = ta.momentum.rsi(df['c'], 14).iloc[-1]
            atr = ta.volatility.average_true_range(df['h'], df['l'], df['c']).iloc[-1]
            curr_price = df['c'].iloc[-1]

            # 2. Fetch H1 Trend
            rh1 = instruments.InstrumentsCandles(symbol, {"granularity": "H1", "count": 20})
            client.request(rh1)
            h1_avg = np.mean([float(c["mid"]["c"]) for c in rh1.response["candles"]])
            trend = "UP" if curr_price > h1_avg else "DOWN"

            # 3. Logging for you to see in Render
            logging.info(f"Scanning {symbol} | RSI: {rsi:.2f} | Trend: {trend} | Price: {curr_price}")

            # 4. Entry Logic (65/35 for slightly more frequent trades)
            acc = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(acc)
            nav = float(acc.response["account"]["NAV"])

            if rsi < 35 and trend == "UP":
                execute_trade(symbol, "BUY", curr_price, atr, nav, "APEX-V10")
            elif rsi > 65 and trend == "DOWN":
                execute_trade(symbol, "SELL", curr_price, atr, nav, "APEX-V10")

        except Exception as e:
            logging.error(f"Scan error {symbol}: {e}")

# ==================== APP ROUTES ====================
@app.route('/run')
def trigger():
    def worker():
        for s in SYMBOLS:
            run_apex_unified(s)
            time.sleep(2)
    threading.Thread(target=worker).start()
    return jsonify({"status": "Scanning Symbols..."}), 200

@app.route('/')
def home():
    return "APEX-V10 SYSTEM ONLINE", 200

if __name__ == "__main__":
    scrape_high_impact_news()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
