import os, logging, threading, requests, time, numpy as np, pandas as pd, ta
from datetime import datetime, timezone, timedelta
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
DAILY_LOSS_LIMIT = 0.02
PORTFOLIO_LOCK = threading.Lock()

# Global tracking
SESSION_START_NAV = None
DAILY_TRADE_COUNT = 0
HIGH_IMPACT_EVENTS = []

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V10] | %(message)s")

# ==================== TELEGRAM & PERFORMANCE ====================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def send_daily_summary():
    global DAILY_TRADE_COUNT, SESSION_START_NAV
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        curr_nav = float(r.response["account"]["NAV"])
        pnl = curr_nav - SESSION_START_NAV
        pnl_pct = (pnl / SESSION_START_NAV) * 100
        
        summary = (
            f"📊 --- DAILY TRADING REPORT ---\n"
            f"💰 Ending NAV: ${curr_nav:,.2f}\n"
            f"📈 Net P/L: ${pnl:,.2f} ({pnl_pct:.2f}%)\n"
            f"🚀 Trades Today: {DAILY_TRADE_COUNT}\n"
            f"📅 Session: {datetime.now().strftime('%Y-%m-%d')}"
        )
        send_telegram(summary)
        DAILY_TRADE_COUNT = 0
        SESSION_START_NAV = curr_nav # Reset baseline for next session
    except Exception as e:
        logging.error(f"Summary Error: {e}")

# ==================== NEWS ENGINE ====================
def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        xml = response.text
        events = []
        for item in xml.split("<event>")[1:]:
            if "<impact>High</impact>" in item:
                currency = item.split("<currency>")[1].split("</currency>")[0]
                title = item.split("<title>")[1].split("</title>")[0]
                date = item.split("<date>")[1].split("</date>")[0]
                time_str = item.split("<time>")[1].split("</time>")[0]
                dt_str = f"{date} {time_str}"
                event_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                events.append({"currency": currency, "title": title, "time": event_dt})
        HIGH_IMPACT_EVENTS = events
        logging.info(f"Loaded {len(events)} High Impact events.")
    except Exception as e:
        logging.error(f"Calendar Scrape Fail: {e}")

def is_news_window(symbol):
    base, quote = symbol.split("_")
    now = datetime.now(timezone.utc)
    for ev in HIGH_IMPACT_EVENTS:
        if ev["currency"] in (base, quote):
            delta = (ev["time"] - now).total_seconds()
            if -60 <= delta <= 300: # 1 min before to 5 mins after
                return True, ev
    return False, None

# ==================== TRADING LOGIC ====================
def order_flow_bias(df):
    delta = df['close'] - df['open']
    bull = delta[delta > 0].sum()
    bear = abs(delta[delta < 0].sum())
    return "BUY" if bull > bear * 1.4 else "SELL" if bear > bull * 1.4 else "NEUTRAL"

def execute_unified_trade(symbol, side, price, atr, nav, trade_type):
    global DAILY_TRADE_COUNT
    try:
        pip = 0.01 if "JPY" in symbol else 0.0001
        # Position sizing based on 3*ATR stop distance
        units = max(1, int((nav * BASE_RISK) / (atr * 3 * pip)))
        if side == "SELL": units *= -1

        prec = 3 if "JPY" in symbol else 5
        sl_price = round(price - atr*3 if side=="BUY" else price + atr*3, prec)
        tp_price = round(price + atr*2.5 if side=="BUY" else price - atr*2.5, prec)
        
        # Trailing Stop distance: 1.5 * ATR
        tsl_dist = round(atr * 1.5, prec)

        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(sl_price)).data,
            takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(tsl_dist)).data
        )
        
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        DAILY_TRADE_COUNT += 1
        
        msg = (f"🚀 {trade_type}\n📦 {symbol} {side}\nEntry: {price}\n"
               f"🛡 SL: {sl_price} | 🎯 TP: {tp_price}\n🔄 Trailing: {tsl_dist} pips")
        send_telegram(msg)
    except Exception as e:
        logging.error(f"Trade Fail: {e}")

def run_apex_unified(symbol, index):
    with PORTFOLIO_LOCK:
        try:
            r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 50})
            client.request(r)
            df = pd.DataFrame([{"open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), 
                                "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])} for c in r.response["candles"]])
            
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
            rsi = ta.momentum.rsi(df['close'], 14).iloc[-1]
            flow = order_flow_bias(df.tail(10))
            
            acc = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(acc)
            nav = float(acc.response["account"]["NAV"])

            # 1. News Signal Priority
            news_active, ev = is_news_window(symbol)
            if news_active and flow != "NEUTRAL":
                execute_unified_trade(symbol, flow, df.iloc[-1]['close'], atr, nav, f"NEWS ({ev['title']})")
                return

            # 2. APEX Technical Signal
            r_h1 = instruments.InstrumentsCandles(symbol, {"granularity": "H1", "count": 20})
            client.request(r_h1)
            h1_closes = [float(c["mid"]["c"]) for c in r_h1.response["candles"]]
            trend = "UP" if h1_closes[-1] > np.mean(h1_closes) else "DOWN"

            if rsi < 30 and trend == "UP" and flow == "BUY":
                execute_unified_trade(symbol, "BUY", df.iloc[-1]['close'], atr, nav, "APEX-STRATEGY")
            elif rsi > 70 and trend == "DOWN" and flow == "SELL":
                execute_unified_trade(symbol, "SELL", df.iloc[-1]['close'], atr, nav, "APEX-STRATEGY")

        except Exception as e:
            logging.error(f"Logic Error for {symbol}: {e}")

# ==================== SCHEDULER & APP ====================
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(func=scrape_high_impact_news, trigger="interval", minutes=60)
scheduler.add_job(func=send_daily_summary, trigger="cron", hour=17, minute=0)
scheduler.start()

@app.route('/run')
def trigger():
    def run_cycle():
        for i, s in enumerate(SYMBOLS):
            run_apex_unified(s, i)
            time.sleep(15) 

    threading.Thread(target=run_cycle).start()
    return jsonify({"status": "Engine Running", "trailing_stop": "Enabled"}), 200

@app.route('/')
def home():
    return "APEX-V10 LIVE (Trailing Stop Active)", 200

if __name__ == "__main__":
    scrape_high_impact_news()
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
    SESSION_START_NAV = float(r.response["account"]["NAV"])
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
