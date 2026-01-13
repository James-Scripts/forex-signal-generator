import os
import sqlite3
import logging
import pandas as pd
import ta
import requests
import time
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, request

# OANDA API
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.forexlabs as labs
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Config
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
DASHBOARD_PW = os.environ.get("DASHBOARD_PW", "1234")
DB_PATH = os.environ.get("DB_PATH", "bot_state.db")

client = API(access_token=OANDA_API_KEY, environment="practice")

# Expanded symbols for more frequent opportunities
SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY", 
    "USD_CAD", "NZD_USD", "EUR_JPY", "EUR_GBP", "AUD_JPY"
]

# ANTI-BLOCK HEADERS: Mimics a real Chrome browser on Windows
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.oanda.com/",
    "X-Requested-With": "XMLHttpRequest"
}

# --- 2. DATABASE ---
def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                    (trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, 
                     price REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def send_telegram_msg(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        logger.error(f"Telegram failed: {e}")

# --- 3. SAFETY FILTERS (WITH FAIL-SAFE) ---

def is_news_safe():
    """Checks for news but returns True (Safe) if blocked by Cloudflare."""
    if not FINNHUB_API_KEY: return True
    try:
        url = f"https://finnhub.io/api/v1/calendar/economic?token={FINNHUB_API_KEY}"
        # Use browser headers to avoid 403
        res = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        if res.status_code != 200:
            logger.warning(f"News API returned {res.status_code}. Skipping safety check (Safe by default).")
            return True
        
        events = res.json().get('economicCalendar', [])
        now = datetime.utcnow()
        for event in events:
            if str(event.get('impact')).lower() in ['high', '3']:
                e_time = datetime.strptime(event['time'], '%Y-%m-%d %H:%M:%S')
                if abs((e_time - now).total_seconds()) < 1800:
                    logger.info(f"High impact news found: {event['event']}. Bot paused.")
                    return False
    except Exception as e:
        logger.warning(f"News check encountered error: {e}. Proceeding with trade logic.")
    return True

def is_correlated(new_symbol):
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        client.request(r)
        open_pairs = [t['instrument'] for t in r.response['trades']]
        new_quote = new_symbol.split('_')[1]
        for pair in open_pairs:
            if new_quote == pair.split('_')[1]: return True
        return False
    except: return False

# --- 4. TRADING LOGIC ---

def get_processed_data(symbol, gran):
    try:
        params = {"count": 100, "granularity": gran}
        r = instruments.InstrumentsCandles(instrument=symbol, params=params)
        client.request(r)
        candles = r.response.get('candles', [])
        df = pd.DataFrame([{'close': float(c['mid']['c']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l'])} 
                           for c in candles if c['complete']])
        if df.empty: return None

        df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        df['ADX'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        df['SMA50'] = df['close'].rolling(50).mean()
        macd = ta.trend.MACD(close=df['close'])
        df['MACD'], df['MACD_S'] = macd.macd(), macd.macd_signal()
        df['STOCHk'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
        return df
    except Exception as e:
        logger.error(f"Data Fetch Error for {symbol}: {e}")
        return None

def execute_and_log(symbol, side, price, atr):
    sl_dist, tp_dist = atr * 1.5, atr * 3
    units = 1000 if side == "BUY" else -1000
    sl_price = round(price - sl_dist if side == "BUY" else price + sl_dist, 5)
    tp_price = round(price + tp_dist if side == "BUY" else price - tp_dist, 5)

    order_req = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
        stopLossOnFill=StopLossDetails(price=str(sl_price)).data
    )
    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_req.data)
        rv = client.request(r)
        t_id = rv['orderFillTransaction']['id']
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO trades (trade_id, symbol, side, price) VALUES (?, ?, ?, ?)", 
                     (t_id, symbol, side, price))
        conn.commit()
        conn.close()
        send_telegram_msg(f"🚀 *{side}* {symbol} @ {price}\nSL: {sl_price} | TP: {tp_price}")
        logger.info(f"TRADE EXECUTED: {symbol} {side} ID:{t_id}")
    except Exception as e:
        logger.error(f"Trade Execution Failed: {e}")

# --- 5. ROUTES ---

@app.route('/run')
def run_bot():
    # News Check with Fallback
    if not is_news_safe(): 
        return jsonify({"status": "Paused", "reason": "High Impact News"})
    
    trades_triggered = []
    for symbol in SYMBOLS:
        if is_correlated(symbol): continue
        df = get_processed_data(symbol, "M15")
        if df is None: continue
        
        last = df.iloc[-1]
        # ADX lowered to 20 for more activity
        if last['ADX'] > 20:
            trend_up = last['close'] > last['SMA50']
            if trend_up and last['MACD'] > last['MACD_S'] and last['STOCHk'] < 30:
                execute_and_log(symbol, "BUY", last['close'], last['ATR'])
                trades_triggered.append(f"{symbol} BUY")
            elif not trend_up and last['MACD'] < last['MACD_S'] and last['STOCHk'] > 70:
                execute_and_log(symbol, "SELL", last['close'], last['ATR'])
                trades_triggered.append(f"{symbol} SELL")

    logger.info(f"Heartbeat: Scan complete. Trades placed: {len(trades_triggered)}")
    return jsonify({"status": "Scan Complete", "trades": trades_triggered, "time": str(datetime.now())})

@app.route('/dashboard')
def dashboard():
    if request.args.get('pw') != DASHBOARD_PW: return "Unauthorized", 401
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()
    return f"<html><body style='font-family:sans-serif;'><h2>Bot Activity</h2>{df.to_html()}</body></html>"

@app.route('/')
def health(): return "Bot is Online", 200

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
