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
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# --- 2. RETRY DECORATOR ---
def retry_request(max_tries=3, initial_delay=2, backoff=2):
    """Retries a function with exponential backoff if it fails."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            tries, delay = max_tries, initial_delay
            while tries > 0:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    tries -= 1
                    if tries == 0:
                        logger.error(f"❌ Final failure for {f.__name__}: {e}")
                        raise e
                    logger.warning(f"⚠️ {f.__name__} failed. Retrying in {delay}s... ({tries} left)")
                    time.sleep(delay)
                    delay *= backoff
            return None
        return wrapper
    return decorator

# --- 3. DATABASE ---
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
    logger.info(f"✅ Database initialized at {DB_PATH}")

# --- 4. TELEGRAM ---
def send_telegram_msg(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logger.error(f"Telegram failed: {e}")

# --- 5. SAFETY FILTERS (NEWS & CORRELATION) ---

@retry_request(max_tries=3)
def is_news_safe():
    """Checks both OANDA Labs and Finnhub for high-impact news."""
    now = datetime.utcnow()
    
    # Check OANDA Labs (Forex-specific)
    try:
        params = {"instrument": "EUR_USD", "period": 3600} 
        r = labs.Calendar(params=params)
        client.request(r)
        for event in r.response:
            if int(event.get('impact', 0)) == 3: # 3 = High Impact
                e_ts = event.get('timestamp')
                if abs(e_ts - now.timestamp()) < 1800: # 30-minute window
                    msg = f"⚠️ OANDA NEWS ALERT: {event.get('title')}"
                    send_telegram_msg(msg)
                    return False
    except Exception as e:
        logger.error(f"OANDA news check failed: {e}")

    # Check Finnhub (Global Economic)
    if FINNHUB_API_KEY:
        try:
            url = f"https://finnhub.io/api/v1/calendar/economic?token={FINNHUB_API_KEY}"
            res = requests.get(url, timeout=5).json()
            for event in res.get('economicCalendar', []):
                if str(event.get('impact')).lower() in ['high', '3']:
                    e_time = datetime.strptime(event['time'], '%Y-%m-%d %H:%M:%S')
                    if abs((e_time - now).total_seconds()) < 1800:
                        msg = f"⚠️ FINNHUB NEWS ALERT: {event['event']}"
                        send_telegram_msg(msg)
                        return False
        except Exception as e:
            logger.error(f"Finnhub news check failed: {e}")

    return True

@retry_request(max_tries=2)
def is_correlated(new_symbol):
    """Prevents opening multiple trades with the same quote currency."""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        client.request(r)
        open_pairs = [t['instrument'] for t in r.response['trades']]
        new_quote = new_symbol.split('_')[1]
        for pair in open_pairs:
            if new_quote == pair.split('_')[1]:
                logger.info(f"Correlation skip: {pair} already open.")
                return True
        return False
    except: 
        return False

# --- 6. DATA & EXECUTION ---

@retry_request(max_tries=3)
def get_processed_data(symbol, gran):
    """Fetches candles and calculates indicators."""
    params = {"count": 100, "granularity": gran}
    r = instruments.InstrumentsCandles(instrument=symbol, params=params)
    client.request(r)
    
    candles = r.response.get('candles', [])
    df = pd.DataFrame([{'close': float(c['mid']['c']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l'])} 
                       for c in candles if c['complete']])
    
    if df.empty: return None

    df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    macd = ta.trend.MACD(close=df['close'])
    df['MACD'], df['MACD_S'] = macd.macd(), macd.macd_signal()
    df['STOCHk'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    return df

def execute_and_log(symbol, side, price, atr):
    """Executes market order and logs to DB."""
    sl_dist, tp_dist = atr * 2, atr * 4
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
        logger.info(f"Order Executed: {t_id}")
    except Exception as e:
        logger.error(f"Order failed for {symbol}: {e}")

# --- 7. FLASK ROUTES ---

@app.route('/run')
def run_bot():
    if not is_news_safe(): 
        return jsonify({"status": "Paused", "reason": "High Impact News Detected"})
    
    results = []
    for symbol in SYMBOLS:
        try:
            if is_correlated(symbol): continue
            
            df_m15 = get_processed_data(symbol, "M15")
            df_d1 = get_processed_data(symbol, "D")
            
            if df_m15 is None or df_d1 is None: continue
            
            last = df_m15.iloc[-1]
            d1_trend_up = df_d1['close'].iloc[-1] > df_d1['close'].rolling(50).mean().iloc[-1]
            
            if last['ADX'] > 25:
                if d1_trend_up and last['MACD'] > last['MACD_S'] and last['STOCHk'] < 25:
                    execute_and_log(symbol, "BUY", last['close'], last['ATR'])
                    results.append(f"{symbol} BUY")
                elif not d1_trend_up and last['MACD'] < last['MACD_S'] and last['STOCHk'] > 75:
                    execute_and_log(symbol, "SELL", last['close'], last['ATR'])
                    results.append(f"{symbol} SELL")
        except Exception as e: 
            logger.error(f"Logic Error for {symbol}: {e}")
            
    return jsonify({"status": "Complete", "trades_triggered": results, "timestamp": str(datetime.now())})

@app.route('/dashboard')
def dashboard():
    if request.args.get('pw') != DASHBOARD_PW: return "Unauthorized", 401
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
        conn.close()
        return f"""
        <html><head><title>Bot Dashboard</title><style>
        body{{font-family:sans-serif; padding:20px; background:#f9f9f9;}} 
        table{{width:100%; border-collapse:collapse; background:white;}} 
        th,td{{padding:12px; border:1px solid #ddd; text-align:left;}} 
        th{{background:#333; color:white;}}
        tr:nth-child(even){{background:#f2f2f2;}}
        </style></head>
        <body><h2>Last 20 Trades</h2>{df.to_html(index=False)}</body></html>
        """
    except Exception as e: return f"Dashboard Error: {str(e)}"

@app.route('/')
def health(): return "Bot is Online", 200

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
