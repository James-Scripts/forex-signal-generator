import os
import sqlite3
import logging
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime
from flask import Flask, jsonify

# OANDA API
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Config
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
FMP_API_KEY = "A0xuQ94tqyjfAKitVIGoNKPNnBX2K0JT"
DB_PATH = os.environ.get("DB_PATH", "/var/data/bot_state.db")

client = API(access_token=OANDA_API_KEY, environment="practice")
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]
RISK_PERCENT = 0.01

# --- 2. DATABASE (THE BOT'S MEMORY) ---
def init_db():
    """Initializes the persistent database for trade tracking."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                    (trade_id TEXT PRIMARY KEY, symbol TEXT, be_active INTEGER)''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

# --- 3. SAFETY FILTERS ---
def is_news_safe():
    """Pauses trading if high-impact news is within 30 minutes."""
    try:
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={FMP_API_KEY}"
        res = requests.get(url, timeout=5).json()
        now = datetime.utcnow()
        for event in res[:15]:
            e_time = datetime.strptime(event['date'], '%Y-%m-%d %H:%M:%S')
            if event.get('impact') == 'High' and abs((e_time - now).total_seconds()) < 1800:
                logger.warning(f"PAUSED: High Impact News - {event['event']}")
                return False
        return True
    except: return True

def is_correlated(new_symbol):
    """Prevents doubling risk on highly correlated pairs."""
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    client.request(r)
    open_pairs = [t['instrument'] for t in r.response['trades']]
    if not open_pairs or new_symbol in open_pairs: return False
    
    # Simple logic: If we already have a USD pair, be cautious with another
    for pair in open_pairs:
        if new_symbol.split('_')[1] == pair.split('_')[1]:
            logger.info(f"Correlation skip: Already have open {pair} trade.")
            return True
    return False

# --- 4. DATA ENGINE ---
def get_processed_data(symbol, gran):
    r = instruments.InstrumentsCandles(instrument=symbol, params={"count": 100, "granularity": gran})
    client.request(r)
    df = pd.DataFrame([{'close': float(c['mid']['c']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l'])} for c in r.response['candles']])
    
    # Calculate Indicators
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)
    df['STOCHk'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']
    return df

# --- 5. EXECUTION & PERSISTENCE ---
def execute_and_log(symbol, side, price, atr):
    sl_dist = atr * 2
    tp_dist = atr * 4
    
    # Calculate Units (Risk Management)
    units = 1000 # Default fallback
    if side == "SELL": units *= -1
    
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
        
        # Save to Memory (Database)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO trades (trade_id, symbol, be_active) VALUES (?, ?, ?)", (t_id, symbol, 0))
        conn.commit()
        conn.close()
        logger.info(f"Trade Logged in DB: {symbol} ID {t_id}")
    except Exception as e:
        logger.error(f"Order failed: {e}")

# --- 6. FLASK ROUTES ---
@app.route('/run')
def run_bot():
    if not is_news_safe(): return jsonify({"status": "News Pause"})
    
    results = []
    for symbol in SYMBOLS:
        try:
            if is_correlated(symbol): continue
            
            df_m15 = get_processed_data(symbol, "M15")
            df_d1 = get_processed_data(symbol, "D")
            
            last = df_m15.iloc[-1]
            d1_trend_up = df_d1['close'].iloc[-1] > df_d1['close'].rolling(50).mean().iloc[-1]
            
            # Entry Signal: Strong Trend + Confluence
            if last['ADX'] > 25:
                if d1_trend_up and last['MACD_12_26_9'] > last['MACDs_12_26_9'] and last['STOCHk'] < 25:
                    execute_and_log(symbol, "BUY", last['close'], last['ATR'])
                    results.append(f"{symbol} BUY")
                elif not d1_trend_up and last['MACD_12_26_9'] < last['MACDs_12_26_9'] and last['STOCHk'] > 75:
                    execute_and_log(symbol, "SELL", last['close'], last['ATR'])
                    results.append(f"{symbol} SELL")
                    
        except Exception as e:
            logger.error(f"Error checking {symbol}: {e}")
            
    return jsonify({"status": "Complete", "trades": results})

@app.route('/')
def health():
    return "Bot is Live", 200

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
