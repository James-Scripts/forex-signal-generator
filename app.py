import os
import sqlite3
import logging
import pandas as pd
import ta
import time
from datetime import datetime
from flask import Flask, jsonify, request

# OANDA API Imports
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
DB_PATH = "bot_state.db"

# Initialize Client
client = API(access_token=OANDA_API_KEY, environment="practice")
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# --- 2. HELPER FUNCTIONS ---

def get_precision(symbol):
    """Ensures JPY pairs get 3 decimals and others get 5."""
    return 3 if "JPY" in symbol else 5

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                    (trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, 
                     price REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def is_already_open(symbol):
    """Prevents FIFO violations by checking for open positions."""
    try:
        r = positions.PositionDetails(accountID=OANDA_ACCOUNT_ID, instrument=symbol)
        client.request(r)
        pos = r.response.get('position', {})
        return int(pos.get('long', {}).get('units', 0)) != 0 or int(pos.get('short', {}).get('units', 0)) != 0
    except:
        return False

# --- 3. DATA & ANALYSIS ---

def get_market_data(symbol, granularity="M15"):
    try:
        params = {"count": 100, "granularity": granularity}
        r = instruments.InstrumentsCandles(instrument=symbol, params=params)
        client.request(r)
        
        candles = r.response.get('candles', [])
        data = [{'close': float(c['mid']['c']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l'])} 
                for c in candles if c['complete']]
        
        df = pd.DataFrame(data)
        if df.empty or len(df) < 50: return None

        # Indicators
        df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
        macd = ta.trend.MACD(df['close'], window_slow=17, window_fast=8, window_sign=9)
        df['MACD'], df['MACD_S'] = macd.macd(), macd.macd_signal()
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        return df
    except Exception as e:
        logger.warning(f"Connection issue for {symbol}: {e}")
        return None

# --- 4. EXECUTION ---

def execute_trade(symbol, side, price, atr):
    if is_already_open(symbol):
        logger.info(f"Skipping {symbol}: Position exists.")
        return False

    # Calculate units for ~$5 risk (1.5 * ATR distance)
    # If account is $100, 1% risk = $1
    units = 1000 if side == "BUY" else -1000 
    
    prec = get_precision(symbol)
    sl_val = round(price - (atr * 1.5) if side == "BUY" else price + (atr * 1.5), prec)
    tp_val = round(price + (atr * 3.0) if side == "BUY" else price - (atr * 3.0), prec)

    # The String Fix: Ensures OANDA accepts the decimal count
    sl_str = "{:.{}f}".format(sl_val, prec)
    tp_str = "{:.{}f}".format(tp_val, prec)

    order_req = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp_str).data,
        stopLossOnFill=StopLossDetails(price=sl_str).data
    )
    
    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_req.data)
        rv = client.request(r)
        if 'orderFillTransaction' in rv:
            logger.info(f"✅ {side} {symbol} EXECUTED at {price}")
            return True
    except Exception as e:
        logger.error(f"❌ Execution failed for {symbol}: {e}")
    return False

# --- 5. ROUTES ---

@app.route('/run')
def run_cycle():
    init_db()
    results = []
    
    for symbol in SYMBOLS:
        df = get_market_data(symbol)
        if df is None: continue
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # MACD Crossover Logic
        cross_up = prev['MACD'] < prev['MACD_S'] and curr['MACD'] > curr['MACD_S']
        cross_down = prev['MACD'] > prev['MACD_S'] and curr['MACD'] < curr['MACD_S']

        # Trend Filter: Price vs SMA50
        if curr['close'] > curr['SMA50'] and cross_up:
            if execute_trade(symbol, "BUY", curr['close'], curr['ATR']):
                results.append(f"BUY {symbol}")

        elif curr['close'] < curr['SMA50'] and cross_down:
            if execute_trade(symbol, "SELL", curr['close'], curr['ATR']):
                results.append(f"SELL {symbol}")

    return jsonify({"status": "Success", "trades": results, "time": str(datetime.now())})

@app.route('/')
def health(): return "Bot is Online", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
