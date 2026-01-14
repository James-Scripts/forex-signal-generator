import os
import sqlite3
import logging
import pandas as pd
import ta
import requests
import time
from datetime import datetime
from flask import Flask, jsonify, request

# OANDA API Imports
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Config - Use Environment Variables for security
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
DASHBOARD_PW = os.environ.get("DASHBOARD_PW", "1234")
DB_PATH = "bot_state.db"

# Initialize OANDA Client
client = API(access_token=OANDA_API_KEY, environment="practice")

# Symbols to watch
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# ANTI-BLOCK HEADERS
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.oanda.com/",
    "Connection": "keep-alive"
}

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                    (trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, 
                     price REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# --- 3. HELPER FUNCTIONS ---

def get_precision(symbol):
    """OANDA requires 3 decimals for JPY pairs and 5 for others."""
    return 3 if "JPY" in symbol else 5

# --- 4. TRADING LOGIC ---

def get_market_data(symbol):
    """Fetches candles and applies indicators."""
    try:
        params = {"count": 100, "granularity": "M15"}
        r = instruments.InstrumentsCandles(instrument=symbol, params=params)
        client.request(r)
        
        candles = r.response.get('candles', [])
        data = []
        for c in candles:
            if c['complete']:
                data.append({
                    'close': float(c['mid']['c']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l'])
                })
        
        df = pd.DataFrame(data)
        if df.empty: return None

        # Indicators
        df['SMA50'] = df['close'].rolling(50).mean()
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        macd = ta.trend.MACD(df['close'])
        df['MACD'], df['MACD_S'] = macd.macd(), macd.macd_signal()
        
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch data for {symbol}: {e}")
        return None

def execute_trade(symbol, side, price, atr):
    """Places a market order with SL and TP with correct precision."""
    units = 1000 if side == "BUY" else -1000
    sl_dist = atr * 1.5
    tp_dist = atr * 3.0
    
    # Calculate and round based on symbol precision
    prec = get_precision(symbol)
    sl_price = round(price - sl_dist if side == "BUY" else price + sl_dist, prec)
    tp_price = round(price + tp_dist if side == "BUY" else price - tp_dist, prec)

    # OANDA API requires prices to be strings
    order_req = MarketOrderRequest(
        instrument=symbol, 
        units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
        stopLossOnFill=StopLossDetails(price=str(sl_price)).data
    )
    
    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_req.data)
        rv = client.request(r)
        trade_id = rv['orderFillTransaction']['id']
        
        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO trades (trade_id, symbol, side, price) VALUES (?, ?, ?, ?)", 
                     (trade_id, symbol, side, price))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ TRADE PLACED: {symbol} {side} at {price} (SL: {sl_price}, TP: {tp_price})")
        return True
    except Exception as e:
        logger.error(f"❌ EXECUTION FAILED for {symbol}: {e}")
        return False

# --- 5. FLASK ROUTES ---

@app.route('/run')
def run_cycle():
    """Endpoint to trigger a scan (Heartbeat)."""
    results = []
    logger.info("Heartbeat: Starting market scan...")
    
    for symbol in SYMBOLS:
        df = get_market_data(symbol)
        if df is None: continue
        
        curr = df.iloc[-1]
        # Basic Strategy: Price above SMA50 and MACD Cross
        if curr['close'] > curr['SMA50'] and curr['MACD'] > curr['MACD_S']:
            if execute_trade(symbol, "BUY", curr['close'], curr['ATR']):
                results.append(f"BUY {symbol}")
        elif curr['close'] < curr['SMA50'] and curr['MACD'] < curr['MACD_S']:
            if execute_trade(symbol, "SELL", curr['close'], curr['ATR']):
                results.append(f"SELL {symbol}")
                
    logger.info(f"Scan complete. Trades found: {len(results)}")
    return jsonify({
        "status": "Success", 
        "trades_found": len(results), 
        "details": results, 
        "time": str(datetime.now())
    })

@app.route('/dashboard')
def view_dashboard():
    """Secured dashboard to see trade history."""
    if request.args.get('pw') != DASHBOARD_PW:
        return "Unauthorized", 401
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
        conn.close()
        return f"<h2>Bot Trade History (Last 20)</h2>{df.to_html(index=False)}"
    except Exception as e:
        return f"Error loading dashboard: {e}"

@app.route('/')
def health(): return "Bot is Online and Monitoring", 200

if __name__ == "__main__":
    init_db()
    # Port is handled by Render's environment
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
