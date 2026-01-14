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

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
DASHBOARD_PW = os.environ.get("DASHBOARD_PW", "1234")
DB_PATH = "bot_state.db"

client = API(access_token=OANDA_API_KEY, environment="practice")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                        (trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, 
                         price REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Init Error: {e}")

# --- 3. HELPER FUNCTIONS ---
def get_precision(symbol):
    return 3 if "JPY" in symbol else 5

# --- 4. TRADING LOGIC ---

def get_market_data(symbol):
    """Fetches candles and applies filtered indicators."""
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
        if df.empty or len(df) < 50: return None

        # --- ADVANCED INDICATORS ---
        # 1. Trend Filter
        df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # 2. Faster MACD (8, 17, 9) to catch moves early
        macd = ta.trend.MACD(df['close'], window_slow=17, window_fast=8, window_sign=9)
        df['MACD'], df['MACD_S'] = macd.macd(), macd.macd_signal()
        
        # 3. RSI Filter (Strength check)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        # 4. Volatility (ATR)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch data for {symbol}: {e}")
        return None

def execute_trade(symbol, side, price, atr):
    """Places a market order with SL and TP with updated $5 size."""
    units = 5 if side == "BUY" else -5 
    prec = get_precision(symbol)
    
    # Risk Management: 1.5x ATR Stop Loss, 3x ATR Take Profit
    sl_price = round(price - (atr * 1.5) if side == "BUY" else price + (atr * 1.5), prec)
    tp_price = round(price + (atr * 3) if side == "BUY" else price - (atr * 3), prec)

    order_req = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
        stopLossOnFill=StopLossDetails(price=str(sl_price)).data
    )
    
    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_req.data)
        rv = client.request(r)
        
        if 'orderFillTransaction' in rv:
            trade_id = rv['orderFillTransaction']['id']
            logger.info(f"✅ $5 TRADE PLACED: {symbol} {side} ID:{trade_id}")
            
            # Save to Database
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO trades (trade_id, symbol, side, price) VALUES (?, ?, ?, ?)", 
                         (trade_id, symbol, side, price))
            conn.commit()
            conn.close()
            return True
        else:
            reason = rv.get('orderRejectTransaction', {}).get('rejectReason', 'Unknown')
            logger.warning(f"⚠️ Order Reject: {symbol} Reason: {reason}")
            return False
    except Exception as e:
        logger.error(f"❌ EXECUTION FAILED for {symbol}: {str(e)}")
        return False

# --- 5. FLASK ROUTES ---

@app.route('/run')
def run_cycle():
    init_db() # Ensure DB is ready
    results = []
    logger.info("Heartbeat: Starting Filtered Market Scan...")
    
    for symbol in SYMBOLS:
        df = get_market_data(symbol)
        if df is None: continue
        
        curr = df.iloc[-1]
        
        # --- IMPROVED STRATEGY LOGIC ---
        # BUY Condition: 
        # 1. Price above SMA50 (Bullish trend)
        # 2. MACD crosses above Signal
        # 3. RSI < 65 (Not overbought)
        if curr['close'] > curr['SMA50'] and curr['MACD'] > curr['MACD_S'] and curr['RSI'] < 65:
            if execute_trade(symbol, "BUY", curr['close'], curr['ATR']):
                results.append(f"BUY {symbol} (RSI: {round(curr['RSI'], 1)})")
        
        # SELL Condition: 
        # 1. Price below SMA50 (Bearish trend)
        # 2. MACD crosses below Signal
        # 3. RSI > 35 (Not oversold)
        elif curr['close'] < curr['SMA50'] and curr['MACD'] < curr['MACD_S'] and curr['RSI'] > 35:
            if execute_trade(symbol, "SELL", curr['close'], curr['ATR']):
                results.append(f"SELL {symbol} (RSI: {round(curr['RSI'], 1)})")
                
    logger.info(f"Scan complete. Trades found: {len(results)}")
    return jsonify({"status": "Success", "trades_found": len(results), "details": results})

@app.route('/dashboard')
def view_dashboard():
    if request.args.get('pw') != DASHBOARD_PW: return "Unauthorized", 401
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
        conn.close()
        return f"<h2>Bot Trade History</h2>{df.to_html(index=False)}"
    except: return "No trades recorded yet."

@app.route('/')
def health(): return "Bot is Online", 200

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
