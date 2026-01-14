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
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# --- 1. SETTINGS & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
DB_PATH = "bot_state.db"

client = API(access_token=OANDA_API_KEY, environment="practice")
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
RISK_PER_TRADE = 0.01  # Risk 1% of balance per trade

# --- 2. DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                    (trade_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, 
                     price REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# --- 3. ADVANCED ANALYSIS ---

def get_market_data(symbol, granularity="M15", count=100):
    try:
        params = {"count": count, "granularity": granularity}
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

        # Indicators
        df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
        macd = ta.trend.MACD(df['close'], window_slow=17, window_fast=8, window_sign=9)
        df['MACD'], df['MACD_S'] = macd.macd(), macd.macd_signal()
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        return df
    except Exception as e:
        logger.error(f"Data Fetch Error ({symbol} {granularity}): {e}")
        return None

def get_account_balance():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    client.request(r)
    return float(r.response['account']['balance'])

# --- 4. EXECUTION ENGINE ---

def execute_trade(symbol, side, price, atr):
    balance = get_account_balance()
    
    # Position Sizing Logic: Risk $ / (Stop Loss Distance)
    # 1 pip = 0.0001 for most, 0.01 for JPY. 
    pip_val = 0.01 if "JPY" in symbol else 0.0001
    sl_pips = (atr * 1.5) / pip_val
    
    # Calculate units to risk 1% of balance
    risk_amount = balance * RISK_PER_TRADE
    # Units = Risk / (SL pips * pip value) -> simplified to:
    units = int(risk_amount / (atr * 1.5))
    if side == "SELL": units *= -1

    prec = 3 if "JPY" in symbol else 5
    sl_price = round(price - (atr * 1.5) if side == "BUY" else price + (atr * 1.5), prec)
    tp_price = round(price + (atr * 3.0) if side == "BUY" else price - (atr * 3.0), prec)

    order_req = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp_price)).data,
        stopLossOnFill=StopLossDetails(price=str(sl_price)).data
    )
    
    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_req.data)
        rv = client.request(r)
        if 'orderFillTransaction' in rv:
            logger.info(f"🚀 {side} {symbol} | Units: {units} | SL: {sl_price} | TP: {tp_price}")
            return True
    except Exception as e:
        logger.error(f"Trade Failed: {e}")
    return False

# --- 5. THE BRAIN (STRATEGY) ---

@app.route('/run')
def run_cycle():
    init_db()
    results = []
    
    for symbol in SYMBOLS:
        # Step 1: Check Higher Timeframe (H1) for Trend Filter
        df_h1 = get_market_data(symbol, "H1", 50)
        if df_h1 is None: continue
        h1_trend = "BULL" if df_h1['close'].iloc[-1] > df_h1['SMA50'].iloc[-1] else "BEAR"

        # Step 2: Get M15 Data for Execution
        df = get_market_data(symbol, "M15", 100)
        if df is None: continue
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # MACD Crossover Detection
        cross_up = prev['MACD'] < prev['MACD_S'] and curr['MACD'] > curr['MACD_S']
        cross_down = prev['MACD'] > prev['MACD_S'] and curr['MACD'] < curr['MACD_S']

        # Logic for BUY
        if h1_trend == "BULL" and cross_up and curr['RSI'] < 65:
            if execute_trade(symbol, "BUY", curr['close'], curr['ATR']):
                results.append(f"BUY {symbol}")

        # Logic for SELL
        elif h1_trend == "BEAR" and cross_down and curr['RSI'] > 35:
            if execute_trade(symbol, "SELL", curr['close'], curr['ATR']):
                results.append(f"SELL {symbol}")

    return jsonify({"status": "Complete", "trades": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
