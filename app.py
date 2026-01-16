import os, time, logging, threading, requests, numpy as np, pandas as pd, ta
from datetime import datetime
from flask import Flask, jsonify
from bayes_opt import BayesianOptimization # Updated for v3.2.0
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# ---------------- SETTINGS ----------------
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
RISK_PER_TRADE = 0.01  
BINANCE_URL = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=50"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [SYSTEM-APEX] | %(message)s")
app = Flask(__name__)
client = API(access_token=OANDA_API_KEY, environment="practice")

# ---------------- 1. MACRO ORDER FLOW (BINANCE PROXY) ----------------

def get_liquidity_imbalance():
    """Senses global liquidity direction via BTC Order Book"""
    try:
        # We look at the 'Weight' of the book (volume * distance from mid)
        data = requests.get(BINANCE_URL, timeout=2).json()
        bid_vol = sum([float(b[1]) for b in data['bids']])
        ask_vol = sum([float(a[1]) for a in data['asks']])
        # Positive = Buyers dominate (USD Weakness likely), Negative = Sellers (USD Strength)
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
    except: return 0

# ---------------- 2. SELF-OPTIMIZING BRAIN (v3.2.0 Compatible) ----------------

def objective_function(df, rsi_l, rsi_h, adx_t):
    """Optimizes for Sortino Ratio (Reward-to-Downside Risk)"""
    temp_df = df.copy()
    temp_df['RSI'] = ta.momentum.rsi(temp_df['close'], 14)
    temp_df['ADX'] = ta.trend.adx(temp_df['high'], temp_df['low'], temp_df['close'], 14)
    
    trades = []
    for i in range(1, len(temp_df)):
        # Range Mean-Reversion Logic
        if temp_df['ADX'].iloc[i] < adx_t:
            if temp_df['RSI'].iloc[i] < rsi_l: trades.append(temp_df['close'].iloc[i] - temp_df['close'].iloc[i-1])
            if temp_df['RSI'].iloc[i] > rsi_h: trades.append(temp_df['close'].iloc[i-1] - temp_df['close'].iloc[i])
            
    if len(trades) < 3: return -1.0
    downside = np.std([t for t in trades if t < 0]) or 0.0001
    return np.sum(trades) / downside

# ---------------- 3. EXECUTION ENGINE ----------------

def smart_order(symbol, side, price, atr, regime, imbalance):
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        nav = float(r.response["account"]["NAV"])
        
        # Volatility-Adjusted Positioning
        pip_val = 0.01 if "JPY" in symbol else 0.0001
        units = int((nav * RISK_PER_TRADE) / (atr * 3 * pip_val))
        if side == "SELL": units *= -1
        
        prec = 3 if "JPY" in symbol else 5
        tp_mult = 6 if regime == "TREND" else 3 # Asymmetric TP strategy

        order = MarketOrderRequest(
            instrument=symbol, units=units,
            takeProfitOnFill=TakeProfitDetails(price=str(round(price + (atr*tp_mult if side=="BUY" else -atr*tp_mult), prec))).data,
            stopLossOnFill=StopLossDetails(price=str(round(price + (-atr*3 if side=="BUY" else atr*3), prec))).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        logging.info(f">>> EXECUTED {side} {symbol} | Imbalance: {round(imbalance, 3)}")
    except Exception as e:
        logging.error(f"Execution Error: {e}")

# ---------------- 4. ASYNCHRONOUS CYCLE ----------------

def run_pair_logic(symbol):
    try:
        r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 250})
        client.request(r)
        df = pd.DataFrame([{"close": float(c["mid"]["c"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} for c in r.response["candles"]])
        
        # Bayesian Optimization (v3.2.0 Syntax)
        optimizer = BayesianOptimization(
            f=lambda l, h, a: objective_function(df, l, h, a),
            pbounds={'l': (20, 40), 'h': (60, 80), 'a': (20, 30)},
            verbose=0, random_state=42
        )
        # Optimized for speed in 2026 kernels
        optimizer.maximize(init_points=3, n_iter=5)
        best = optimizer.max['params']

        # Indicators & Sentiment
        imbalance = get_liquidity_imbalance()
        df['RSI'] = ta.momentum.rsi(df['close'], 14)
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        curr = df.iloc[-1]
        regime = "TREND" if curr['ADX'] > best['a'] else "RANGE"

        # Signal Matrix
        if regime == "RANGE":
            if curr['RSI'] < best['l'] and imbalance > 0.05:
                smart_order(symbol, "BUY", curr['close'], curr['ATR'], regime, imbalance)
            elif curr['RSI'] > best['h'] and imbalance < -0.05:
                smart_order(symbol, "SELL", curr['close'], curr['ATR'], regime, imbalance)
        else:
            slope = df['close'].rolling(10).mean().diff().iloc[-1]
            if slope > 0 and curr['RSI'] > 50 and imbalance > 0.1:
                smart_order(symbol, "BUY", curr['close'], curr['ATR'], regime, imbalance)
            elif slope < 0 and curr['RSI'] < 50 and imbalance < -0.1:
                smart_order(symbol, "SELL", curr['close'], curr['ATR'], regime, imbalance)

    except Exception as e:
        logging.error(f"Pair Logic Error ({symbol}): {e}")

@app.route("/run")
def master_trigger():
    if datetime.utcnow().weekday() >= 5: return jsonify({"status": "Market Closed"})
    
    threads = [threading.Thread(target=run_pair_logic, args=(s,)) for s in SYMBOLS]
    for t in threads: t.start()
    for t in threads: t.join()

    return jsonify({"status": "Institutional Cycle Complete", "timestamp": str(datetime.now())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
