import os, time, logging, threading, requests, numpy as np, pandas as pd, ta
from datetime import datetime, time as dt_time
from flask import Flask, jsonify
from bayes_opt import BayesianOptimization
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# ---------------- QUANT SETTINGS (The Final Frontier) ----------------
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
RISK_PER_TRADE = 0.01  # 1% Strict Mathematical Risk
BINANCE_ORDERBOOK_URL = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [QUANT-CORE] | %(message)s")
app = Flask(__name__)
client = API(access_token=OANDA_API_KEY, environment="practice")

# ---------------- 1. CRYPTO-LIQUIDITY RADAR (Order Flow Proxy) ----------------

def get_global_liquidity_bias():
    """Reads real Binance Order Book depth to sense global USD appetite."""
    try:
        data = requests.get(BINANCE_ORDERBOOK_URL, timeout=2).json()
        bids = sum([float(b[1]) for b in data['bids']]) # Buying power
        asks = sum([float(a[1]) for a in data['asks']]) # Selling power
        imbalance = (bids - asks) / (bids + asks)
        # High imbalance means money is moving into assets (USD Weakness)
        return imbalance 
    except: return 0

# ---------------- 2. BAYESIAN WALK-FORWARD ENGINE ----------------

def alpha_fitness(df, rsi_l, rsi_h, adx_t):
    """Calculates Sortino Ratio (Risk-Adjusted Return) for last 200 periods."""
    df = df.copy()
    df['RSI'] = ta.momentum.rsi(df['close'], 14)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
    
    returns = []
    for i in range(1, len(df)):
        if df['ADX'].iloc[i] < adx_t: # Range optimization
            if df['RSI'].iloc[i] < rsi_l: returns.append(df['close'].iloc[i] - df['close'].iloc[i-1])
            if df['RSI'].iloc[i] > rsi_h: returns.append(df['close'].iloc[i-1] - df['close'].iloc[i])
            
    if not returns: return -10
    downside_std = np.std([r for r in returns if r < 0]) or 0.0001
    return np.sum(returns) / downside_std

# ---------------- 3. INSTITUTIONAL EXECUTION ----------------

def execute_with_precision(symbol, side, price, atr, regime, liq_bias):
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    client.request(r)
    nav = float(r.response["account"]["NAV"])
    
    # ATR-Based Dynamic Volatility Sizing
    pip_val = 0.01 if "JPY" in symbol else 0.0001
    sl_dist = atr * 3
    units = int((nav * RISK_PER_TRADE) / (sl_dist * pip_val))
    if side == "SELL": units *= -1

    prec = 3 if "JPY" in symbol else 5
    # Asymmetric TP: Trends get 6x, Ranges get 3x
    tp_mult = 6 if regime == "TREND" else 3 

    order = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(round(price + (atr*tp_mult if side=="BUY" else -atr*tp_mult), prec))).data,
        stopLossOnFill=StopLossDetails(price=str(round(price + (-atr*3 if side=="BUY" else atr*3), prec))).data
    )
    client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
    logging.info(f"🔥 ORDER SENT: {side} {symbol} | LIQ_BIAS: {round(liq_bias, 2)}")

# ---------------- 4. THE ASYNCHRONOUS PROCESSING ----------------

def analyze_and_trade(symbol):
    try:
        # Fetching M15 Data
        r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 200})
        client.request(r)
        df = pd.DataFrame([{"close": float(c["mid"]["c"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} for c in r.response["candles"]])
        
        # Bayesian Optimization: Finds the "hidden" thresholds of the current hour
        opt = BayesianOptimization(f=lambda l, h, a: alpha_fitness(df, l, h, a),
                                   pbounds={'l': (20, 38), 'h': (62, 80), 'a': (20, 30)}, verbose=0)
        opt.maximize(init_points=2, n_iter=3)
        p = opt.max['params']

        # Intelligence Gathering
        liq_bias = get_global_liquidity_bias()
        df['RSI'] = ta.momentum.rsi(df['close'], 14)
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        curr = df.iloc[-1]
        regime = "TREND" if curr['ADX'] > p['a'] else "RANGE"

        # Apex Signal Logic (Technical + Bayesian + Order Flow)
        if regime == "RANGE":
            if curr['RSI'] < p['l'] and liq_bias > 0.05: # Buy if oversold + Global Liquidity is positive
                execute_with_precision(symbol, "BUY", curr['close'], curr['ATR'], regime, liq_bias)
            elif curr['RSI'] > p['h'] and liq_bias < -0.05: # Sell if overbought + Global Liquidity is negative
                execute_with_precision(symbol, "SELL", curr['close'], curr['ATR'], regime, liq_bias)
        
        elif regime == "TREND":
            slope = df['close'].rolling(12).mean().diff().iloc[-1]
            if slope > 0 and curr['RSI'] > 52 and liq_bias > 0.1:
                execute_with_precision(symbol, "BUY", curr['close'], curr['ATR'], regime, liq_bias)
            elif slope < 0 and curr['RSI'] < 48 and liq_bias < -0.1:
                execute_with_precision(symbol, "SELL", curr['close'], curr['ATR'], regime, liq_bias)

    except Exception as e:
        logging.error(f"Critical Failure {symbol}: {e}")

@app.route("/run")
def master_hub():
    if datetime.utcnow().weekday() >= 5: return jsonify({"status": "Weekend Mode"})
    
    threads = [threading.Thread(target=analyze_and_trade, args=(s,)) for s in SYMBOLS]
    for t in threads: t.start()
    for t in threads: t.join()

    return jsonify({"status": "Institutional Scan Complete", "timestamp": str(datetime.now())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
