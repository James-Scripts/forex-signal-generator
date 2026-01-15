import os, time, logging, feedparser, pandas as pd, ta, numpy as np
from datetime import datetime, time as dt_time
from flask import Flask, jsonify
from openai import OpenAI
from bayes_opt import BayesianOptimization # pip install bayesian-optimization

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails, TrailingStopLossDetails

# ================= CONFIG & RISK =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MASTER_QUANT")

app = Flask(__name__)

# Credentials
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = API(access_token=OANDA_API_KEY, environment="practice")
ai_client = OpenAI(api_key=OPENAI_API_KEY)

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
BASE_RISK = 0.01  # 1% per trade
MAX_DAILY_DD = 0.05 # 5% circuit breaker

# ================= BAYESIAN OPTIMIZER (Self-Learning) =================

def evaluate_strategy(df, rsi_low, rsi_high):
    """Backtests parameters in real-time to find the best win-rate"""
    pnl = []
    for i in range(1, len(df)):
        if df['RSI'].iloc[i] < rsi_low:
            pnl.append(df['close'].iloc[i] - df['close'].iloc[i-1]) # Buying
        elif df['RSI'].iloc[i] > rsi_high:
            pnl.append(df['close'].iloc[i-1] - df['close'].iloc[i]) # Selling
    if len(pnl) < 2: return 0
    return np.mean(pnl) / (np.std(pnl) + 1e-6) # Sharpe Ratio

def get_optimal_params(df):
    """Finds the best RSI levels for the CURRENT market volatility"""
    def objective(rsi_low, rsi_high):
        return evaluate_strategy(df, rsi_low, rsi_high)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={'rsi_low': (20, 45), 'rsi_high': (55, 80)},
        verbose=0, random_state=1
    )
    optimizer.maximize(init_points=5, n_iter=10)
    return optimizer.max['params']

# ================= INTELLIGENCE LAYERS =================

def get_account_health():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    client.request(r)
    nav = float(r.response['account']['NAV'])
    bal = float(r.response['account']['balance'])
    return nav, bal

def get_sentiment():
    """AI Sentiment Layer to align with 'Massive News' moves"""
    try:
        feed = feedparser.parse("https://xml.fxstreet.com/news/rss.xml")
        context = "\n".join([e.title for e in feed.entries[:10]])
        res = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Return ONLY: BULLISH, BEARISH, or NEUTRAL for USD."},
                      {"role": "user", "content": context}]
        )
        return res.choices[0].message.content.strip().upper()
    except: return "NEUTRAL"

# ================= EXECUTION =================

def fetch_data(symbol, timeframe="M15", count=200):
    r = instruments.InstrumentsCandles(symbol, {"granularity": timeframe, "count": count})
    client.request(r)
    rows = [{"close": float(c["mid"]["c"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} 
            for c in r.response["candles"] if c["complete"]]
    df = pd.DataFrame(rows)
    df["SMA50"] = ta.trend.sma_indicator(df["close"], 50)
    df["RSI"] = ta.momentum.rsi(df["close"], 14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    return df.dropna()

def execute(symbol, side, price, atr):
    # Dynamic Position Sizing based on 1% Risk
    nav, _ = get_account_health()
    pip_val = 0.01 if "JPY" in symbol else 0.0001
    sl_pips = (atr * 3) / pip_val
    units = int((nav * BASE_RISK) / (sl_pips * pip_val))
    if side == "SELL": units *= -1

    prec = 3 if "JPY" in symbol else 5
    sl = round(price - (atr*3) if side=="BUY" else price + (atr*3), prec)
    tp = round(price + (atr*6) if side=="BUY" else price - (atr*6), prec)
    tr = round(atr*2.5, prec)

    order = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(tp)).data,
        stopLossOnFill=StopLossDetails(price=str(sl)).data,
        trailingStopLossOnFill=TrailingStopLossDetails(distance=str(tr)).data
    )
    client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
    logger.info(f"💰 {side} {symbol} | Optimal Bayesian Entry | Units: {units}")

# ================= MAIN ROUTER =================

@app.route("/run")
def run():
    nav, bal = get_account_health()
    if nav < bal * (1 - MAX_DAILY_DD): return jsonify({"status": "CIRCUIT_BREAKER_ACTIVE"})

    sentiment = get_sentiment()
    executed = []

    for s in SYMBOLS:
        df = fetch_data(s)
        params = get_optimal_params(df)
        curr = df.iloc[-1]
        
        # Check Bayesian Optimized RSI Levels
        buy_sig = curr["RSI"] < params['rsi_low']
        sell_sig = curr["RSI"] > params['rsi_high']
        trend_up = curr["close"] > curr["SMA50"]

        # Final Alignment with AI Sentiment
        if sentiment == "BULLISH" and s == "USD_JPY" and buy_sig and trend_up:
            execute(s, "BUY", curr["close"], curr["ATR"]); executed.append(s)
        elif sentiment == "BEARISH" and s in ["EUR_USD", "GBP_USD"] and buy_sig and trend_up:
            execute(s, "BUY", curr["close"], curr["ATR"]); executed.append(s)

    return jsonify({"sentiment": sentiment, "trades": executed})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
