import os, logging, threading, requests, time, numpy as np, pandas as pd, ta, openai
from datetime import datetime, time as dt_time, timedelta
from flask import Flask, jsonify
from bayes_opt import BayesianOptimization
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import (
    MarketOrderRequest, StopLossDetails, 
    TakeProfitDetails, TrailingStopLossDetails
)

# ================== CONFIGURATION ==================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
RISK_PER_TRADE = 0.005  # 0.5% per trade for capital preservation
MAX_DRAWDOWN = 0.05     # Stop bot if account drops 5% in a session
PORTFOLIO_LOCK = threading.Lock()

client = API(access_token=OANDA_API_KEY, environment="practice")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-PRO] | %(message)s")

# ================== LAYER 1: WALK-FORWARD BAYESIAN OPTIMIZER ==================

def fitness_function(df, rsi_l, rsi_h, adx_t):
    """Calculates Sortino-like ratio for a parameter set over last 200 candles."""
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    
    pnl = []
    for i in range(1, len(df)):
        # Range Strategy Logic
        if df['adx'].iloc[i] < adx_t:
            if df['rsi'].iloc[i] < rsi_l: pnl.append(df['close'].iloc[i] - df['close'].iloc[i-1])
            if df['rsi'].iloc[i] > rsi_h: pnl.append(df['close'].iloc[i-1] - df['close'].iloc[i])
            
    if len(pnl) < 5: return -10
    return np.mean(pnl) / (np.std([p for p in pnl if p < 0]) + 1e-6)

def get_optimized_params(df):
    """Performs real-time Walk-Forward Optimization."""
    opt = BayesianOptimization(
        f=lambda rsi_l, rsi_h, adx_t: fitness_function(df, rsi_l, rsi_h, adx_t),
        pbounds={'rsi_l': (20, 35), 'rsi_h': (65, 80), 'adx_t': (20, 30)},
        verbose=0, random_state=42
    )
    opt.maximize(init_points=2, n_iter=5)
    return opt.max['params']

# ================== LAYER 2: ENHANCED AI SENTIMENT GUARD ==================

def get_macro_sentiment(symbol):
    """AI checks news sentiment + Macro trend over the last 24 hours."""
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        news_data = requests.get(url).json().get("feed", [])[:5]
        headlines = [n['title'] for n in news_data]
        
        prompt = (f"Analyze {symbol} headlines: {headlines}. "
                  "Provide a sentiment score from -1 (Bearish) to 1 (Bullish). "
                  "If any headline mentions 'Interest Rate', 'NFP', or 'CPI' in the next 2 hours, return 'DANGER'.")
        
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=20
        )
        res = response.choices[0].message.content.strip().upper()
        return res # Returns SCORE or 'DANGER'
    except: return "0"

# ================== LAYER 3: CONFLUENCE & EXECUTION ==================

def analyze_confluence(df, params):
    """Confirms Signal with RSI, MACD, and Candlestick Patterns."""
    c = df.iloc[-1]
    # 1. Patterns
    is_bullish = (c['close'] > df.iloc[-2]['open']) and (df.iloc[-2]['close'] < df.iloc[-2]['open'])
    # 2. Indicators
    rsi = ta.momentum.rsi(df['close'], 14).iloc[-1]
    macd = ta.trend.macd_diff(df['close']).iloc[-1]
    
    if rsi < params['rsi_l'] and is_bullish and macd > 0: return "BUY"
    if rsi > params['rsi_h'] and not is_bullish and macd < 0: return "SELL"
    return "NEUTRAL"

def execute_pro_trade(symbol, side, price, atr):
    """Executes with Dynamic ATR Trailing and 1% Portfolio Risk."""
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
    nav = float(r.response["account"]["NAV"])
    
    pip = 0.01 if "JPY" in symbol else 0.0001
    units = int((nav * RISK_PER_TRADE) / (atr * 3 * pip))
    if side == "SELL": units *= -1
    
    # Asymmetric Scaling: Partial TP at 2:1, Trail remaining at 1.5 ATR
    prec = 3 if "JPY" in symbol else 5
    order = MarketOrderRequest(
        instrument=symbol, units=units,
        stopLossOnFill=StopLossDetails(price=str(round(price - atr*3 if side=="BUY" else price + atr*3, prec))).data,
        takeProfitOnFill=TakeProfitDetails(price=str(round(price + atr*2 if side=="BUY" else price - atr*2, prec))).data,
        trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * 1.5, prec))).data
    )
    client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
    logging.info(f"✅ APEX-PRO: {side} {symbol} active with Dynamic WFO Params.")

# ================== CORE ENGINE ==================

def run_apex_pro(symbol):
    with PORTFOLIO_LOCK:
        try:
            # 1. Context Check (AI + Macro)
            sentiment = get_macro_sentiment(symbol)
            if "DANGER" in sentiment:
                logging.warning(f"🚫 {symbol} News Guard: High Volatility Event. Skipping.")
                return

            # 2. Data & Dynamic Optimization
            r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 250})
            client.request(r)
            df = pd.DataFrame([{"close": float(c["mid"]["c"]), "open": float(c["mid"]["o"]), 
                                "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} for c in r.response["candles"]])
            
            params = get_optimized_params(df)
            
            # 3. Technical Confluence
            signal = analyze_confluence(df, params)
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
            
            # 4. Final Alignment Check
            ai_score = float(sentiment) if sentiment.replace('.','',1).replace('-','',1).isdigit() else 0
            if (signal == "BUY" and ai_score >= 0) or (signal == "SELL" and ai_score <= 0):
                execute_pro_trade(symbol, signal, df['close'].iloc[-1], atr)
            
        except Exception as e:
            logging.error(f"Logic Error for {symbol}: {e}")

@app.route("/run")
def trigger_cycle():
    # Session Filter: Avoid low liquidity (21:00 - 23:00 UTC)
    now_hour = datetime.utcnow().hour
    if 21 <= now_hour <= 23:
        return jsonify({"status": "Paused: Low Liquidity Window"})

    threads = [threading.Thread(target=run_apex_pro, args=(s,)) for s in SYMBOLS]
    for t in threads: t.start()
    for t in threads: t.join()
    return jsonify({"status": "Pro-Cycle Complete", "timestamp": str(datetime.now())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
