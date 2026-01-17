import os, logging, threading, requests, time, numpy as np, pandas as pd, ta, openai
from datetime import datetime, timezone
import pytz
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
RISK_PER_TRADE = 0.005  # 0.5% risk
DAILY_LOSS_LIMIT = 0.02 # 2% Daily Kill-Switch
PORTFOLIO_LOCK = threading.Lock()

# Global state for Drawdown Guard
SESSION_START_NAV = None
SESSION_DATE = None

client = API(access_token=OANDA_API_KEY, environment="practice")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-PRO] | %(message)s")

# ================== LAYER 1: RISK & TIME UTILS ==================

def is_market_open():
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    if now.weekday() == 5: return False 
    if now.weekday() == 4 and now.hour >= 17: return False
    if now.weekday() == 6 and now.hour < 17: return False
    return True

def drawdown_guard_passed():
    """Implementation of Suggestion: Max Drawdown Limit."""
    global SESSION_START_NAV, SESSION_DATE
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        curr_nav = float(r.response["account"]["NAV"])
        today = datetime.now(timezone.utc).date()

        if SESSION_DATE != today or SESSION_START_NAV is None:
            SESSION_START_NAV = curr_nav
            SESSION_DATE = today
            return True

        drawdown = (SESSION_START_NAV - curr_nav) / SESSION_START_NAV
        if drawdown >= DAILY_LOSS_LIMIT:
            logging.critical(f"🚨 DAILY LIMIT HIT ({drawdown:.2%}). TRADING HALTED.")
            return False
        return True
    except Exception as e:
        logging.error(f"Guard Error: {e}")
        return False

# ================== LAYER 2: ANALYTICS ==================

def get_optimized_params(df):
    def fitness(rsi_l, rsi_h, adx_t):
        d = df.copy()
        d['rsi'] = ta.momentum.rsi(d['close'], 14)
        d['adx'] = ta.trend.adx(d['high'], d['low'], d['close'])
        pnl = []
        for i in range(1, len(d)):
            if d['adx'].iloc[i] < adx_t:
                if d['rsi'].iloc[i] < rsi_l: pnl.append(d['close'].iloc[i] - d['close'].iloc[i-1])
                elif d['rsi'].iloc[i] > rsi_h: pnl.append(d['close'].iloc[i-1] - d['close'].iloc[i])
        return np.mean(pnl) / (np.std([p for p in pnl if p < 0]) + 1e-6) if len(pnl) > 5 else -10

    opt = BayesianOptimization(f=fitness, pbounds={'rsi_l':(20,35), 'rsi_h':(65,80), 'adx_t':(20,30)}, verbose=0)
    opt.maximize(init_points=2, n_iter=3)
    return opt.max['params']

def check_trend_alignment(symbol):
    """Implementation of Suggestion: Multi-Timeframe Analysis (1H Filter)."""
    try:
        r = instruments.InstrumentsCandles(symbol, {"granularity": "H1", "count": 20})
        client.request(r)
        closes = [float(c["mid"]["c"]) for c in r.response["candles"]]
        sma = sum(closes) / len(closes)
        return "UP" if closes[-1] > sma else "DOWN"
    except: return "NEUTRAL"

# ================== LAYER 3: CORE LOGIC ==================

def get_macro_sentiment(symbol):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            news = requests.get(url).json().get("feed", [])[:5]
            prompt = f"Analyze {symbol} news: {[n['title'] for n in news]}. Score -1 to 1. If NFP/CPI imminent, return 'DANGER'."
            resp = ai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=15)
            return resp.choices[0].message.content.strip().upper()
        except openai.RateLimitError:
            time.sleep((2 ** attempt) * 8)
    return "0"

def execute_pro_trade(symbol, side, price, atr, nav):
    """Refined position sizing for any capital size."""
    pip = 0.01 if "JPY" in symbol else 0.0001
    # Ensure units are at least 1, even for small accounts ($10)
    units = max(1, int((nav * RISK_PER_TRADE) / (atr * 3 * pip)))
    if side == "SELL": units *= -1
    
    prec = 3 if "JPY" in symbol else 5
    order = MarketOrderRequest(
        instrument=symbol, units=units,
        stopLossOnFill=StopLossDetails(price=str(round(price - atr*3 if side=="BUY" else price + atr*3, prec))).data,
        takeProfitOnFill=TakeProfitDetails(price=str(round(price + atr*2 if side=="BUY" else price - atr*2, prec))).data,
        trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * 1.5, prec))).data
    )
    client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
    logging.info(f"🚀 {side} {symbol} | Units: {abs(units)} | Capital-Adjusted")

def run_apex_pro(symbol, index):
    time.sleep(index * 5)
    if not is_market_open() or not drawdown_guard_passed(): return

    with PORTFOLIO_LOCK:
        # 1. MTFA Alignment
        htf_trend = check_trend_alignment(symbol)
        
        # 2. News Guard
        sentiment = get_macro_sentiment(symbol)
        if "DANGER" in sentiment: return

        # 3. Indicator Optimization
        r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 150})
        client.request(r)
        df = pd.DataFrame([{"close":float(c["mid"]["c"]),"open":float(c["mid"]["o"]),"high":float(c["mid"]["h"]),"low":float(c["mid"]["l"])} for c in r.response["candles"]])
        
        params = get_optimized_params(df)
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
        
        # 4. Confluence
        c = df.iloc[-1]
        rsi = ta.momentum.rsi(df['close'], 14).iloc[-1]
        is_bullish_pat = c['close'] > df.iloc[-2]['open']
        
        signal = "NEUTRAL"
        if rsi < params['rsi_l'] and is_bullish_pat and htf_trend == "UP": signal = "BUY"
        if rsi > params['rsi_h'] and not is_bullish_pat and htf_trend == "DOWN": signal = "SELL"

        if signal != "NEUTRAL":
            try:
                ai_score = float(sentiment) if any(char.isdigit() for char in sentiment) else 0
                if (signal == "BUY" and ai_score >= -0.1) or (signal == "SELL" and ai_score <= 0.1):
                    r_acc = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r_acc)
                    execute_pro_trade(symbol, signal, c['close'], atr, float(r_acc.response["account"]["NAV"]))
            except: pass

@app.route('/')
def home(): return jsonify({"status": "Active", "guard": "Drawdown Protected"}), 200

@app.route("/run")
def trigger():
    threads = [threading.Thread(target=run_apex_pro, args=(s, i)) for i, s in enumerate(SYMBOLS)]
    for t in threads: t.start()
    return jsonify({"status": "Cycle Launched"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
