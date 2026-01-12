import os
import sqlite3
import logging
import pandas as pd
import ta  # Using the stable 'ta' library to avoid dependency errors
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

# Config - Ensure these are set in your Render environment variables
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
FMP_API_KEY = "A0xuQ94tqyjfAKitVIGoNKPNnBX2K0JT"
# For Render, mount a disk at /var/data
DB_PATH = os.environ.get("DB_PATH", "/var/data/bot_state.db")

client = API(access_token=OANDA_API_KEY, environment="practice")
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# --- 2. DATABASE (THE BOT'S MEMORY) ---
def init_db():
    """Initializes the persistent database for trade tracking."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trades 
                    (trade_id TEXT PRIMARY KEY, symbol TEXT, be_active INTEGER)''')
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized at {DB_PATH}")

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
    except Exception as e:
        logger.error(f"News check failed, defaulting to safe: {e}")
        return True

def is_correlated(new_symbol):
    """Prevents doubling risk on highly correlated pairs (e.g., two USD pairs)."""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        client.request(r)
        open_pairs = [t['instrument'] for t in r.response['trades']]
        if not open_pairs: return False
        
        new_base, new_quote = new_symbol.split('_')
        for pair in open_pairs:
            p_base, p_quote = pair.split('_')
            if new_quote == p_quote: # Check if same quote currency like USD
                logger.info(f"Correlation skip: Already have open {pair} trade.")
                return True
        return False
    except: return False

# --- 4. DATA ENGINE (Using stable 'ta' library) ---
def get_processed_data(symbol, gran):
    params = {"count": 100, "granularity": gran}
    r = instruments.InstrumentsCandles(instrument=symbol, params=params)
    client.request(r)
    
    df = pd.DataFrame([
        {
            'close': float(c['mid']['c']), 
            'high': float(c['mid']['h']), 
            'low': float(c['mid']['l'])
        } for c in r.response['candles'] if c['complete']
    ])
    
    # ATR (Volatility)
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # ADX (Trend Strength)
    adx_ind = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_ind.adx()
    
    # MACD (Momentum)
    macd_ind = ta.trend.MACD(close=df['close'])
    df['MACD'] = macd_ind.macd()
    df['MACD_Signal'] = macd_ind.macd_signal()
    
    # Stochastic (Overbought/Oversold)
    stoch_ind = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['STOCHk'] = stoch_ind.stoch()
    
    return df

# --- 5. EXECUTION & PERSISTENCE ---
def execute_and_log(symbol, side, price, atr):
    sl_dist = atr * 2
    tp_dist = atr * 4
    
    units = 1000 # Example fixed units
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
        
        # Save to Persistent Memory
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO trades (trade_id, symbol, be_active) VALUES (?, ?, ?)", (t_id, symbol, 0))
        conn.commit()
        conn.close()
        logger.info(f"🚀 {side} Order Executed & Logged: {symbol} ID {t_id}")
    except Exception as e:
        logger.error(f"Order failed: {e}")

# --- 6. FLASK ROUTES ---
@app.route('/run')
def run_bot():
    if not is_news_safe(): 
        return jsonify({"status": "Paused", "reason": "High Impact News"})
    
    results = []
    for symbol in SYMBOLS:
        try:
            if is_correlated(symbol): continue
            
            df_m15 = get_processed_data(symbol, "M15")
            df_d1 = get_processed_data(symbol, "D")
            
            last = df_m15.iloc[-1]
            # D1 50-period SMA for trend filter
            d1_sma50 = df_d1['close'].rolling(50).mean().iloc[-1]
            d1_trend_up = df_d1['close'].iloc[-1] > d1_sma50
            
            # Entry Signal: Trend Strength (ADX) + Momentum (MACD) + Reversal (Stoch)
            if last['ADX'] > 25:
                if d1_trend_up and last['MACD'] > last['MACD_Signal'] and last['STOCHk'] < 25:
                    execute_and_log(symbol, "BUY", last['close'], last['ATR'])
                    results.append(f"{symbol} BUY")
                elif not d1_trend_up and last['MACD'] < last['MACD_Signal'] and last['STOCHk'] > 75:
                    execute_and_log(symbol, "SELL", last['close'], last['ATR'])
                    results.append(f"{symbol} SELL")
                    
        except Exception as e:
            logger.error(f"Error checking {symbol}: {e}")
            
    return jsonify({"status": "Complete", "trades_opened": results})

@app.route('/')
def health():
    return "Bot is Live and Tracking State", 200

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
