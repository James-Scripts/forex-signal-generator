import os, time, logging, threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

# ==========================================
# WEB SERVER SETUP (REQUIRED FOR RENDER)
# ==========================================
app = Flask(__name__)

@app.route('/')
def health_check():
    """Render health check endpoint."""
    return jsonify({
        "status": "active",
        "bot": "Oanda Institutional Engine",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }), 200

# =========================
# PRODUCTION CONFIG
# =========================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV = "practice" 

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
RISK_PER_TRADE = 0.01      # 1% of Balance
MAX_UNITS_LIMIT = 500000   # Safety cap to prevent accidental over-leveraging
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0   
TP_RATIO = 1.5         

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

# Tracking to prevent duplicate entries on the same 5-minute candle
last_candle_time = {inst: None for inst in INSTRUMENTS}

# =========================
# UTILITY FUNCTIONS
# =========================

def get_account_balance():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    api.request(r)
    return float(r.response["account"]["NAV"])

def has_open_position(inst):
    """Checks if a trade is already active to prevent stacking."""
    try:
        r = positions.PositionDetails(OANDA_ACCOUNT_ID, instrument=inst)
        api.request(r)
        pos = r.response.get("position", {})
        long_units = float(pos.get("long", {}).get("units", 0))
        short_units = float(pos.get("short", {}).get("units", 0))
        return abs(long_units) > 0 or abs(short_units) > 0
    except Exception:
        return False

def get_advanced_data(inst, tf="M5", count=250):
    """Calculates True Range ATR and 200 EMA Filter."""
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=inst, params=params)
    api.request(r)
    
    df = pd.DataFrame([
        {'time': c['time'], 'o': float(c['mid']['o']), 'h': float(c['mid']['h']), 
         'l': float(c['mid']['l']), 'c': float(c['mid']['c'])} 
        for c in r.response['candles'] if c['complete']
    ])
    
    # ATR Calculation
    df['h-l'] = df['h'] - df['l']
    df['h-pc'] = abs(df['h'] - df['c'].shift(1))
    df['l-pc'] = abs(df['l'] - df['c'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    # Trend Filter
    df['ema_200'] = df['c'].ewm(span=200, adjust=False).mean()
    return df

def calculate_units(inst, balance, sl_dist, price):
    """Institutional Position Sizing based on ATR volatility."""
    pip_val = 0.01 if "JPY" in inst else 0.0001
    risk_amount = balance * RISK_PER_TRADE
    
    # Calculate units: (Balance * Risk%) / (StopLoss Distance)
    units = risk_amount / sl_dist
    
    # Apply safety cap
    final_units = int(min(units, MAX_UNITS_LIMIT))
    return final_units

# =========================
# TRADING ENGINE
# =========================

def run_strategy():
    logging.info("--- OANDA PRODUCTION ENGINE ENGAGED ---")
    while True:
        try:
            balance = get_account_balance()
            for inst in INSTRUMENTS:
                df = get_advanced_data(inst)
                if df.empty: continue
                
                curr_candle = df.iloc[-1]
                last_close = curr_candle['c']
                atr = curr_candle['atr']
                ema = curr_candle['ema_200']
                
                # Check for new candle and existing positions
                if last_candle_time[inst] == curr_candle['time']: continue
                if has_open_position(inst): continue

                # Entry Logic: Trend Direction + Slope
                slope = (df['c'].iloc[-1] - df['c'].iloc[-10]) / 10
                direction = None
                
                if last_close > ema and slope > 0:
                    direction = "LONG"
                elif last_close < ema and slope < 0:
                    direction = "SHORT"
                
                if direction:
                    sl_dist = atr * ATR_MULTIPLIER
                    units = calculate_units(inst, balance, sl_dist, last_close)
                    
                    sl_price = last_close - sl_dist if direction == "LONG" else last_close + sl_dist
                    tp_price = last_close + (sl_dist * TP_RATIO) if direction == "LONG" else last_close - (sl_dist * TP_RATIO)
                    
                    # FIXED ORDER DATA
                    order_data = {
                        "order": {
                            "instrument": inst,
                            "units": str(units if direction == "LONG" else -units),
                            "type": "MARKET",
                            "timeInForce": "FOK",  # Fixed: Market orders must be FOK
                            "positionFill": "DEFAULT",
                            "stopLossOnFill": {
                                "price": f"{sl_price:.5f}",
                                "timeInForce": "GTC"
                            },
                            "takeProfitOnFill": {
                                "price": f"{tp_price:.5f}",
                                "timeInForce": "GTC"
                            }
                        }
                    }
                    
                    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_data)
                    api.request(r)
                    logging.info(f"SUCCESS: {inst} {direction} | Units: {units} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                    
                    # Update candle tracker to avoid multiple entries on the same bar
                    last_candle_time[inst] = curr_candle['time']
                    
        except Exception as e:
            logging.error(f"Execution Error: {e}")
            
        time.sleep(20) # Scans every 20 seconds

# =========================
# DEPLOYMENT STARTUP
# =========================

# Run bot in a daemon thread so the Flask server stays responsive
bot_thread = threading.Thread(target=run_strategy, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    # Get port from environment for Render/Heroku
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
