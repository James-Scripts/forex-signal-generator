import os, time, logging, threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

app = Flask(__name__)

@app.route('/')
def health_check():
    return jsonify({"status": "running", "bot": "Oanda Win-Rate Optimized"}), 200

# =========================
# PRODUCTION CONFIG
# =========================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-001-37473620-001")
OANDA_ENV = "practice" 

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
RISK_PER_TRADE = 0.01  # 1% Account Risk per trade
MAX_UNITS = 50000      # SAFETY CAP: Limits position size to prevent massive losses
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5   # Increased slightly to give trades more room to breathe
RSI_PERIOD = 14

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)
last_candle_time = {inst: None for inst in INSTRUMENTS}

# =========================
# TECHNICAL INDICATORS
# =========================

def compute_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_advanced_data(inst, tf="M5", count=250):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=inst, params=params)
    api.request(r)
    
    df = pd.DataFrame([
        {'time': c['time'], 'h': float(c['mid']['h']), 'l': float(c['mid']['l']), 'c': float(c['mid']['c'])} 
        for c in r.response['candles'] if c['complete']
    ])
    
    # ATR for Volatility-based Stop Loss
    df['tr'] = np.maximum(df['h'] - df['l'], 
                          np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    # Trend and Momentum Filters
    df['ema_200'] = df['c'].ewm(span=200, adjust=False).mean()
    df['rsi'] = compute_rsi(df['c'], RSI_PERIOD)
    
    return df

def calculate_units(inst, balance, sl_dist):
    """Protects account balance by limiting unit size based on price distance."""
    pip_val = 0.01 if "JPY" in inst else 0.0001
    risk_amount = balance * RISK_PER_TRADE
    
    # Formula: Risk Amount / SL distance in price
    raw_units = risk_amount / sl_dist
    
    # Apply safety cap to resolve the huge losses shown in screenshots
    return int(min(raw_units, MAX_UNITS))

def has_open_position(inst):
    try:
        r = positions.PositionDetails(OANDA_ACCOUNT_ID, instrument=inst)
        api.request(r)
        pos = r.response.get("position", {})
        return abs(float(pos.get("long", {}).get("units", 0))) > 0 or \
               abs(float(pos.get("short", {}).get("units", 0))) > 0
    except:
        return False

# =========================
# TRADING ENGINE
# =========================

def run_strategy():
    logging.info("--- WIN-RATE OPTIMIZED ENGINE ENGAGED ---")
    while True:
        try:
            r_acc = accounts.AccountSummary(OANDA_ACCOUNT_ID)
            api.request(r_acc)
            balance = float(r_acc.response["account"]["NAV"])

            for inst in INSTRUMENTS:
                df = get_advanced_data(inst)
                if df.empty: continue
                
                curr = df.iloc[-1]
                prev_candles = df.iloc[-10:] # Used for slope calculation
                
                if last_candle_time[inst] == curr['time']: continue
                if has_open_position(inst): continue

                # HIGH WIN-RATE LOGIC: Trend + Momentum + Slope
                ema_slope = (curr['ema_200'] - prev_candles['ema_200'].iloc[0]) / 10
                direction = None

                # Only buy if in uptrend AND RSI is not overbought (prevents buying the peak)
                if curr['c'] > curr['ema_200'] and ema_slope > 0 and 45 < curr['rsi'] < 65:
                    direction = "LONG"
                # Only sell if in downtrend AND RSI is not oversold (prevents selling the bottom)
                elif curr['c'] < curr['ema_200'] and ema_slope < 0 and 35 < curr['rsi'] < 55:
                    direction = "SHORT"

                if direction:
                    sl_dist = curr['atr'] * ATR_MULTIPLIER
                    units = calculate_units(inst, balance, sl_dist)
                    
                    sl_price = curr['c'] - sl_dist if direction == "LONG" else curr['c'] + sl_dist
                    tp_price = curr['c'] + (sl_dist * 1.5) if direction == "LONG" else curr['c'] - (sl_dist * 1.5)

                    # CORRECTED ORDER: timeInForce is FOK to stop rejections
                    order_data = {
                        "order": {
                            "instrument": inst,
                            "units": str(units if direction == "LONG" else -units),
                            "type": "MARKET",
                            "timeInForce": "FOK", 
                            "positionFill": "DEFAULT",
                            "stopLossOnFill": {"price": f"{sl_price:.5f}", "timeInForce": "GTC"},
                            "takeProfitOnFill": {"price": f"{tp_price:.5f}", "timeInForce": "GTC"}
                        }
                    }
                    
                    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_data)
                    api.request(r)
                    logging.info(f"TRADE: {inst} {direction} | Units: {units} | RSI: {curr['rsi']:.2f}")
                    last_candle_time[inst] = curr['time']

        except Exception as e:
            logging.error(f"Strategy Error: {e}")
        time.sleep(15)

# =========================
# STARTUP
# =========================
bot_thread = threading.Thread(target=run_strategy, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
