import os, time, logging
import numpy as np
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

# =========================
# PRODUCTION CONFIG
# =========================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV = "practice" 

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
RISK_PER_TRADE = 0.01  # 1% Account Risk
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0   # SL distance
TP_RATIO = 1.5         # 1.5x SL distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

# Dictionary to track the last candle we processed per pair
last_candle_time = {inst: None for inst in INSTRUMENTS}

# =========================
# CORE FUNCTIONS
# =========================

def get_account_balance():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    api.request(r)
    return float(r.response["account"]["NAV"])

def has_open_position(inst):
    """Safety: Checks if we already have a trade in this instrument."""
    try:
        r = positions.PositionDetails(OANDA_ACCOUNT_ID, instrument=inst)
        api.request(r)
        # If long or short units are non-zero, we have a position
        pos = r.response.get("position", {})
        long_units = float(pos.get("long", {}).get("units", 0))
        short_units = float(pos.get("short", {}).get("units", 0))
        return abs(long_units) > 0 or abs(short_units) > 0
    except:
        return False

def get_advanced_data(inst, tf="M5", count=250):
    """Fetches OHLC and calculates ATR + EMA Trend Filter."""
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=inst, params=params)
    api.request(r)
    
    df = pd.DataFrame([
        {'time': c['time'], 'o': float(c['mid']['o']), 'h': float(c['mid']['h']), 
         'l': float(c['mid']['l']), 'c': float(c['mid']['c'])} 
        for c in r.response['candles'] if c['complete']
    ])
    
    # 1. Calculate True Range
    df['h-l'] = df['h'] - df['l']
    df['h-pc'] = abs(df['h'] - df['c'].shift(1))
    df['l-pc'] = abs(df['l'] - df['c'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    # 2. Market Regime Filter (200 EMA)
    df['ema_200'] = df['c'].ewm(span=200, adjust=False).mean()
    
    return df

def calculate_units(inst, balance, sl_pips, price):
    """Professional Position Sizing."""
    # Simplified pip value logic (needs adjustment for JPY pairs)
    pip_val = 0.01 if "JPY" in inst else 0.0001
    risk_amount = balance * RISK_PER_TRADE
    # Units = Risk / (SL distance in pips * pip value)
    sl_dist_pips = sl_pips / pip_val
    units = risk_amount / (sl_dist_pips * pip_val)
    return int(units)

# =========================
# TRADING ENGINE
# =========================

def run_strategy():
    logging.info("--- OANDA PRODUCTION ENGINE ENGAGED ---")
    
    while True:
        balance = get_account_balance()
        
        for inst in INSTRUMENTS:
            df = get_advanced_data(inst)
            if df.empty: continue
            
            curr_candle = df.iloc[-1]
            last_close = curr_candle['c']
            atr = curr_candle['atr']
            ema = curr_candle['ema_200']
            
            # 1. New Candle Check
            if last_candle_time[inst] == curr_candle['time']:
                continue
            
            # 2. Position Check
            if has_open_position(inst):
                continue

            # 3. Strategy Logic (Price > EMA + Slope)
            slope = (df['c'].iloc[-1] - df['c'].iloc[-10]) / 10
            direction = None
            
            if last_close > ema and slope > 0:
                direction = "LONG"
            elif last_close < ema and slope < 0:
                direction = "SHORT"
                
            if direction:
                # 4. Sizing & Risk
                sl_dist = atr * ATR_MULTIPLIER
                units = calculate_units(inst, balance, sl_dist, last_close)
                
                sl_price = last_close - sl_dist if direction == "LONG" else last_close + sl_dist
                tp_price = last_close + (sl_dist * TP_RATIO) if direction == "LONG" else last_close - (sl_dist * TP_RATIO)
                
                # 5. Place Order
                order_data = {
                    "order": {
                        "instrument": inst, "units": str(units if direction == "LONG" else -units),
                        "type": "MARKET", "timeInForce": "GTC",
                        "stopLossOnFill": {"price": f"{sl_price:.5f}"},
                        "takeProfitOnFill": {"price": f"{tp_price:.5f}"}
                    }
                }
                try:
                    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_data)
                    api.request(r)
                    logging.info(f"TRADE: {inst} {direction} | Units: {units} | SL: {sl_price:.4f}")
                    last_candle_time[inst] = curr_candle['time']
                except Exception as e:
                    logging.error(f"Order Failed: {e}")

        time.sleep(15) # Pulse every 15s, but candle-lock handles frequency

if __name__ == "__main__":
    run_strategy()
