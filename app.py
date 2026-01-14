import os
import time
import logging
import pandas as pd
import ta
from flask import Flask, jsonify

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# ---------------- SETTINGS ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

app = Flask(__name__)

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
client = API(access_token=OANDA_API_KEY, environment="practice")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# ---------------- HELPERS ----------------

def precision(symbol):
    return 3 if "JPY" in symbol else 5

def fetch_data(symbol, tf, count):
    # Connection Protection: Sleep 1 second before every request to avoid "RemoteDisconnected"
    time.sleep(1) 
    try:
        r = instruments.InstrumentsCandles(symbol, {"granularity": tf, "count": count})
        client.request(r)
        candles = r.response["candles"]
        data = [{"close": float(c["mid"]["c"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} 
                for c in candles if c["complete"]]
        df = pd.DataFrame(data)
        if len(df) < 50: return None

        df["SMA50"] = ta.trend.sma_indicator(df["close"], 50)
        macd = ta.trend.MACD(df["close"], 17, 8, 9)
        df["MACD"], df["MACD_S"] = macd.macd(), macd.macd_signal()
        df["RSI"] = ta.momentum.rsi(df["close"], 14)
        df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
        return df
    except Exception as e:
        logger.error(f"Data error {symbol}: {e}")
        return None

def has_open_position(symbol):
    try:
        r = positions.PositionDetails(OANDA_ACCOUNT_ID, symbol)
        client.request(r)
        pos = r.response["position"]
        return int(pos["long"]["units"]) != 0 or int(pos["short"]["units"]) != 0
    except:
        return False

# ---------------- EXECUTION (LOCKED TO 5 UNITS) ----------------

def execute_trade(symbol, side, price, atr):
    if has_open_position(symbol):
        logger.info(f"Skipping {symbol}: Existing trade found.")
        return False

    # HARD LIMIT: 5 Units Only
    units = 5 if side == "BUY" else -5

    prec = precision(symbol)
    sl_dist = atr * 1.5
    tp_dist = atr * 3.0
    
    sl = f"{round(price - sl_dist if side=='BUY' else price + sl_dist, prec):.{prec}f}"
    tp = f"{round(price + tp_dist if side=='BUY' else price - tp_dist, prec):.{prec}f}"

    order = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp).data,
        stopLossOnFill=StopLossDetails(price=sl).data
    )

    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data)
        client.request(r)
        logger.info(f"✅ PLACED 5 UNITS: {side} {symbol}")
        return True
    except Exception as e:
        logger.error(f"❌ OANDA REJECTED {symbol}: {e}")
        return False

# ---------------- STRATEGY ----------------

@app.route("/run")
def run_bot():
    trades = []
    for symbol in SYMBOLS:
        h1 = fetch_data(symbol, "H1", 60)
        m15 = fetch_data(symbol, "M15", 100)
        if h1 is None or m15 is None: continue

        h1_curr = h1.iloc[-1]
        m15_curr, m15_prev = m15.iloc[-1], m15.iloc[-2]

        # Trend Filter
        h1_bull = h1_curr["close"] > h1_curr["SMA50"]
        
        # MACD Signals
        macd_up = m15_prev["MACD"] < m15_prev["MACD_S"] and m15_curr["MACD"] > m15_curr["MACD_S"]
        macd_dn = m15_prev["MACD"] > m15_prev["MACD_S"] and m15_curr["MACD"] < m15_curr["MACD_S"]

        if h1_bull and macd_up and m15_curr["RSI"] < 65:
            if execute_trade(symbol, "BUY", m15_curr["close"], m15_curr["ATR"]):
                trades.append(f"BUY {symbol}")
        
        elif not h1_bull and macd_dn and m15_curr["RSI"] > 35:
            if execute_trade(symbol, "SELL", m15_curr["close"], m15_curr["ATR"]):
                trades.append(f"SELL {symbol}")
        else:
            logger.info(f"Scan {symbol}: No setup.")

    return jsonify({"status": "Complete", "trades": trades})

@app.route("/")
def health(): return "Bot Online - 5 Unit Limit Locked"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
