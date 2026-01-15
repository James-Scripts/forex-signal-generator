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
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails, TrailingStopLossDetails

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
    for attempt in range(3):
        try:
            time.sleep(1.2) 
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
        except Exception:
            time.sleep(2)
            continue
    return None

def has_open_position(symbol):
    try:
        r = positions.PositionDetails(OANDA_ACCOUNT_ID, symbol)
        client.request(r)
        return True
    except:
        return False

# ---------------- EXECUTION WITH TRAILING STOP ----------------

def execute_trade(symbol, side, price, atr):
    if has_open_position(symbol): return False

    units = 5 if side == "BUY" else -5
    prec = precision(symbol)
    
    # 1. Initial Hard Stop Loss (2.5x ATR)
    sl_dist = atr * 2.5
    sl_price = f"{round(price - sl_dist if side=='BUY' else price + sl_dist, prec):.{prec}f}"
    
    # 2. Take Profit (Targeting 5x ATR for better Risk/Reward)
    tp_dist = atr * 5.0
    tp_price = f"{round(price + tp_dist if side=='BUY' else price - tp_dist, prec):.{prec}f}"

    # 3. Trailing Stop Distance (2.0x ATR)
    # This distance is fixed when the trade opens and "trails" the price
    trail_dist = f"{round(atr * 2.0, prec):.{prec}f}"

    order = MarketOrderRequest(
        instrument=symbol,
        units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp_price).data,
        stopLossOnFill=StopLossDetails(price=sl_price).data,
        trailingStopLossOnFill=TrailingStopLossDetails(distance=trail_dist).data
    )

    try:
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        logger.info(f"🚀 {side} {symbol} | Units: 5 | SL: {sl_price} | Trail: {trail_dist}")
        return True
    except Exception as e:
        logger.error(f"❌ Order Failed: {e}")
        return False

# ---------------- STRATEGY ----------------

@app.route("/run")
def run_bot():
    trades = []
    for symbol in SYMBOLS:
        h1 = fetch_data(symbol, "H1", 60)
        m15 = fetch_data(symbol, "M15", 100)
        if h1 is None or m15 is None: continue

        curr = m15.iloc[-1]
        prev = m15.iloc[-2]
        h1_trend_up = h1.iloc[-1]["close"] > h1.iloc[-1]["SMA50"]

        macd_buy = prev["MACD"] < prev["MACD_S"] and curr["MACD"] > curr["MACD_S"]
        macd_sell = prev["MACD"] > prev["MACD_S"] and curr["MACD"] < curr["MACD_S"]

        # BUY: Trend UP + Confirmed MACD Cross + Healthy RSI
        if h1_trend_up and macd_buy and 40 < curr["RSI"] < 60:
            if execute_trade(symbol, "BUY", curr["close"], curr["ATR"]):
                trades.append(f"BUY {symbol}")
        
        # SELL: Trend DOWN + Confirmed MACD Cross + Healthy RSI
        elif not h1_trend_up and macd_sell and 40 < curr["RSI"] < 60:
            if execute_trade(symbol, "SELL", curr["close"], curr["ATR"]):
                trades.append(f"SELL {symbol}")

    return jsonify({"status": "Complete", "executed": trades})

@app.route("/")
def health(): return "Bot Active: Trailing Stop Enabled (5 Units)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
