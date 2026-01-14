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

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

COOLDOWN = 3600  # 1 hour per pair
LAST_TRADE = {}

# ---------------- HELPERS ----------------

def precision(symbol):
    return 3 if "JPY" in symbol else 5

def fetch_data(symbol, tf, count):
    try:
        r = instruments.InstrumentsCandles(symbol, {"granularity": tf, "count": count})
        client.request(r)
        candles = r.response["candles"]

        data = []
        for c in candles:
            if c["complete"]:
                data.append({
                    "close": float(c["mid"]["c"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"])
                })

        df = pd.DataFrame(data)
        if len(df) < 50:
            return None

        df["SMA50"] = ta.trend.sma_indicator(df["close"], 50)
        macd = ta.trend.MACD(df["close"], 17, 8, 9)
        df["MACD"] = macd.macd()
        df["MACD_S"] = macd.macd_signal()
        df["RSI"] = ta.momentum.rsi(df["close"], 14)
        df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

        return df
    except Exception as e:
        logger.error(f"Data error {symbol} {tf}: {e}")
        return None

def has_open_position(symbol):
    try:
        r = positions.PositionDetails(OANDA_ACCOUNT_ID, symbol)
        client.request(r)
        pos = r.response["position"]
        return int(pos["long"]["units"]) != 0 or int(pos["short"]["units"]) != 0
    except:
        return False

# ---------------- EXECUTION ----------------

def execute_trade(symbol, side, price, atr):
    # 1. FIFO & Cooldown Protection
    if has_open_position(symbol):
        logger.info(f"{symbol}: Skipped (FIFO active trade).")
        return False

    if symbol in LAST_TRADE and time.time() - LAST_TRADE[symbol] < COOLDOWN:
        logger.info(f"{symbol}: Cooldown active.")
        return False

    # 2. Dynamic Position Sizing (The $5 / 1% Risk Improvement)
    try:
        balance = get_account_balance()
        if balance <= 0:
            balance = 500  # Fallback if API fails to fetch balance
            
        risk_amount = balance * 0.01  # Risking 1% of account
        # Units = Risk Amount / (ATR * 1.5)
        units = int(risk_amount / (atr * 1.5))
        
        # Ensure a minimum viable trade size (OANDA minimum is usually 1 unit)
        if units < 1: units = 100 
    except:
        units = 1000  # Safe default if balance check fails
        
    if side == "SELL": units *= -1

    # 3. Precision & Formatting Fixes
    prec = precision(symbol)
    sl_val = round(price - atr*1.5 if side=="BUY" else price + atr*1.5, prec)
    tp_val = round(price + atr*3 if side=="BUY" else price - atr*3, prec)

    # Convert to strings with exact decimals for OANDA
    sl_str = f"{sl_val:.{prec}f}"
    tp_str = f"{tp_val:.{prec}f}"

    # 4. Constructing the Order with correct .data attributes
    order = MarketOrderRequest(
        instrument=symbol,
        units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp_str).data,
        stopLossOnFill=StopLossDetails(price=sl_str).data
    )

    # 5. API Request
    try:
        r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data)
        client.request(r)
        
        # Success: Update Cooldown and Log
        LAST_TRADE[symbol] = time.time()
        logger.info(f"🚀 {side} {symbol} | Units: {units} | SL: {sl_str} | TP: {tp_str}")
        return True
    except Exception as e:
        logger.error(f"❌ OANDA ERROR {symbol}: {e}")
        return False




# ---------------- STRATEGY ----------------

@app.route("/run")
def run_bot():
    trades = []

    for symbol in SYMBOLS:

        h1 = fetch_data(symbol, "H1", 60)
        if h1 is None: continue

        h1_trend = "BULL" if h1.iloc[-1]["close"] > h1.iloc[-1]["SMA50"] else "BEAR"

        df = fetch_data(symbol, "M15", 120)
        if df is None: continue

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        macd_up = prev["MACD"] < prev["MACD_S"] and curr["MACD"] > curr["MACD_S"]
        macd_dn = prev["MACD"] > prev["MACD_S"] and curr["MACD"] < curr["MACD_S"]

        # BUY
        if h1_trend=="BULL" and macd_up and curr["RSI"] < 65:
            if execute_trade(symbol, "BUY", curr["close"], curr["ATR"]):
                trades.append(f"BUY {symbol}")

        # SELL
        elif h1_trend=="BEAR" and macd_dn and curr["RSI"] > 35:
            if execute_trade(symbol, "SELL", curr["close"], curr["ATR"]):
                trades.append(f"SELL {symbol}")

    return jsonify({"status":"complete","trades":trades})

@app.route("/")
def health():
    return "OANDA Trading Bot Online"

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
