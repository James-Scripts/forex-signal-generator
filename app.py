import os
import time
import logging
import pandas as pd
import ta
import feedparser
from datetime import datetime, time as dt_time
from flask import Flask, jsonify
from openai import OpenAI

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails, TrailingStopLossDetails

# ---------------- CONFIGURATION ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

app = Flask(__name__)

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = API(access_token=OANDA_API_KEY, environment="practice")
ai_client = OpenAI(api_key=OPENAI_API_KEY)

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# ---------------- RISK & TIME FILTERS ----------------

def is_market_active():
    """Active Trading Hours: 08:00 - 17:00 UTC"""
    now = datetime.utcnow()
    # Check if it's a weekday
    if now.weekday() >= 5: return False 
    return dt_time(8, 0) <= now.time() <= dt_time(17, 0)

def is_weekend_closing_time():
    """Friday Risk-Off: Close everything 1 hour before market close"""
    now = datetime.utcnow()
    # Friday (weekday 4) after 20:00 UTC
    return now.weekday() == 4 and now.hour >= 20

def close_all_positions():
    """Safety Protocol: Wipes all open trades before the weekend"""
    try:
        for symbol in SYMBOLS:
            # Check for Long or Short positions
            data = {"longUnits": "ALL", "shortUnits": "ALL"}
            r = positions.PositionClose(OANDA_ACCOUNT_ID, symbol, data)
            client.request(r)
            logger.warning(f"🧹 Weekend Cleanup: Closed all positions for {symbol}")
        return True
    except: return False

# ---------------- AI & DATA HELPERS ----------------

def get_market_sentiment():
    try:
        feed = feedparser.parse("https://xml.fxstreet.com/news/rss.xml")
        headlines = [entry.title for entry in feed.entries[:10]]
        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Analyze headlines. Reply with ONLY: BULLISH, BEARISH, or NEUTRAL for the USD."},
                {"role": "user", "content": str(headlines)}
            ]
        )
        return response.choices[0].message.content.strip().upper()
    except: return "NEUTRAL"

def fetch_data(symbol, tf, count):
    try:
        r = instruments.InstrumentsCandles(symbol, {"granularity": tf, "count": count})
        client.request(r)
        data = [{"close": float(c["mid"]["c"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} 
                for c in r.response["candles"] if c["complete"]]
        df = pd.DataFrame(data)
        df["SMA50"] = ta.trend.sma_indicator(df["close"], 50)
        macd = ta.trend.MACD(df["close"], 17, 8, 9)
        df["MACD"], df["MACD_S"] = macd.macd(), macd.macd_signal()
        df["RSI"] = ta.momentum.rsi(df["close"], 14)
        df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
        return df
    except: return None

# ---------------- TRADING EXECUTION ----------------

def execute_trade(symbol, side, price, atr):
    units = 5 if side == "BUY" else -5
    prec = 3 if "JPY" in symbol else 5
    sl_p = f"{round(price - (atr*3) if side=='BUY' else price + (atr*3), prec):.{prec}f}"
    tp_p = f"{round(price + (atr*6) if side=='BUY' else price - (atr*6), prec):.{prec}f}"
    tr_d = f"{round(atr*2.5, prec):.{prec}f}"

    order = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp_p).data,
        stopLossOnFill=StopLossDetails(price=sl_p).data,
        trailingStopLossOnFill=TrailingStopLossDetails(distance=tr_d).data
    )
    client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
    logger.info(f"✅ Trade Placed: {side} {symbol}")

# ---------------- APP ROUTES ----------------

@app.route("/run")
def run_bot():
    # 1. Friday Risk-Off Check
    if is_weekend_closing_time():
        close_all_positions()
        return jsonify({"status": "Risk-Off", "message": "Closed all positions for the weekend."})

    # 2. Market Hours Check
    if not is_market_active():
        return jsonify({"status": "Paused", "reason": "Outside Active Market Hours (08:00-17:00 UTC)"})

    # 3. AI Sentiment Check
    usd_bias = get_market_sentiment()
    executed = []

    for symbol in SYMBOLS:
        m15 = fetch_data(symbol, "M15", 100)
        if m15 is None: continue
        curr, prev = m15.iloc[-1], m15.iloc[-2]

        sig_buy = (prev["MACD"] < prev["MACD_S"] and curr["MACD"] > curr["MACD_S"] and 40 < curr["RSI"] < 60)
        sig_sell = (prev["MACD"] > prev["MACD_S"] and curr["MACD"] < curr["MACD_S"] and 40 < curr["RSI"] < 60)

        # 4. Strategy Alignment
        if usd_bias == "BULLISH":
            if symbol == "USD_JPY" and sig_buy: execute_trade(symbol, "BUY", curr["close"], curr["ATR"]); executed.append(symbol)
            if symbol in ["EUR_USD", "GBP_USD"] and sig_sell: execute_trade(symbol, "SELL", curr["close"], curr["ATR"]); executed.append(symbol)
        elif usd_bias == "BEARISH":
            if symbol in ["EUR_USD", "GBP_USD"] and sig_buy: execute_trade(symbol, "BUY", curr["close"], curr["ATR"]); executed.append(symbol)
            if symbol == "USD_JPY" and sig_sell: execute_trade(symbol, "SELL", curr["close"], curr["ATR"]); executed.append(symbol)

    return jsonify({"sentiment": usd_bias, "trades": executed})

@app.route("/")
def health(): return "AI-Quant Master Active: Weekend Risk-Off Mode Enabled."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
