import os
import time
import logging
import pandas as pd
import ta
import feedparser
from datetime import datetime
from flask import Flask, jsonify
from openai import OpenAI  # pip install openai

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails, TrailingStopLossDetails

# ---------------- CONFIGURATION ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

app = Flask(__name__)

# API Keys from Environment Variables (Set these in Render!)
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALPHA_VANTAGE_KEY = "ZTH6HG7SPAT753XH"

client = API(access_token=OANDA_API_KEY, environment="practice")
ai_client = OpenAI(api_key=OPENAI_API_KEY)

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]

# ---------------- AI SENTIMENT ENGINE ----------------

def get_market_sentiment():
    """Uses GPT-4 to read headlines and return a bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'"""
    try:
        # 1. Get latest headlines via RSS
        feed = feedparser.parse("https://xml.fxstreet.com/news/rss.xml")
        headlines = [entry.title for entry in feed.entries[:10]]
        context = "\n".join(headlines)

        # 2. Ask AI for Directional Bias (Focus on USD)
        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior currency strategist. Analyze these headlines and respond with ONLY one word: BULLISH, BEARISH, or NEUTRAL regarding the US Dollar (USD)."},
                {"role": "user", "content": f"Headlines:\n{context}"}
            ]
        )
        sentiment = response.choices[0].message.content.strip().upper()
        logger.info(f"🤖 AI Sentiment Analysis: USD is {sentiment}")
        return sentiment
    except Exception as e:
        logger.error(f"AI Sentiment Error: {e}")
        return "NEUTRAL"

# ---------------- TRADING LOGIC ----------------

def fetch_data(symbol, tf, count):
    try:
        time.sleep(1)
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

def execute_trade(symbol, side, price, atr):
    units = 5 if side == "BUY" else -5
    prec = 3 if "JPY" in symbol else 5
    
    # Dynamic ATR protection (Wider stops for news volatility)
    sl_dist, tp_dist, trail_dist = atr * 3.0, atr * 6.0, atr * 2.5
    
    sl_p = f"{round(price - sl_dist if side=='BUY' else price + sl_dist, prec):.{prec}f}"
    tp_p = f"{round(price + tp_dist if side=='BUY' else price - tp_dist, prec):.{prec}f}"
    tr_d = f"{round(trail_dist, prec):.{prec}f}"

    order = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp_p).data,
        stopLossOnFill=StopLossDetails(price=sl_p).data,
        trailingStopLossOnFill=TrailingStopLossDetails(distance=tr_d).data
    )
    client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
    logger.info(f"🚀 EXECUTED {side} {symbol} based on AI Sentiment + Technicals")

# ---------------- MAIN APP ----------------

@app.route("/run")
def run_bot():
    # STEP 1: Get AI Bias (What your friend does)
    usd_bias = get_market_sentiment() 
    trades_taken = []

    for symbol in SYMBOLS:
        m15 = fetch_data(symbol, "M15", 100)
        if m15 is None: continue
        
        curr, prev = m15.iloc[-1], m15.iloc[-2]
        macd_buy = prev["MACD"] < prev["MACD_S"] and curr["MACD"] > curr["MACD_S"]
        macd_sell = prev["MACD"] > prev["MACD_S"] and curr["MACD"] < curr["MACD_S"]

        # STEP 2: Align Technicals with AI Sentiment
        # If USD is BULLISH, we SELL EUR/USD or BUY USD/JPY
        should_buy = (macd_buy and 40 < curr["RSI"] < 60)
        should_sell = (macd_sell and 40 < curr["RSI"] < 60)

        # Logic: Only take the trade if Technicals match the AI's News prediction
        if usd_bias == "BULLISH":
            if symbol in ["USD_JPY"] and should_buy: 
                execute_trade(symbol, "BUY", curr["close"], curr["ATR"])
                trades_taken.append(symbol)
            if symbol in ["EUR_USD", "GBP_USD", "AUD_USD"] and should_sell:
                execute_trade(symbol, "SELL", curr["close"], curr["ATR"])
                trades_taken.append(symbol)
        
        elif usd_bias == "BEARISH":
            if symbol in ["EUR_USD", "GBP_USD", "AUD_USD"] and should_buy:
                execute_trade(symbol, "BUY", curr["close"], curr["ATR"])
                trades_taken.append(symbol)
            if symbol in ["USD_JPY"] and should_sell:
                execute_trade(symbol, "SELL", curr["close"], curr["ATR"])
                trades_taken.append(symbol)

    return jsonify({"sentiment": usd_bias, "executed": trades_taken})

@app.route("/")
def health(): return "AI-Quant Bot: Online & Analyzing News Headlines"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
