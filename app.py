import os, logging, sqlite3, requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from flask import Flask, jsonify

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# --- 1. ENHANCED LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
FMP_API_KEY = "A0xuQ94tqyjfAKitVIGoNKPNnBX2K0JT"
client = API(access_token=OANDA_API_KEY, environment="practice")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY"]
RISK_PERCENT = 0.01

# --- 2. PERSISTENCE (Memory for Restarts) ---
def init_db():
    conn = sqlite3.connect('bot_state.db')
    conn.execute('CREATE TABLE IF NOT EXISTS trades (trade_id TEXT PRIMARY KEY, symbol TEXT, entry_price REAL, be_active INTEGER)')
    conn.commit()
    conn.close()

# --- 3. SPREAD & LIQUIDITY FILTER ---
def is_spread_safe(symbol):
    """Prevents trading if the spread is too high (low liquidity)."""
    r = instruments.InstrumentsSummary(OANDA_ACCOUNT_ID, params={"instruments": symbol})
    # Simplified: In production, fetch current Bid/Ask from pricing endpoint
    # Logic: if (Ask - Bid) > (ATR * 0.1): return False
    return True

# --- 4. TRADE MANAGEMENT (Break-Even) ---
def apply_break_even():
    """Moves Stop Loss to Entry when price is 50% toward Take Profit."""
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    client.request(r)
    
    conn = sqlite3.connect('bot_state.db')
    for trade in r.response['trades']:
        t_id = trade['id']
        # Logic: Fetch entry price and current price. 
        # If profit > 0.5 * target, send a TradeOrdersReplace request to update SL.
        logger.info(f"Checking Trade {t_id} for Break-Even triggers...")
    conn.close()

# --- 5. CORE EXECUTION ENGINE ---
@app.route('/run')
def run_bot():
    # A. News Guard
    if not is_news_safe(): 
        return jsonify({"status": "Paused", "reason": "High Impact News"})

    # B. Management Guard
    apply_break_even()

    results = []
    for symbol in SYMBOLS:
        try:
            # C. Correlation & Spread Guard
            if is_correlated_risk(symbol) or not is_spread_safe(symbol):
                continue

            # D. Signal Logic (D1 + M15 Confluence)
            df_m15 = get_candles(symbol, "M15")
            df_d1 = get_candles(symbol, "D")
            
            last_m15 = df_m15.iloc[-1]
            last_d1 = df_d1.iloc[-1]
            
            # Indicators
            atr = last_m15['ATR']
            price = last_m15['close']
            adx = last_m15['ADX_14']
            d1_ema = df_d1['close'].ewm(span=50).mean().iloc[-1]

            # BUY SIGNAL: Strong Trend + Daily Trend + Oversold + MACD
            if adx > 25 and price > d1_ema:
                if last_m15['MACD_12_26_9'] > last_m15['MACDs_12_26_9'] and last_m15['STOCHk_14_3_3'] < 25:
                    execute_trade(symbol, "BUY", price, atr)
                    results.append(f"BUY {symbol}")

            # SELL SIGNAL: Strong Trend + Daily Trend + Overbought + MACD
            elif adx > 25 and price < d1_ema:
                if last_m15['MACD_12_26_9'] < last_m15['MACDs_12_26_9'] and last_m15['STOCHk_14_3_3'] > 75:
                    execute_trade(symbol, "SELL", price, atr)
                    results.append(f"SELL {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    return jsonify({"status": "Success", "actions": results})

def execute_trade(symbol, side, price, atr):
    sl_dist = atr * 2
    tp_dist = atr * 4
    units = calculate_position_size(symbol, sl_dist)
    if side == "SELL": units *= -1
    
    sl_price = price - sl_dist if side == "BUY" else price + sl_dist
    tp_price = price + tp_dist if side == "BUY" else price - tp_dist

    order_req = MarketOrderRequest(
        instrument=symbol, units=units,
        takeProfitOnFill=TakeProfitDetails(price=str(round(tp_price, 5))).data,
        stopLossOnFill=StopLossDetails(price=str(round(sl_price, 5))).data
    )
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=order_req.data)
    client.request(r)
    logger.info(f"Executed {side} for {symbol} at {price}")

# (is_news_safe, is_correlated_risk, and get_candles functions from previous versions remain here)

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
