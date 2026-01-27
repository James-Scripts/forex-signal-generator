import os, logging, threading, time, numpy as np, requests, json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import defaultdict
from scipy.stats import beta

from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    StopLossDetails,
    TrailingStopLossDetails
)

# ================================
# CONFIGURATION
# ================================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

BASE_RISK = 0.003
DAILY_LOSS_LIMIT = 0.02
NEWS_WINDOW = 60
MIN_COOLDOWN = 600  # 10 Minutes per symbol
HISTORY_FILE = "trade_history.txt"

client = API(access_token=OANDA_API_KEY, environment="practice")

# ================================
# STATE MANAGEMENT
# ================================
ENGINE_LOCK = threading.Lock()
STRATEGY_STATS = defaultdict(lambda: {"wins": 0, "losses": 0})
LAST_TRADE_TIME = {}
LAST_TRANSACTION_ID = 0  # CRITICAL: For deduplication
HIGH_IMPACT_EVENTS = []
SESSION_START_NAV = None
ENGINE_HALTED = False

# ================================
# BAYESIAN BRAIN & DYNAMIC RR
# ================================
def win_prob(key):
    s = STRATEGY_STATS[key]
    return beta(s["wins"] + 1, s["losses"] + 1).mean()

def dynamic_expectancy(p, atr, price):
    """Adjusts Reward/Risk expectations based on Volatility Regime."""
    # High Volatility (ATR > 0.1% of price) allows for higher RR targets
    rr = 1.8 if (atr / price) > 0.001 else 1.3
    return (p * rr) - ((1 - p) * 1)

def update_learning_engine():
    """Corrected: Deduplicated learning using Transaction IDs."""
    global LAST_TRANSACTION_ID
    try:
        r = trades.TradesList(OANDA_ACCOUNT_ID, params={"state": "CLOSED", "count": 20})
        client.request(r)
        
        # Sort trades by ID to process chronologically
        closed_trades = sorted(r.response.get("trades", []), key=lambda x: int(x["id"]))

        for t in closed_trades:
            t_id = int(t["id"])
            
            # 1. Skip if already processed (The Fix)
            if t_id <= LAST_TRANSACTION_ID:
                continue

            pnl = float(t.get("realizedPL", 0))
            symbol = t["instrument"]
            side = "BUY" if int(t["initialUnits"]) > 0 else "SELL"
            key = f"{symbol}_{side}"

            with ENGINE_LOCK:
                if pnl > 0:
                    STRATEGY_STATS[key]["wins"] += 1
                elif pnl < 0:
                    STRATEGY_STATS[key]["losses"] += 1
            
            LAST_TRANSACTION_ID = t_id # Update state
            
            log_msg = f"LEARNED: {key} | ID: {t_id} | PnL: {pnl}"
            logging.info(f"🧠 {log_msg}")
            save_log(log_msg)
            
    except Exception as e:
        logging.error(f"Learning Engine Error: {e}")

# ================================
# EXECUTION LOGIC
# ================================
def save_log(msg):
    with open(HISTORY_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def run_master_cycle(symbol):
    global SESSION_START_NAV, ENGINE_HALTED

    try:
        # 1. Deduplicated Learning
        update_learning_engine()

        # 2. Cooldown Guard
        now_ts = time.time()
        if symbol in LAST_TRADE_TIME and (now_ts - LAST_TRADE_TIME[symbol]) < MIN_COOLDOWN:
            return

        # 3. Account & Kill-Switch
        ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
        nav = float(ra.response["account"]["NAV"])
        if SESSION_START_NAV is None: SESSION_START_NAV = nav
        if (SESSION_START_NAV - nav) / SESSION_START_NAV >= DAILY_LOSS_LIMIT:
            ENGINE_HALTED = True
            logging.error("🛑 KILL-SWITCH: Daily Loss Limit Reached.")
            return

        # 4. ATR Check (No Fallback - Fail Safe)
        atr = compute_atr(symbol)
        if not atr or atr <= 0:
            return

        # 5. Signal Generation (News Filter Optimized)
        side = None
        now_dt = datetime.now(timezone.utc)
        base, quote = symbol.split("_")

        for curr, dt in HIGH_IMPACT_EVENTS:
            # Fund Fix: Focus News trades primarily on USD Quote pairs to reduce noise
            if curr == quote and 0 < (dt - now_dt).total_seconds() <= NEWS_WINDOW:
                r = instruments.InstrumentsCandles(symbol, {"count": 2, "granularity": "M1"})
                client.request(r); c = r.response["candles"][-1]["mid"]
                side = "BUY" if float(c["c"]) > float(c["o"]) else "SELL"
                logging.info(f"🎯 News Hunter Triggered: {symbol}")

        if not side:
            r = instruments.InstrumentsCandles(symbol, {"count": 10, "granularity": "M5"})
            client.request(r); p_list = [float(x["mid"]["c"]) for x in r.response["candles"]]
            side = "BUY" if p_list[-1] > np.mean(p_list) else "SELL"
            current_price = p_list[-1]
        else:
            current_price = float(c["c"])

        # 6. Bayesian Expectancy Check (Dynamic RR)
        p = win_prob(f"{symbol}_{side}")
        e = dynamic_expectancy(p, atr, current_price)

        if e <= 0.05: # Strict Positive Edge Threshold
            return

        # 7. Order Execution
        units = int((nav * BASE_RISK) / (atr * 3))
        if side == "SELL": units = -units
        prec = 3 if "JPY" in symbol else 5

        with ENGINE_LOCK:
            order = MarketOrderRequest(
                instrument=symbol, units=units,
                stopLossOnFill=StopLossDetails(price=str(round(current_price - atr*3 if units > 0 else current_price + atr*3, prec))).data,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * 1.5, prec))).data
            )
            client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
            LAST_TRADE_TIME[symbol] = time.time() # Start cooldown

        save_log(f"OPENED: {symbol} {side} | Units: {units} | E: {e:.2f}")

    except Exception as ex:
        logging.error(f"Cycle Error {symbol}: {ex}")

# (Rest of utility functions: fetch_news_calendar, compute_atr, Main Loop remain similar)
