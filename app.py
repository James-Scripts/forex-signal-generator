import os, logging, threading, time, numpy as np, requests, json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import defaultdict
from scipy.stats import beta
from flask import Flask, jsonify

from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    StopLossDetails,
    TrailingStopLossDetails,
    ClientExtensions
)

# ================================
# CONFIGURATION
# ================================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

BASE_RISK = 0.003
DAILY_LOSS_LIMIT = 0.5
NEWS_WINDOW = 60
MIN_COOLDOWN = 300  # Lowered to 5 mins for better activity
STATS_FILE = "strategy_stats.json"
HISTORY_FILE = "trade_history.txt"

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)

# ================================
# PERSISTENT STATE MANAGEMENT
# ================================
ENGINE_LOCK = threading.Lock()
STRATEGY_STATS = defaultdict(lambda: {"wins": 0, "losses": 0})
LAST_TRADE_TIME = {}
LAST_TRANSACTION_ID = 0  
HIGH_IMPACT_EVENTS = []
SESSION_START_NAV = None
ENGINE_HALTED = False

def load_memory():
    global STRATEGY_STATS, LAST_TRANSACTION_ID
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                data = json.load(f)
                STRATEGY_STATS.update(data.get("stats", {}))
                LAST_TRANSACTION_ID = data.get("last_id", 0)
            logging.info("🧠 Brain loaded from disk.")
        except Exception as e:
            logging.error(f"Failed to load brain: {e}")

def save_memory():
    with ENGINE_LOCK:
        try:
            with open(STATS_FILE, "w") as f:
                json.dump({"stats": STRATEGY_STATS, "last_id": LAST_TRANSACTION_ID}, f)
        except Exception as e:
            logging.error(f"Failed to save brain: {e}")

# ================================
# BAYESIAN & EXPECTANCY
# ================================
def win_prob(key):
    s = STRATEGY_STATS[key]
    # Optimistic prior (2,2) to encourage exploration on new strategies
    return beta(s["wins"] + 2, s["losses"] + 2).mean()

def dynamic_expectancy(p, atr, price):
    # Tuned RR: News trades often have higher momentum/RR potential
    rr = 1.8 if (atr / price) > 0.0009 else 1.3
    return (p * rr) - ((1 - p) * 1)

# ================================
# UTILITIES
# ================================
def save_log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(HISTORY_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def compute_atr(symbol, n=10): # Shorter window for higher sensitivity
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": n + 1, "granularity": "M5"})
        client.request(r)
        c = r.response["candles"]
        trs = []
        for i in range(1, len(c)):
            h, l, pc = float(c[i]["mid"]["h"]), float(c[i]["mid"]["l"]), float(c[i-1]["mid"]["c"])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return np.mean(trs)
    except: return None

def fetch_news_calendar():
    global HIGH_IMPACT_EVENTS
    try:
        res = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=10)
        root = ET.fromstring(res.content)
        events = []
        for e in root.findall("event"):
            if e.find("impact").text == "High":
                dt_str = f"{e.find('date').text} {e.find('time').text}"
                dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                events.append((e.find("country").text, dt))
        HIGH_IMPACT_EVENTS = events
    except Exception as e:
        logging.error(f"News failure: {e}")

# ================================
# EXECUTION ENGINE
# ================================
def update_learning_engine():
    global LAST_TRANSACTION_ID
    try:
        r = trades.TradesList(OANDA_ACCOUNT_ID, params={"state": "CLOSED", "count": 20})
        client.request(r)
        closed = sorted(r.response.get("trades", []), key=lambda x: int(x["id"]))

        new_data = False
        for t in closed:
            t_id = int(t["id"])
            if t_id <= LAST_TRANSACTION_ID: continue

            pnl = float(t.get("realizedPL", 0))
            side = "BUY" if int(t["initialUnits"]) > 0 else "SELL"
            tag = t.get("clientExtensions", {}).get("tag", "MEAN")
            key = f"{t['instrument']}_{side}_{tag}"

            with ENGINE_LOCK:
                if pnl > 0: STRATEGY_STATS[key]["wins"] += 1
                elif pnl < 0: STRATEGY_STATS[key]["losses"] += 1
            
            LAST_TRANSACTION_ID = t_id
            new_data = True
            save_log(f"RESULT: {key} | PnL: {pnl}")
        
        if new_data: save_memory()
    except: pass

def run_master_cycle(symbol):
    global SESSION_START_NAV, ENGINE_HALTED
    try:
        update_learning_engine()
        
        if symbol in LAST_TRADE_TIME and (time.time() - LAST_TRADE_TIME[symbol]) < MIN_COOLDOWN: return

        ra = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(ra)
        nav = float(ra.response["account"]["NAV"])
        if SESSION_START_NAV is None: SESSION_START_NAV = nav
        if (SESSION_START_NAV - nav) / SESSION_START_NAV >= DAILY_LOSS_LIMIT:
            ENGINE_HALTED = True; return

        atr = compute_atr(symbol)
        if not atr: return

        # Regime Selection
        side, logic_tag = None, "MEAN"
        now = datetime.now(timezone.utc)
        base, quote = symbol.split("_")

        for curr, dt in HIGH_IMPACT_EVENTS:
            if curr in [base, quote] and 0 < (dt - now).total_seconds() <= NEWS_WINDOW:
                r = instruments.InstrumentsCandles(symbol, {"count": 2, "granularity": "M1"})
                client.request(r); c = r.response["candles"][-1]["mid"]
                side = "BUY" if float(c["c"]) > float(c["o"]) else "SELL"
                price, logic_tag = float(c["c"]), "NEWS"
                break

        if not side:
            r = instruments.InstrumentsCandles(symbol, {"count": 10, "granularity": "M5"})
            client.request(r); p_list = [float(x["mid"]["c"]) for x in r.response["candles"]]
            side = "BUY" if p_list[-1] > np.mean(p_list) else "SELL"
            price = p_list[-1]

        # Bayesian Logic
        key = f"{symbol}_{side}_{logic_tag}"
        p = win_prob(key)
        e = dynamic_expectancy(p, atr, price)

        # TUNED: e > 0.015 for higher frequency
        if e > 0.015:
            units = int((nav * BASE_RISK) / (atr * 2))
            if side == "SELL": units = -units
            prec = 3 if "JPY" in symbol else 5
            
            # SL adjusted to 2.5 * ATR for better breathing room
            sl_price = price - atr*2.5 if units > 0 else price + atr*2.5
            
            ext = ClientExtensions(clientTag=logic_tag, comment="ApexV4").data
            order = MarketOrderRequest(
                instrument=symbol, units=units,
                stopLossOnFill=StopLossDetails(price=str(round(sl_price, prec))).data,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * 2.0, prec))).data,
                clientExtensions=ext
            )

            client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
            LAST_TRADE_TIME[symbol] = time.time()
            save_log(f"OPENED: {key} | E: {e:.2f} | P: {p:.2f}")

    except Exception as ex:
        logging.error(f"Cycle Error [{symbol}]: {ex}")

# ================================
# FLASK & THREADING
# ================================
@app.route('/')
def health():
    return jsonify({"status": "RUNNING", "halted": ENGINE_HALTED, "last_id": LAST_TRANSACTION_ID, "stats": STRATEGY_STATS})

def trading_thread():
    load_memory()
    fetch_news_calendar()
    while not ENGINE_HALTED:
        for s in SYMBOLS:
            run_master_cycle(s)
            time.sleep(1)
        time.sleep(15)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    threading.Thread(target=trading_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
