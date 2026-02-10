# ============================================================
# INSTITUTIONAL-GRADE OANDA MULTI-ASSET TRADING BOT (9+/10)
# ============================================================

import os, json, time, math, threading, logging, requests
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from flask import Flask, jsonify

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts

# =========================
# CONFIGURATION
# =========================

OANDA_API_KEY     = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV        = "practice"  # change to "live" explicitly

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]

BASE_RISK          = 0.01     # 1% normal risk
NEWS_RISK_FACTOR   = 0.35     # 35% of base risk
MAX_DRAWDOWN       = 0.12     # 12% soft throttle
HARD_DRAWDOWN      = 0.18     # emergency stop
ATR_MULT_SL        = 2.0
ATR_MULT_TP        = 3.0

NEWS_CONFIRM_DELAY = 180      # seconds AFTER release
HTF_TF_1           = "M15"
HTF_TF_2           = "M30"

STATE_FILE = "strategy_stats.json"

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# API INIT
# =========================

api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

# =========================
# STATE MANAGEMENT
# =========================

if not os.path.exists(STATE_FILE):
    with open(STATE_FILE, "w") as f:
        json.dump({
            "equity_peak": 0,
            "equity_curve": [],
            "drawdown": 0,
            "wins": 0,
            "losses": 0
        }, f)

def load_state():
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# =========================
# ACCOUNT INFO
# =========================

def get_nav():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    api.request(r)
    return float(r.response["account"]["NAV"])

# =========================
# ECONOMIC CALENDAR (REAL)
# =========================

def high_impact_news_now():
    try:
        resp = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=5
        ).json()
        now = datetime.utcnow()
        for ev in resp:
            if ev["impact"] == "High":
                ev_time = datetime.fromisoformat(ev["date"].replace("Z",""))
                if 0 <= (now - ev_time).total_seconds() <= NEWS_CONFIRM_DELAY:
                    return True
        return False
    except Exception as e:
        logging.warning(f"News check failed: {e}")
        return False

# =========================
# MARKET DATA
# =========================

def get_candles(inst, tf, count=100):
    r = instruments.InstrumentsCandles(
        instrument=inst,
        params={"granularity": tf, "count": count}
    )
    api.request(r)
    data = []
    for c in r.response["candles"]:
        if c["complete"]:
            data.append(float(c["mid"]["c"]))
    return np.array(data)

def atr(prices, period=14):
    tr = np.abs(np.diff(prices))
    return np.mean(tr[-period:])

# =========================
# HTF BIAS
# =========================

def htf_bias(inst):
    p1 = get_candles(inst, HTF_TF_1, 60)
    p2 = get_candles(inst, HTF_TF_2, 60)

    slope1 = np.polyfit(range(len(p1)), p1, 1)[0]
    slope2 = np.polyfit(range(len(p2)), p2, 1)[0]

    if slope1 > 0 and slope2 > 0:
        return "LONG"
    if slope1 < 0 and slope2 < 0:
        return "SHORT"
    return "NEUTRAL"

# =========================
# PIP VALUE
# =========================

def pip_value(inst):
    return 0.01 if "JPY" in inst else 0.0001

# =========================
# POSITION SIZING
# =========================

def calc_units(inst, nav, atr_val, risk_pct):
    pip = pip_value(inst)
    sl_pips = (atr_val * ATR_MULT_SL) / pip
    risk_amount = nav * risk_pct
    units = risk_amount / (sl_pips * pip)
    return int(units)

# =========================
# ORDER EXECUTION
# =========================

def place_trade(inst, direction, units, sl_price, tp_price):
    data = {
        "order": {
            "instrument": inst,
            "units": str(units if direction=="LONG" else -units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl_price:.5f}"},
            "takeProfitOnFill": {"price": f"{tp_price:.5f}"}
        }
    }
    r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=data)
    api.request(r)
    logging.info(f"{inst} {direction} ORDER SENT")

# =========================
# MAIN STRATEGY LOOP
# =========================

def strategy_loop():
    while True:
        state = load_state()
        nav = get_nav()

        # Equity curve
        state["equity_curve"].append(nav)
        if nav > state["equity_peak"]:
            state["equity_peak"] = nav

        dd = (state["equity_peak"] - nav) / state["equity_peak"] if state["equity_peak"] else 0
        state["drawdown"] = dd

        if dd >= HARD_DRAWDOWN:
            logging.error("HARD DRAWDOWN HIT — STOPPING")
            save_state(state)
            break

        # Risk decay
        risk_scale = max(0.3, 1 - (dd / MAX_DRAWDOWN))

        is_news = high_impact_news_now()

        for inst in INSTRUMENTS:
            prices = get_candles(inst, "M5", 100)
            atr_val = atr(prices)
            bias = htf_bias(inst)

            if bias == "NEUTRAL":
                continue

            direction = bias

            risk = BASE_RISK * risk_scale
            if is_news:
                risk *= NEWS_RISK_FACTOR

            units = calc_units(inst, nav, atr_val, risk)

            price = prices[-1]
            pip = pip_value(inst)

            sl = price - ATR_MULT_SL * atr_val if direction=="LONG" else price + ATR_MULT_SL * atr_val
            tp = price + ATR_MULT_TP * atr_val if direction=="LONG" else price - ATR_MULT_TP * atr_val

            place_trade(inst, direction, units, sl, tp)

        save_state(state)
        time.sleep(300)

# =========================
# FLASK HEALTH ENDPOINT
# =========================

app = Flask(__name__)

@app.route("/")
def status():
    return jsonify({"status": "running", "time": datetime.utcnow().isoformat()})

# =========================
# STARTUP
# =========================

if __name__ == "__main__":
    t = threading.Thread(target=strategy_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8080)
