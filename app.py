import os, logging, threading, requests, time, numpy as np, pandas as pd, csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, StopLossDetails, TakeProfitDetails, TrailingStopLossDetails

# ================= CONFIGURATION =================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_JPY","GBP_JPY"]

BASE_RISK = 0.0025
MAX_DAILY_TRADES = 12
MAX_OPEN_TRADES = 5
DAILY_LOSS_LIMIT = 0.02
COOLDOWN_MINUTES = 8
MAX_SPREAD_PIPS = 3.5

SPIKE_SENSITIVITY = 1.6
HUNTER_WINDOW_SECONDS = 60

JOURNAL_FILE = "trade_journal.csv"
ENGINE_LOCK = threading.Lock()

SESSION_START_NAV = None
LAST_TRADE_TIME = {} 
DAILY_TRADE_COUNT = 0 
ENGINE_HALTED = False
HIGH_IMPACT_EVENTS = []

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-ELITE] | %(message)s")

# ==================== UTILITIES ====================

def compute_atr_true_range(symbol, period=14):
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": period+1, "granularity": "M15"})
        client.request(r)
        candles = r.response["candles"]
        trs = []
        for i in range(1, len(candles)):
            h = float(candles[i]["mid"]["h"])
            l = float(candles[i]["mid"]["l"])
            pc = float(candles[i-1]["mid"]["c"])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(pd.Series(trs).mean())
    except:
        return None

def compute_ema_rsi(symbol):
    try:
        r = instruments.InstrumentsCandles(symbol, {"count": 50, "granularity": "M5"})
        client.request(r)
        closes = pd.Series([float(c['mid']['c']) for c in r.response['candles']])

        ema_fast = closes.ewm(span=9).mean().iloc[-1]
        ema_slow = closes.ewm(span=21).mean().iloc[-1]

        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return ema_fast, ema_slow, rsi.iloc[-1]
    except:
        return None, None, None

def reset_daily_metrics():
    global SESSION_START_NAV, DAILY_TRADE_COUNT, LAST_TRADE_TIME, ENGINE_HALTED
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        SESSION_START_NAV = float(r.response["account"]["NAV"])
        DAILY_TRADE_COUNT, ENGINE_HALTED, LAST_TRADE_TIME = 0, False, {}
        send_telegram(f"🌅 SESSION RESET: NAV @ {SESSION_START_NAV}")
    except:
        logging.error("Reset Failed")

def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=5
        )
    except:
        pass

def log_to_journal(symbol, side, price, atr, mode, units):
    file_exists = os.path.isfile(JOURNAL_FILE)
    try:
        with open(JOURNAL_FILE,'a',newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp","Symbol","Side","Price","ATR","Mode","Units"])
            writer.writerow([datetime.now(timezone.utc),symbol,side,price,round(atr,5),mode,units])
    except:
        pass

# ==================== NEWS / HUNTER ====================

def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=15)
        tree = ET.fromstring(r.content)
        events = []
        for event in tree.findall('event'):
            if event.find('impact').text == 'High':
                curr = event.find('country').text
                dt_str = f"{event.find('date').text} {event.find('time').text}"
                dt = datetime.strptime(dt_str,"%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                events.append((curr, dt))
        HIGH_IMPACT_EVENTS = events
        logging.info(f"News Scraped: {len(HIGH_IMPACT_EVENTS)} high-impact events loaded.")
    except:
        logging.error("News Scrape Failed")

def get_hunter_signal(symbol):
    now = datetime.now(timezone.utc)
    base, quote = symbol.split("_")
    for curr, dt in HIGH_IMPACT_EVENTS:
        if curr in (base, quote):
            diff = (dt - now).total_seconds()
            if 0 < diff <= HUNTER_WINDOW_SECONDS:
                r = instruments.InstrumentsCandles(symbol, {"count": 2, "granularity": "M1"})
                client.request(r)
                c = r.response['candles']
                if float(c[-1]['mid']['c']) > float(c[-1]['mid']['o']):
                    return "BUY"
                if float(c[-1]['mid']['c']) < float(c[-1]['mid']['o']):
                    return "SELL"
    return None

def get_open_positions_count():
    try:
        r = positions.OpenPositions(OANDA_ACCOUNT_ID)
        client.request(r)
        return len(r.response.get("positions", []))
    except:
        return 99

# ==================== EXECUTION ====================

def execute_trade(symbol, side, price, atr, mode_label, nav, drawdown_pct):
    global DAILY_TRADE_COUNT, LAST_TRADE_TIME

    if symbol in LAST_TRADE_TIME:
        if (datetime.now(timezone.utc) - LAST_TRADE_TIME[symbol]).seconds < COOLDOWN_MINUTES * 60:
            logging.info(f"[{symbol}] Cooldown active.")
            return

    try:
        risk_mod = max(0.25, 1 - (drawdown_pct / DAILY_LOSS_LIMIT))
        units = int((nav * (BASE_RISK * risk_mod)) / (atr * 2))
        if units < 1000:
            return

        units = min(units, 100000)
        if side == "SELL":
            units = -units

        prec = 3 if "JPY" in symbol else 5

        sl_mult = 1.8 if mode_label == "NORMAL" else 2.2
        tp_mult = 2.5 if mode_label == "NORMAL" else 3.2
        trail_dist = 1.8 if mode_label == "SPIKE" else 2.2 if mode_label == "HUNTER" else 1.5

        stop = price - atr*sl_mult if side=="BUY" else price + atr*sl_mult
        tp   = price + atr*tp_mult if side=="BUY" else price - atr*tp_mult

        order = MarketOrderRequest(
            instrument=symbol,
            units=units,
            stopLossOnFill=StopLossDetails(price=str(round(stop, prec))).data,
            takeProfitOnFill=TakeProfitDetails(price=str(round(tp, prec))).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * trail_dist, prec))).data
        )

        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))

        log_to_journal(symbol, side, price, atr, mode_label, units)
        LAST_TRADE_TIME[symbol] = datetime.now(timezone.utc)
        DAILY_TRADE_COUNT += 1

        send_telegram(f"🚀 {mode_label} {side}: {symbol} @ {round(price,prec)}")

    except Exception as e:
        logging.error(f"Trade Error: {e}")

# ==================== ENGINE CYCLE ====================

def run_apex_cycle(symbol):
    global ENGINE_HALTED, SESSION_START_NAV

    if SESSION_START_NAV is None:
        reset_daily_metrics()
        if SESSION_START_NAV is None:
            return

    if ENGINE_HALTED:
        return

    with ENGINE_LOCK:
        try:
            ra = accounts.AccountSummary(OANDA_ACCOUNT_ID)
            client.request(ra)
            nav = float(ra.response["account"]["NAV"])
            drawdown = max(0, (SESSION_START_NAV - nav) / SESSION_START_NAV)

            if drawdown >= DAILY_LOSS_LIMIT:
                ENGINE_HALTED = True
                send_telegram("🛑 DAILY LOSS LIMIT HIT. ENGINE HALTED.")
                return

            if DAILY_TRADE_COUNT >= MAX_DAILY_TRADES:
                return

            open_count = get_open_positions_count()
            if open_count >= MAX_OPEN_TRADES:
                return

            atr = compute_atr_true_range(symbol)
            r = instruments.InstrumentsCandles(symbol, {"count": 20, "granularity": "M5"})
            client.request(r)
            data = [float(c['mid']['c']) for c in r.response['candles']]
            price = data[-1]

            if not atr:
                return

            # HUNTER MODE
            h_side = get_hunter_signal(symbol)
            if h_side:
                execute_trade(symbol, h_side, price, atr, "HUNTER", nav, drawdown)
                return

            # SPIKE MODE
            move = abs(data[-1] - data[-2])
            avg_move = np.mean([abs(data[i] - data[i-1]) for i in range(-6, -1)])
            threshold = max(atr * 0.8, avg_move * 2)

            if move > threshold:
                side = "BUY" if data[-1] > data[-2] else "SELL"
                execute_trade(symbol, side, price, atr, "SPIKE", nav, drawdown)
                return

            # NORMAL MODE
            ema_fast, ema_slow, rsi = compute_ema_rsi(symbol)
            if ema_fast and ema_slow:
                if ema_fast > ema_slow and 45 < rsi < 70:
                    execute_trade(symbol, "BUY", price, atr, "NORMAL", nav, drawdown)
                    return

                if ema_fast < ema_slow and 30 < rsi < 55:
                    execute_trade(symbol, "SELL", price, atr, "NORMAL", nav, drawdown)
                    return

        except Exception as e:
            logging.error(f"Cycle Error {symbol}: {e}")

# ==================== SERVER ====================

scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scrape_high_impact_news, 'interval', minutes=30)
scheduler.add_job(reset_daily_metrics, 'cron', hour=0, minute=0)
scheduler.start()

@app.route('/status')
def status():
    return jsonify({
        "engine": "APEX-ELITE v1.2",
        "halted": ENGINE_HALTED,
        "trades": DAILY_TRADE_COUNT,
        "nav": SESSION_START_NAV,
        "news_events": len(HIGH_IMPACT_EVENTS)
    }), 200

@app.route('/run')
def trigger():
    threading.Thread(target=lambda: [run_apex_cycle(s) for s in SYMBOLS]).start()
    return "Scanning Symbols...", 200

if __name__ == "__main__":
    logging.info("Initializing APEX-ELITE...")
    reset_daily_metrics()
    scrape_high_impact_news()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
