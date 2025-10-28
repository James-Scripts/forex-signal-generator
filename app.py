import requests
import pandas as pd
from ta.trend import MACD 
from ta.momentum import StochasticOscillator 
from datetime import datetime, timedelta
import time
import json
from itertools import product
from typing import Dict, List, Any, Optional
from io import StringIO
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import traceback 
from threading import Thread 
# 💡 Flask Imports
from flask import Flask, jsonify, request
# --- Flask Application Setup ---
app = Flask(__name__)
# --- API Keys Configuration ---
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY")
# --- EMAIL Configuration ---
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
MAIL_SERVER = os.environ.get("MAIL_SERVER", 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get("MAIL_PORT", 465))
MAIL_SENDER = os.environ.get("MAIL_SENDER")
MAIL_RECIPIENT = os.environ.get("MAIL_RECIPIENT")
# --- 0. Email Utility Function ---
def send_email_notification(subject: str, body: str, recipient: str):
    """Sends an email using the configured SMTP settings.""" # <-- INDENTATION FIXED
    if not all([MAIL_USERNAME, MAIL_PASSWORD, MAIL_SENDER, recipient]):
        print("ERROR: Email configuration missing. Skipping email notification.")
        return
    msg = MIMEMultipart()
    msg['From'] = MAIL_SENDER
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    print(f"DEBUG: Attempting to send email to {recipient}...")
    try:
        # Use SSL connection (port 465)
        with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as server:
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_SENDER, recipient, msg.as_string())
        print("DEBUG: Email notification sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("FATAL EMAIL ERROR: SMTP Authentication failed. Check your username and App Password.")
    except Exception as e:
        print(f"FATAL EMAIL ERROR: Failed to send email: {e}")
# --- 1. Data Feed (Alpha Vantage) (Updated for robust data validation) ---
class DataFeed:
    """Base class for fetching both historical and real-time market data."""
    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        raise NotImplementedError
    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        raise NotImplementedError
class AlphaVantageDataFeed(DataFeed):
    BASE_URL = "https://www.alphavantage.co/query"
    TIME_INTERVAL = "60min"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_cache = {}
    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        """
        Fetches up to 2 years of 60min historical Forex data for EUR/USD.

        Now includes robust checks to prevent 'Length mismatch' errors.
        """

        from_symbol, to_symbol = symbol[:3], symbol[4:]

        if symbol in self.data_cache:
            print(f"DEBUG: Historical Data for {symbol} loaded from cache.")
            return self.data_cache[symbol].copy()

        print(f"DEBUG: Fetching historical {interval} data for {symbol} (up to 2 years)...")

        slices = [f"year{y}month{m}" for y in [1, 2] for m in range(1, 13)]
        all_data = []

        for i, slice_name in enumerate(slices):
            if i > 0:
                print(f"DEBUG: Pausing for 15 seconds after slice {i+1} to respect Alpha Vantage rate limit...")
                time.sleep(15)
            params = {
                "function": "FX_INTRADAY_EXTENDED",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "interval": interval,
                "slice": slice_name,
                "outputsize": "full",
                "apikey": self.api_key,
                "datatype": "csv"
            }

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                if 'Error Message' in response.text or 'Invalid API call' in response.text:
                    print(f"Alpha Vantage API Error for {slice_name}: {response.text.strip()}")
                    continue
                df_slice = pd.read_csv(StringIO(response.text))

                # 💡 FIX 1: Crucial check: only append if it looks like correct data (must have 6 columns)
                if len(df_slice.columns) == 6:
                    all_data.append(df_slice)
                    print(f"DEBUG: Fetched slice: {slice_name} ({len(df_slice)} rows)")
                else:
                    print(f"WARNING: Slice {slice_name} was filtered out due to unexpected column count ({len(df_slice.columns)} instead of 6).")
            except requests.exceptions.RequestException as e:
                print(f"API request failed for slice {slice_name}: {e}")
                time.sleep(60)
                continue
            except Exception as e:
                print(f"Failed to process slice {slice_name}: {e}")
                continue
        if not all_data:
            print("ERROR: Failed to fetch any valid historical data slices.")
            return pd.DataFrame()
        full_df = pd.concat(all_data, ignore_index=True)

        # 💡 FIX 2: Check column count on final concatenated DataFrame before assigning names
        if len(full_df.columns) != 6:
            print(f"FATAL ERROR: Final concatenated DataFrame has {len(full_df.columns)} columns, expected 6. Returning empty DataFrame.")
            return pd.DataFrame()
        full_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        full_df.set_index('timestamp', inplace=True)
        full_df.sort_index(inplace=True)
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        full_df.dropna(inplace=True)
        self.data_cache[symbol] = full_df.drop(columns=['volume'])
        return self.data_cache[symbol].copy()
    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        """
        Fetches the latest intraday bar data.
        """
        from_symbol, to_symbol = symbol[:3], symbol[4:]

        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "interval": interval,
            "outputsize": "compact",
            "apikey": self.api_key
        }
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            time_series_key = f"Time Series FX ({interval})"

            if time_series_key not in data:
                error_msg = data.get('Error Message', 'Unknown error structure in response.')
                if "Thank you for using Alpha Vantage" in str(data):
                    print("Alpha Vantage rate limit hit. Pausing and returning empty series.")
                    time.sleep(60)
                print(f"ERROR fetching real-time bar: {error_msg}")
                return pd.Series()

            latest_time = list(data[time_series_key].keys())[0]
            bar_data = data[time_series_key][latest_time]
            latest_bar = pd.Series({
                'open': float(bar_data['1. open']),
                'high': float(bar_data['2. high']),
                'low': float(bar_data['3. low']),
                'close': float(bar_data['4. close'])
            }, name=pd.to_datetime(latest_time))

            return latest_bar
        except requests.exceptions.RequestException as e:
            print(f"Real-time bar request failed: {e}")
            return pd.Series()
        except Exception as e:
            print(f"Failed to process real-time data: {e}")
            return pd.Series()
# --- 2. Fundamental Processor (MarketAux) ---
class FundamentalProcessor:
    """Base class for fetching fundamental/sentiment data."""
    def fetch_realtime_sentiment(self) -> float:
        raise NotImplementedError
class MarketAuxProcessor(FundamentalProcessor):
    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: str):
        self.api_key = api_key
    def fetch_realtime_sentiment(self) -> float:
        """Fetches the latest financial news sentiment relevant to EUR/USD."""

        params = {
            "api_token": self.api_key,
            "search": "euro OR dollar OR eur/usd",
            "filter_entities": "true",
            "language": "en",
            "limit": 10
        }
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get('data', [])

            if not articles:
                print("MarketAux: No news articles found for search query.")
                return 0.0
            all_sentiments = []

            for article in articles:
                entities = article.get('entities', [])
                for entity in entities:
                    if entity.get('symbol') in ['EUR', 'USD'] and 'sentiment_score' in entity:
                        score = entity['sentiment_score']
                        # Weight EUR sentiment positively, USD sentiment negatively
                        if entity['symbol'] == 'EUR':
                            all_sentiments.append(score)
                        elif entity['symbol'] == 'USD':
                            all_sentiments.append(-score) # Negative weight for USD sentiment
            if not all_sentiments:
                # Fallback to article-level sentiment if entity-level is missing
                article_sentiments = [
                    a.get('sentiment_score', 0.0) 
                    for a in articles if 'sentiment_score' in a
                ]
                if article_sentiments:
                    print("MarketAux: Using article-level sentiment (treating as EUR/USD composite).")
                    return sum(article_sentiments) / len(article_sentiments)
                else:
                    return 0.0
            avg_sentiment = sum(all_sentiments) / len(all_sentiments)

            return avg_sentiment
        except requests.exceptions.RequestException as e:
            print(f"MarketAux request failed: {e}")
            return 0.0
        except Exception as e:
            print(f"Failed to process MarketAux data: {e}")
            return 0.0
# --- 3. Trading Strategy and Signal Generation ---
class Backtester:
    """
    Implements the backtesting of the MACD + Stochastic + Fundamental strategy with 
    fixed Take Profit (TP) and Stop Loss (SL).
    """
    def __init__(self, data: pd.DataFrame, fundamental_threshold: float, sl_pips: float, tp_pips: float):
        self.data = data.copy()
        self.fundamental_threshold = fundamental_threshold
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.PIP_CONVERSION = 10000.0 # Standard for EUR/USD (4th decimal place, 5th is a "point")
    def calculate_technical_indicators(self):
        """Calculates MACD and Stochastic Oscillator using the ta library."""
        print("DEBUG: Calculating technical indicators (MACD, Stochastic)...")
        # MACD (12, 26, 9) using ta.trend.MACD
        macd_instance = MACD(close=self.data['close'], window_fast=12, window_slow=26, window_sign=9, fillna=False) 
        # Stochastic Oscillator (14, 3, 3) using ta.momentum.StochasticOscillator
        stoch_instance = StochasticOscillator(high=self.data['high'], low=self.data['low'], close=self.data['close'], window=14, smooth_window=3, fillna=False) 
        
        self.data['MACD_Hist'] = macd_instance.macd_diff() 
        self.data['Stoch_K'] = stoch_instance.stoch() 
        self.data['Stoch_D'] = stoch_instance.stoch_signal() 
        
        self.data.dropna(inplace=True)
    def generate_historical_signals(self) -> pd.DataFrame:
        """
        Generates buy/sell/hold signals based on technical and simulated fundamental data.
        """
        # SIMULATION: Create a synthetic historical sentiment series.
        np.random.seed(42)
        price_diff = self.data['close'].diff().fillna(0)
        simulated_sentiment = np.clip(price_diff.rolling(window=10).mean().fillna(0) * 5 + np.random.normal(0, 0.1, len(self.data)), -1, 1)
        self.data['sentiment'] = simulated_sentiment

        signals = pd.Series(index=self.data.index, dtype=str)

        # MACD Crossover (Primary Trend) - requires shifting for crossover logic
        macd_bullish = self.data['MACD_Hist'] > 0
        macd_bearish = self.data['MACD_Hist'] < 0

        # Stochastic Crossover (Momentum Confirmation/Trigger)
        stoch_k_prev = self.data['Stoch_K'].shift(1)
        stoch_d_prev = self.data['Stoch_D'].shift(1)

        stoch_bullish_cross = (stoch_k_prev < stoch_d_prev) & (self.data['Stoch_K'] > self.data['Stoch_D']) & (self.data['Stoch_D'] < 50)
        stoch_bearish_cross = (stoch_k_prev > stoch_d_prev) & (self.data['Stoch_K'] < self.data['Stoch_D']) & (self.data['Stoch_D'] > 50)

        # Fundamental Filter
        sentiment_positive = self.data['sentiment'] >= self.fundamental_threshold
        sentiment_negative = self.data['sentiment'] <= -self.fundamental_threshold

        # Combined Signal
        buy_condition = macd_bullish & stoch_bullish_cross & sentiment_positive
        sell_condition = macd_bearish & stoch_bearish_cross & sentiment_negative

        signals[buy_condition] = "BUY"
        signals[sell_condition] = "SELL"
        signals.fillna("HOLD", inplace=True)

        self.data['signal'] = signals
        return self.data

    def run_backtest(self) -> Dict[str, Any]:
        """
        Executes the backtest using the fixed TP/SL logic and calculates performance metrics.
        """
        self.calculate_technical_indicators()
        self.generate_historical_signals()

        if self.data.empty or 'signal' not in self.data.columns:
            return {"net_pips": 0, "total_trades": 0, "profit_factor": 0.0, "reason": "Data or signal generation failed."}
        trades_list = []
        in_trade = False

        # Iterate through data to find signals and simulate exits
        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]

            # --- Trade Entry Logic ---
            if not in_trade and current_bar['signal'] in ['BUY', 'SELL']:
                entry_time = current_bar.name
                entry_price = current_bar['close'] # Entry at the close of the signal bar
                direction = current_bar['signal']
                in_trade = True

                # Calculate SL/TP price levels
                if direction == 'BUY':
                    sl_price = entry_price - (self.sl_pips / self.PIP_CONVERSION)
                    tp_price = entry_price + (self.tp_pips / self.PIP_CONVERSION)
                else: # SELL
                    sl_price = entry_price + (self.sl_pips / self.PIP_CONVERSION)
                    tp_price = entry_price - (self.tp_pips / self.PIP_CONVERSION)
                trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'direction': direction,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'exit_time': None,
                    'profit_pips': 0.0,
                    'exit_type': ''
                }

            # --- Trade Exit Logic (Simulate next bars) ---
            if in_trade and current_bar.name > trade['entry_time']:

                exit_logic_triggered = False
                pips_gained = 0.0
                exit_type = ''

                # Check for exit on the current bar's OHLC
                if trade['direction'] == 'BUY':
                    if current_bar['low'] <= trade['sl_price']:
                        exit_logic_triggered = True
                        pips_gained = -self.sl_pips # Loss
                        exit_type = 'SL'
                    elif current_bar['high'] >= trade['tp_price']:
                        exit_logic_triggered = True
                        pips_gained = self.tp_pips # Win
                        exit_type = 'TP'

                elif trade['direction'] == 'SELL':
                    if current_bar['high'] >= trade['sl_price']:
                        exit_logic_triggered = True
                        pips_gained = -self.sl_pips # Loss
                        exit_type = 'SL'
                    elif current_bar['low'] <= trade['tp_price']:
                        exit_logic_triggered = True
                        pips_gained = self.tp_pips # Win
                        exit_type = 'TP'

                if exit_logic_triggered:
                    trade['exit_time'] = current_bar.name
                    trade['profit_pips'] = pips_gained
                    trade['exit_type'] = exit_type
                    trades_list.append(trade)
                    in_trade = False # Reset for the next signal
        # --- Performance Calculation ---
        trades_df = pd.DataFrame(trades_list)
        total_trades = len(trades_df)

        if total_trades == 0:
            return {"net_pips": 0, "total_trades": 0, "profit_factor": 0.0, "reason": "No completed trades generated."}
        total_profit = trades_df[trades_df['profit_pips'] > 0]['profit_pips'].sum()
        total_loss = trades_df[trades_df['profit_pips'] < 0]['profit_pips'].sum()
        net_pips = total_profit + total_loss

        # Avoid division by zero
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else np.inf
        return {
            "net_pips": round(net_pips, 2),
            "total_trades": total_trades,
            "winning_trades": len(trades_df[trades_df['profit_pips'] > 0]),
            "profit_factor": round(profit_factor, 2),
            "reason": f"Backtested over {len(self.data)} bars. SL: {self.sl_pips} pips, TP: {self.tp_pips} pips (R:R 1:{self.tp_pips/self.sl_pips})."
        }
class SignalGenerator:
    # --- Strategy Tuning Parameters ---
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    STOCH_K = 14
    STOCH_D = 3
    FUNDAMENTAL_THRESHOLD = 0.3

    # Risk Management Parameters (25 pips SL, 37.5 pips TP for 1:1.5 R:R)
    STOP_LOSS_PIPS = 25.0
    RISK_REWARD_RATIO = 1.5
    TAKE_PROFIT_PIPS = STOP_LOSS_PIPS * RISK_REWARD_RATIO
    PIP_CONVERSION = 10000.0 # Standard for EUR/USD
    def __init__(self, data_feed: DataFeed, fundamental_processor: FundamentalProcessor):
        self.data_feed = data_feed
        self.fundamental_processor = fundamental_processor
        self.symbol = "EUR/USD"
        self.interval = "60min"
        self.lookback_years = 2
        self.data: Optional[pd.DataFrame] = None
    def initialize_data(self):
        """Fetches historical data to build the model's foundation."""
        print(f"\n--- Initializing Historical Data for {self.symbol} ---")

        if not self.data_feed.api_key:
            print("ERROR: Alpha Vantage API Key is missing. Cannot fetch data.")
            self.data = pd.DataFrame()
            return
        self.data = self.data_feed.fetch_historical_data(self.symbol, self.interval, self.lookback_years)

        if self.data is not None and not self.data.empty:
            print(f"DEBUG: Historical Data loaded. Total bars: {len(self.data)}")

            # Calculate initial technical indicators on the historical data
            macd_instance = MACD(close=self.data['close'], window_fast=self.MACD_FAST, window_slow=self.MACD_SLOW, window_sign=self.MACD_SIGNAL, fillna=False) 
            stoch_instance = StochasticOscillator(high=self.data['high'], low=self.data['low'], close=self.data['close'], window=self.STOCH_K, smooth_window=self.STOCH_D, fillna=False) 
            
            self.data['MACD_Hist'] = macd_instance.macd_diff() 
            self.data['Stoch_K'] = stoch_instance.stoch() 
            self.data['Stoch_D'] = stoch_instance.stoch_signal() 
            
            # --- Robust Column Check ---
            if not all(col in self.data.columns for col in ['MACD_Hist', 'Stoch_K', 'Stoch_D']): 
                print("FATAL: One or more technical indicator columns were not created successfully. Invalidate data.")
                self.data = pd.DataFrame()
                return
            
            self.data.dropna(inplace=True)
            print(f"DEBUG: Historical Data ready. Bars after cleanup: {len(self.data)}")

        else:
            print("ERROR: Initialization failed. Cannot proceed with historical data.")
    def generate_signal(self) -> Dict[str, Any]:
        """
        Generates a trade signal based on the latest bar, technical indicators, and fundamental score, 
        calculating precise entry/exit prices.
        """
        if self.data is None or self.data.empty:
            print("ERROR: Historical data not initialized. Cannot generate signal.")
            return {"signal": "HOLD", "reason": "Historical data missing."}
        # 1. Fetch Latest Bar
        print("\n--- Running Step 1: Fetching Real-Time Market Data ---")
        new_bar = self.data_feed.fetch_realtime_bar(self.symbol, self.interval)

        if new_bar.empty:
            return {"signal": "HOLD", "reason": "Failed to fetch real-time bar."}
        # Append new bar and recalculate indicators for the latest point
        temp_data = self.data.copy()
        temp_data.loc[new_bar.name] = new_bar.copy()
        temp_data.sort_index(inplace=True)

        # Recalculate MACD and Stochastic on the combined dataset
        macd_instance = MACD(close=temp_data['close'], window_fast=self.MACD_FAST, window_slow=self.MACD_SLOW, window_sign=self.MACD_SIGNAL, fillna=False) 
        stoch_instance = StochasticOscillator(high=temp_data['high'], low=temp_data['low'], close=temp_data['close'], window=self.STOCH_K, smooth_window=self.STOCH_D, fillna=False) 
        
        # Update self.data with the full dataset including the new bar for consistency
        self.data = temp_data.copy()
        
        # Assign the newly calculated series to the main data frame
        self.data['MACD_Hist'] = macd_instance.macd_diff() 
        self.data['Stoch_K'] = stoch_instance.stoch() 
        self.data['Stoch_D'] = stoch_instance.stoch_signal() 
        
        # Check for column existence (simplified)
        if not all(col in self.data.columns for col in ['MACD_Hist', 'Stoch_K', 'Stoch_D']): 
            print("ERROR: Real-time indicator calculation failed to produce expected columns.")
            return {"signal": "HOLD", "reason": "Real-time indicator calculation failed."}
        
        self.data.dropna(subset=['MACD_Hist', 'Stoch_K', 'Stoch_D'], inplace=True)
        # Get latest and previous data points
        if len(self.data) < 2:
            return {"signal": "HOLD", "reason": "Not enough data points after latest bar append."}
        latest_data = self.data.iloc[-1]
        prev_data = self.data.iloc[-2]

        latest_close = latest_data['close']
        latest_macd_hist = latest_data['MACD_Hist']
        prev_stoch_k = prev_data['Stoch_K']
        prev_stoch_d = prev_data['Stoch_D']

        # 2. Fetch Latest Fundamental Score
        print("--- Running Step 2: Fetching Real-Time Fundamental Sentiment ---")
        fundamental_score = self.fundamental_processor.fetch_realtime_sentiment()

        # 3. Decision Logic and Price Calculation
        signal = "HOLD"
        stop_loss, take_profit = None, None

        # Technical Bias
        macd_bullish = latest_macd_hist > 0
        macd_bearish = latest_macd_hist < 0
        stoch_bullish_cross = (prev_stoch_k < prev_stoch_d) and (latest_data['Stoch_K'] > latest_data['Stoch_D']) and (latest_data['Stoch_D'] < 50)
        stoch_bearish_cross = (prev_stoch_k > prev_stoch_d) and (latest_data['Stoch_K'] < latest_data['Stoch_D']) and (latest_data['Stoch_D'] > 50)

        technical_bias = "NEUTRAL"
        if macd_bullish and stoch_bullish_cross:
            technical_bias = "BUY"
        elif macd_bearish and stoch_bearish_cross:
            technical_bias = "SELL"

        entry_price = latest_close # Entry is the close price of the signal bar

        # --- Final Signal with Fundamental Filter ---

        if technical_bias == "BUY" and fundamental_score >= self.FUNDAMENTAL_THRESHOLD:
            signal = "BUY"
            stop_loss = entry_price - (self.STOP_LOSS_PIPS / self.PIP_CONVERSION)
            take_profit = entry_price + (self.TAKE_PROFIT_PIPS / self.PIP_CONVERSION)

            reason = (
                f"Confirmed **BUY**: MACD/Stoch alignment (Bullish) with strong sentiment ({fundamental_score:.3f}). "
                f"Entry Price: {entry_price:.5f} | SL: {stop_loss:.5f} ({self.STOP_LOSS_PIPS} pips) | TP: {take_profit:.5f} ({self.TAKE_PROFIT_PIPS} pips)"
            )

        elif technical_bias == "SELL" and fundamental_score <= -self.FUNDAMENTAL_THRESHOLD:
            signal = "SELL"
            stop_loss = entry_price + (self.STOP_LOSS_PIPS / self.PIP_CONVERSION)
            take_profit = entry_price - (self.TAKE_PROFIT_PIPS / self.PIP_CONVERSION)

            reason = (
                f"Confirmed **SELL**: MACD/Stoch alignment (Bearish) with strong sentiment ({fundamental_score:.3f}). "
                f"Entry Price: {entry_price:.5f} | SL: {stop_loss:.5f} ({self.STOP_LOSS_PIPS} pips) | TP: {take_profit:.5f} ({self.TAKE_PROFIT_PIPS} pips)"
            )
        else:
            reason = f"HOLD: Technical bias is {technical_bias}, or fundamental sentiment is neutral/conflicting ({fundamental_score:.3f})."

        return {
            "signal": signal,
            "reason": reason,
            "timestamp": latest_data.name.strftime('%Y-%m-%d %H:%M:%S'),
            "entry_price": entry_price if signal != "HOLD" else None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_pips": self.STOP_LOSS_PIPS,
            "tp_pips": self.TAKE_PROFIT_PIPS
        }
# --- Main Execution Function ---
def run_signal_generation_logic():
    """Initializes and runs the signal generation, backtesting, and email process."""

    if not ALPHA_VANTAGE_API_KEY:
        print("FATAL: ALPHA_VANTAGE_API_KEY environment variable is not set. Exiting.")
        return {"signal": "HOLD", "reason": "Missing Alpha Vantage API Key."}

    if not MARKETAUX_API_KEY:
        print("WARNING: MARKETAUX_API_KEY environment variable is not set. Sentiment will be 0.0.")
    try:
        market_data_api = AlphaVantageDataFeed(api_key=ALPHA_VANTAGE_API_KEY)
        sentiment_api = MarketAuxProcessor(api_key=MARKETAUX_API_KEY)
        generator = SignalGenerator(data_feed=market_data_api, fundamental_processor=sentiment_api)
        # 1. Initialize historical data (runs once)
        print("\n" + "="*50)
        print("### STEP 1: INITIALIZING DATA ###")
        print("="*50)
        generator.initialize_data()

        if generator.data is None or generator.data.empty:
            print("\nFATAL: Initialization failed. Exiting.")
            return {"signal": "HOLD", "reason": "Historical data initialization failed."}
        # 2. Backtest the predicted signals
        print("\n" + "="*50)
        print("### STEP 2: RUNNING BACKTESTING ###")
        print("="*50)

        backtester = Backtester(
            data=generator.data.copy(),
            fundamental_threshold=generator.FUNDAMENTAL_THRESHOLD,
            sl_pips=generator.STOP_LOSS_PIPS,
            tp_pips=generator.TAKE_PROFIT_PIPS
        )
        backtest_results = backtester.run_backtest()
        print("\n" + "*"*50)
        print("### BACKTEST RESULTS (Using Fixed TP/SL) ###")
        print(f"Total Trades Analyzed: {backtest_results['total_trades']}")
        print(f"Winning Trades: {backtest_results['winning_trades']}")
        print(f"Net Pips Gained: **{backtest_results['net_pips']:.2f}**")
        print(f"Profit Factor: **{backtest_results['profit_factor']:.2f}**")
        print(f"Details: {backtest_results['reason']}")
        print("*"*50)

        # 3. Run the real-time signal generation
        print("\n" + "="*50)
        print("### STEP 3: GENERATING REAL-TIME SIGNAL ###")
        print("="*50 + "\n")

        signal_result = generator.generate_signal()

        # 4. Print and Email the result

        # Create the email content body
        email_body = f"""
TRADING SIGNAL NOTIFICATION - EUR/USD ({generator.interval})
PAIR: {generator.symbol}
SIGNAL: {signal_result['signal']}
TIMESTAMP (Signal Close): {signal_result.get('timestamp', 'N/A')}
--- TRADE DETAILS ---
ENTRY PRICE: {signal_result.get('entry_price', 'N/A') if signal_result.get('entry_price') is None else f"{signal_result['entry_price']:.5f}"}
STOP LOSS (SL) PRICE: {signal_result.get('stop_loss', 'N/A') if signal_result.get('stop_loss') is None else f"{signal_result['stop_loss']:.5f}"} ({signal_result.get('sl_pips', 'N/A')} pips)
TAKE PROFIT (TP) PRICE: {signal_result.get('take_profit', 'N/A') if signal_result.get('take_profit') is None else f"{signal_result['take_profit']:.5f}"} ({signal_result.get('tp_pips', 'N/A')} pips)
Risk/Reward Ratio: 1:1.5
REASON: {signal_result['reason']}
*NOTE: Enter the trade at the ENTRY PRICE (close of the signal bar) and set your Stop Loss and Take Profit levels immediately to manage risk.*
"""
        # Console output
        print("\n" + "#"*50)
        print("### 🔔 FINAL TRADING SIGNAL NOTIFICATION 🔔 ###")
        print(f"PAIR: {generator.symbol}")
        print(f"SIGNAL: **{signal_result['signal']}**")
        print(f"Timestamp: {signal_result.get('timestamp', 'N/A')}")
        print(f"Entry: {signal_result.get('entry_price', 'N/A') if signal_result.get('entry_price') is None else f'{signal_result["entry_price"]:.5f}'}")
        print(f"SL: {signal_result.get('stop_loss', 'N/A') if signal_result.get('stop_loss') is None else f'{signal_result["stop_loss"]:.5f}'} | TP: {signal_result.get('take_profit', 'N/A') if signal_result.get('take_profit') is None else f'{signal_result["take_profit"]:.5f}'}")
        print(f"Reason: {signal_result['reason']}")
        print("#"*50 + "\n")
        # Email function call
        if signal_result['signal'] != "HOLD":
            subject = f"**{signal_result['signal']}** Signal Generated for EUR/USD!"
        else:
            subject = "HOLD Signal Generated for EUR/USD."

        send_email_notification(
            subject=subject,
            body=email_body,
            recipient=MAIL_RECIPIENT
        )

        return signal_result
    except Exception as e:
        print("\n" + "!"*50)
        print("### 🛑 CRITICAL EXECUTION ERROR DURING RUNTIME 🛑 ###")
        print(f"An unexpected error occurred: {e}")
        # Use traceback to print the full stack trace for debugging
        print("\n--- Detailed Stack Trace ---")
        traceback.print_exc()
        print("!"*50 + "\n")

        # Attempt to notify via email about the failure
        error_subject = "FATAL: Trading Bot Execution Failed"
        error_body = f"The trading signal logic encountered a critical error:\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}"
        send_email_notification(error_subject, error_body, MAIL_RECIPIENT)

        return {"signal": "ERROR", "reason": f"Critical runtime error: {e}"}
# --- Flask Routes ---
@app.route('/', methods=['GET'])
def home():
    """Simple status check endpoint."""
    return "Trading Signal Generator is running. Use /run to execute the logic."
@app.route('/run', methods=['POST', 'GET'])
def run_script():
    """
    Endpoint to trigger the trading logic. The logic is run in a separate 
    thread to prevent Gunicorn worker timeouts.
    """
    print("--- Received request to run trading logic (Starting Background Thread) ---")

    # Start the long-running logic in a separate thread.
    thread = Thread(target=run_signal_generation_logic)
    thread.start()

    # Return immediately (HTTP 202 Accepted)
    response = {
        "status": "Processing started in background",
        "message": "The signal generation logic is running asynchronously. Check the application logs and your email for the final signal output."
    }

    return jsonify(response), 202
if __name__ == "__main__":
    print("Starting Flask server for manual execution...")
    app.run(host='0.0.0.0', port=os.environ.get("PORT", 5000))
