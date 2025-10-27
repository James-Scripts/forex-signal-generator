import requests
import pandas as pd
import pandas_ta as ta
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
# 💡 VADER Sentiment Analysis Import (Required for the strategy, if used later)
# import nltk 
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, jsonify, request

# --- Flask Application Setup ---
app = Flask(__name__)

# --- API Keys Configuration ---
# ALPHA_VANTAGE_API_KEY REMOVED
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY") # New Price API Key
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
    """Sends an email using the configured SMTP settings."""
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

# --- 1. Data Feed (TwelveData Replacement) ---
class DataFeed:
    """Base class for fetching both historical and real-time market data."""
    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        raise NotImplementedError
    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        raise NotImplementedError

class TwelveDataFeed(DataFeed):
    """Fetches market data using the TwelveData API."""
    BASE_URL = "https://api.twelvedata.com"
    TIME_INTERVAL = "60min"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_cache = {}

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        """Fetches historical data for the given symbol and interval."""
        if symbol in self.data_cache:
            print(f"DEBUG: Historical Data for {symbol} loaded from cache.")
            return self.data_cache[symbol].copy()

        print(f"DEBUG: Fetching historical {interval} data for {symbol}...")
        
        # Calculate start date for lookback
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        
        params = {
            "symbol": symbol.replace('/', ''),
            "interval": interval,
            "start_date": start_date,
            "outputsize": 5000, # Max allowed output size for historical data
            "apikey": self.api_key
        }

        try:
            # TwelveData uses the 'time_series' endpoint for historical bars
            response = requests.get(f"{self.BASE_URL}/time_series", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data or 'message' in data and 'exceeded' in data.get('message', '').lower():
                print(f"TwelveData API Error: {data.get('message', data.get('error', 'Unknown Error'))}")
                return pd.DataFrame()
            
            if 'values' not in data:
                print(f"ERROR: TwelveData response missing 'values' key: {data}")
                return pd.DataFrame()

            df = pd.DataFrame(data['values'])
            df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)

            self.data_cache[symbol] = df
            print(f"DEBUG: Historical Data loaded. Total bars: {len(df)}")
            return df.copy()

        except requests.exceptions.RequestException as e:
            print(f"TwelveData API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Failed to process TwelveData historical data: {e}")
            return pd.DataFrame()

    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        """Fetches the latest intraday bar data (simulated from the latest historical bar)."""
        # For simplicity and to reuse the historical endpoint structure,
        # we fetch the latest bar from the time_series endpoint.
        
        params = {
            "symbol": symbol.replace('/', ''),
            "interval": interval,
            "outputsize": 1, 
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}/time_series", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'values' not in data or not data['values']:
                print("ERROR: TwelveData failed to fetch real-time bar.")
                return pd.Series()

            latest_bar_data = data['values'][0]
            timestamp = pd.to_datetime(latest_bar_data.pop('datetime'))
            
            latest_bar = pd.Series({
                'open': float(latest_bar_data['open']),
                'high': float(latest_bar_data['high']),
                'low': float(latest_bar_data['low']),
                'close': float(latest_bar_data['close'])
            }, name=timestamp)

            return latest_bar

        except requests.exceptions.RequestException as e:
            print(f"TwelveData real-time bar request failed: {e}")
            return pd.Series()
        except Exception as e:
            print(f"Failed to process TwelveData real-time data: {e}")
            return pd.Series()

# --- 2. Fundamental Processor (MarketAux) (Sentiment API, UNCHANGED) ---
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

# --- 3. Trading Strategy and Signal Generation (UNCHANGED) ---
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
        """Calculates MACD and Stochastic Oscillator using the pandas_ta library."""
        print("DEBUG: Calculating technical indicators (MACD, Stochastic)...")
        # MACD (12, 26, 9)
        self.data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        # Stochastic Oscillator (14, 3, 3)
        self.data.ta.stoch(high='high', low='low', close='close', k=14, d=3, smooth_k=3, append=True)

        self.data.dropna(inplace=True)

        self.data.rename(columns={
            'MACDh_12_26_9': 'MACD_Hist',
            'STOCHk_14_3_3': 'Stoch_K',
            'STOCHd_14_3_3': 'Stoch_D'
        }, inplace=True)
        
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
            print("ERROR: TwelveData API Key is missing. Cannot fetch data.")
            self.data = pd.DataFrame()
            return

        self.data = self.data_feed.fetch_historical_data(self.symbol, self.interval, self.lookback_years)

        if self.data is not None and not self.data.empty:
            print(f"DEBUG: Historical Data loaded. Total bars: {len(self.data)}")

            # Calculate initial technical indicators on the historical data
            self.data.ta.macd(close='close', fast=self.MACD_FAST, slow=self.MACD_SLOW, signal=self.MACD_SIGNAL, append=True)
            self.data.ta.stoch(high='high', low='low', close='close', k=self.STOCH_K, d=self.STOCH_D, smooth_k=3, append=True)
            
            # --- Robust Column Renaming and Check ---
            column_mapping = {
                f'MACDh_{self.MACD_FAST}_{self.MACD_SLOW}_{self.MACD_SIGNAL}': 'MACD_Hist',
                f'STOCHk_{self.STOCH_K}_3_3': 'Stoch_K',
                f'STOCHd_{self.STOCH_K}_3_3': 'Stoch_D'
            }

            valid_renames = {old: new for old, new in column_mapping.items() if old in self.data.columns}

            if len(valid_renames) < 3:
                missing_cols = set(column_mapping.keys()) - set(self.data.columns)
                print(f"FATAL: One or more technical indicator columns were not created successfully: {missing_cols}. Invalidate data.")
                self.data = pd.DataFrame()
                return

            self.data.rename(columns=valid_renames, inplace=True)
            # --- End Robust Column Renaming ---
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
        macd_results = temp_data['close'].ta.macd(fast=self.MACD_FAST, slow=self.MACD_SLOW, signal=self.MACD_SIGNAL, append=False)
        stoch_results = temp_data.ta.stoch(high='high', low='low', close='close', k=self.STOCH_K, d=self.STOCH_D, smooth_k=3, append=False)

        # Update self.data with the full dataset including the new bar for consistency
        self.data = temp_data.copy()

        # Ensure indicator columns exist before assigning them
        macd_hist_col = f'MACDh_{self.MACD_FAST}_{self.MACD_SLOW}_{self.MACD_SIGNAL}'
        stoch_k_col = f'STOCHk_{self.STOCH_K}_3_3'
        stoch_d_col = f'STOCHd_{self.STOCH_K}_3_3'

        if macd_hist_col not in macd_results.columns or stoch_k_col not in stoch_results.columns:
            print("ERROR: Real-time indicator calculation failed to produce expected columns.")
            return {"signal": "HOLD", "reason": "Real-time indicator calculation failed."}

        self.data['MACD_Hist'] = macd_results[macd_hist_col]
        self.data['Stoch_K'] = stoch_results[stoch_k_col]
        self.data['Stoch_D'] = stoch_results[stoch_d_col]

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

    if not TWELVEDATA_API_KEY:
        print("FATAL: TWELVEDATA_API_KEY environment variable is not set. Exiting.")
        return {"signal": "HOLD", "reason": "Missing TwelveData API Key."}

    if not MARKETAUX_API_KEY:
        print("WARNING: MARKETAUX_API_KEY environment variable is not set. Sentiment will be 0.0.")
        
    try:
        # 💡 Replaced AlphaVantageDataFeed with TwelveDataFeed
        market_data_api = TwelveDataFeed(api_key=TWELVEDATA_API_KEY)
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
