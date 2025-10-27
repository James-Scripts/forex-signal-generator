import requests
import pandas as pd
import pandas_ta as ta
import yfinance as yf 
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

# --- NLP Imports for Sentiment Analysis (NEW) ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

# NOTE: For the script to run, ensure you have the 'vader_lexicon' downloaded.
# Run this once in your terminal or Python environment: 
# import nltk; nltk.download('vader_lexicon')

# --- Flask Application Setup ---
app = Flask(__name__)

# ==============================================================================
# 0. API Keys & Configuration (Unchanged)
# ==============================================================================

# --- PRICE DATA KEYS ---
TWELVEDATA_API_KEY = "7ded66b4a2184314a57abd4f8f6b304b"
FMP_API_KEY = "A0xuQ94tqyjfAKitVIGoNKPNnBX2K0JT"

# --- NEWS DATA KEYS ---
NEWS_API_KEY_1 = "7ec3a80cd7564d2c8652cd2ec6b83c14" 
MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY") # Keeping MarketAux as an environment variable

# --- EMAIL Configuration ---
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
MAIL_SERVER = os.environ.get("MAIL_SERVER", 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get("MAIL_PORT", 465))
MAIL_SENDER = os.environ.get("MAIL_SENDER")
MAIL_RECIPIENT = os.environ.get("MAIL_RECIPIENT")

# --- GLOBAL TRADING PARAMS ---
SYMBOL = "EUR/USD"
FOREX_TICKER = "EURUSD=X" # Ticker for yfinance
INTERVAL = "60min"
DAYS_OF_DATA = 30 # Lookback for yfinance/FMP

# --- 0. Email Utility Function (Unchanged) ---
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

# ==============================================================================
# 1. ENSEMBLE DATA FEED (Unchanged)
# ==============================================================================

class DataAPIWrapper:
    """Base class for individual price API wrappers."""
    def fetch_historical_ohlc(self, symbol: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError

# --- Twelve Data Wrapper ---
class TwelveDataWrapper(DataAPIWrapper):
    def fetch_historical_ohlc(self, symbol=SYMBOL, interval=INTERVAL):
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": '1h', 
            "apikey": TWELVEDATA_API_KEY,
            "outputsize": 500,
            "timezone": "exchange",
        }
        try:
            response = requests.get(url, params=params, timeout=10).json()
            if 'values' not in response: return pd.DataFrame()
            df = pd.DataFrame(response['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close']].astype(float)
        except Exception: return pd.DataFrame()

# --- FMP Wrapper (Financial Modeling Prep) ---
class FMPWrapper(DataAPIWrapper):
    def fetch_historical_ohlc(self, symbol=SYMBOL, interval=INTERVAL):
        if interval != '60min': return pd.DataFrame()
        limit = 500
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol.replace('/', '')}" 
        params = {"apikey": FMP_API_KEY, "limit": limit}
        try:
            response = requests.get(url, params=params, timeout=10).json()
            if not isinstance(response, list) or not response: return pd.DataFrame()
            df = pd.DataFrame(response)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df[['open', 'high', 'low', 'close']].astype(float)
        except Exception: return pd.DataFrame()

# --- Yahoo Finance Wrapper (using yfinance) ---
class YFinanceWrapper(DataAPIWrapper):
    def fetch_historical_ohlc(self, symbol=FOREX_TICKER, interval=INTERVAL):
        try:
            start_date = (datetime.now() - timedelta(days=DAYS_OF_DATA)).strftime('%Y-%m-%d')
            df = yf.download(symbol, start=start_date, interval='60m', progress=False, timeout=10)
            if df.empty: return pd.DataFrame()
            df.columns = df.columns.str.lower()
            return df[['open', 'high', 'low', 'close']]
        except Exception: return pd.DataFrame()

class ReconcilingDataFeed:
    """Combines historical data from multiple sources for robust analysis."""

    def __init__(self):
        self.td = TwelveDataWrapper()
        self.fmp = FMPWrapper()
        self.yf = YFinanceWrapper()
        self.data_cache = {}

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        """Fetches and reconciles historical OHLC data."""
        if symbol in self.data_cache:
            print(f"DEBUG: Historical Data for {symbol} loaded from cache.")
            return self.data_cache[symbol].copy()

        data_sources = {
            'TwelveData': self.td.fetch_historical_ohlc(symbol, interval),
            'FMP': self.fmp.fetch_historical_ohlc(symbol, interval),
            'YFinance': self.yf.fetch_historical_ohlc(FOREX_TICKER, interval),
        }
        
        valid_data = {}
        for name, df in data_sources.items():
            if not df.empty and len(df) > 50: # Minimum bars for reliable MACD/Stoch
                valid_data[name] = df
            else:
                print(f"WARNING: {name} data failed or has insufficient data points. Skipping.")

        if len(valid_data) < 2:
            print(f"FATAL ERROR: Only {len(valid_data)} valid price source(s) loaded. Reconciliation skipped.")
            # Fallback: return the one valid source or an empty DataFrame
            return next(iter(valid_data.values())) if valid_data else pd.DataFrame()

        # Step 1: Align all data frames on their index (time)
        combined_df = pd.concat(valid_data.values(), keys=valid_data.keys(), axis=1, join='inner')

        # Step 2: Reconcile Prices using the Median
        reconciled_ohlc = pd.DataFrame(index=combined_df.index)
        
        for col in ['open', 'high', 'low', 'close']:
            price_cols = [c for c in combined_df.columns if c[1] == col]
            reconciled_ohlc[col] = combined_df[price_cols].median(axis=1, skipna=True)
            
        # Final cleanup and caching
        reconciled_ohlc.sort_index(inplace=True)
        reconciled_ohlc = reconciled_ohlc[~reconciled_ohlc.index.duplicated(keep='first')]
        self.data_cache[symbol] = reconciled_ohlc.dropna()
        print(f"DEBUG: Successfully reconciled data from {len(valid_data)} sources. Total bars: {len(self.data_cache[symbol])}")
        return self.data_cache[symbol].copy()

    def fetch_latest_bar(self, symbol: str) -> pd.Series:
        """Retrieves the latest bar from the cached historical data."""
        if symbol in self.data_cache and not self.data_cache[symbol].empty:
            return self.data_cache[symbol].iloc[-1]
        return pd.Series()
    

# ==============================================================================
# 2. FUNDAMENTAL PROCESSOR (UPDATED for VADER)
# ==============================================================================

class EnsembleFundamentalProcessor:
    """Fetches and combines sentiment from multiple news APIs using VADER NLP."""

    def __init__(self, api_key_1: str, api_key_2: Optional[str]):
        self.api_key_1 = api_key_1
        self.api_key_2 = api_key_2
        # Initialize VADER Sentiment Analyzer (NEW)
        self.vader = SentimentIntensityAnalyzer()


    def _analyze_sentiment(self, text: str) -> float:
        """Calculates VADER compound score for a given text."""
        if not text:
            return 0.0
        # The compound score is the normalized, weighted composite score (-1 to +1)
        return self.vader.polarity_scores(text)['compound']


    def fetch_realtime_sentiment(self, symbol: str = SYMBOL) -> float:
        """Fetches the latest financial news sentiment from multiple sources."""
        
        scores = []
        
        # --- Source 1: News API (VADER Integration) ---
        url_1 = "https://newsapi.org/v2/everything"
        params_1 = {
            "q": f"forex {symbol}",
            "language": "en",
            "sortBy": "relevancy",
            "apiKey": self.api_key_1,
            "pageSize": 25
        }
        
        if self.api_key_1:
            try:
                response_1 = requests.get(url_1, params=params_1, timeout=5).json()
                articles_1 = response_1.get('articles', [])
                article_scores = []

                for article in articles_1:
                    # Combine title and description for a richer text analysis
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    score = self._analyze_sentiment(text)
                    if score != 0.0:
                        article_scores.append(score)
                
                if article_scores:
                    score_1 = np.mean(article_scores)
                    scores.append(score_1)
                    print(f"Source 1 (News API): Analyzed {len(article_scores)} articles. Average VADER Score: {score_1:.2f}")

            except Exception as e:
                print(f"Source 1 (News API) failed to fetch or process: {e}")

        # --- Source 2: MarketAux Processor (VADER Integration) ---
        if self.api_key_2:
            url_2 = "https://api.marketaux.com/v1/news/all"
            # Focusing the search on the currency pair
            params_2 = {
                "api_token": self.api_key_2,
                "search": "euro OR dollar OR eur/usd", 
                "filter_entities": "true",
                "language": "en",
                "limit": 10
            }
            try:
                response_2 = requests.get(url_2, params=params_2, timeout=5).json()
                articles_2 = response_2.get('data', [])
                article_scores = []

                for article in articles_2:
                    # Combine snippet and title for VADER analysis
                    text = f"{article.get('title', '')} {article.get('snippet', '')}"
                    
                    # Instead of complex entity-based math, use the overall article VADER score
                    score = self._analyze_sentiment(text)
                    if score != 0.0:
                        article_scores.append(score)
                
                if article_scores:
                    score_2 = np.mean(article_scores)
                    scores.append(score_2)
                    print(f"Source 2 (MarketAux): Analyzed {len(article_scores)} snippets. Average VADER Score: {score_2:.2f}")
                
            except Exception as e:
                print(f"Source 2 (MarketAux) failed to fetch or process: {e}")

        if not scores:
            print("WARNING: No valid news scores retrieved from any source. Sentiment defaulted to 0.0.")
            return 0.0
            
        # Return the simple average of all successful sources' average scores
        avg_sentiment = sum(scores) / len(scores)
        return avg_sentiment


# ==============================================================================
# 3. TRADING STRATEGY AND SIGNAL GENERATION (Unchanged Logic)
# ==============================================================================

class Backtester:
    # ... (CLASS CONTENT IS UNCHANGED) ...
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
        # This simulation remains for backtesting purposes. 
        # For LIVE signals, the real sentiment from VADER is used.
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
    # ... (CLASS CONTENT IS UNCHANGED) ...
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

    def __init__(self, data_feed: ReconcilingDataFeed, fundamental_processor: EnsembleFundamentalProcessor):
        self.data_feed = data_feed
        self.fundamental_processor = fundamental_processor
        self.symbol = SYMBOL
        self.interval = INTERVAL
        self.lookback_years = 2 
        self.data: Optional[pd.DataFrame] = None

    def initialize_data(self):
        """Fetches historical data to build the model's foundation using the ReconcilingDataFeed."""
        print(f"\n--- Initializing Historical Data for {self.symbol} ---")

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
        Generates a trade signal based on the latest bar, technical indicators, and fundamental score.
        """
        if self.data is None or self.data.empty:
            print("ERROR: Historical data not initialized. Cannot generate signal.")
            return {"signal": "HOLD", "reason": "Historical data missing."}

        # 1. Fetch Latest Bar (Using the latest bar from the reconciled dataset)
        print("\n--- Running Step 1: Fetching Latest Market Data from Reconciled Set ---")
        new_bar = self.data_feed.fetch_latest_bar(self.symbol) 

        if new_bar.empty:
            return {"signal": "HOLD", "reason": "Failed to fetch real-time bar."}
        
        # Get latest and previous data points
        if len(self.data) < 2:
            return {"signal": "HOLD", "reason": "Not enough data points after latest bar append."}
        
        latest_data = self.data.iloc[-1]
        prev_data = self.data.iloc[-2]

        latest_close = latest_data['close']
        latest_macd_hist = latest_data['MACD_Hist']
        prev_stoch_k = prev_data['Stoch_K']
        prev_stoch_d = prev_data['Stoch_D']

        # 2. Fetch Latest Fundamental Score (Now using VADER)
        print("--- Running Step 2: Fetching Real-Time Fundamental Sentiment (VADER) ---")
        fundamental_score = self.fundamental_processor.fetch_realtime_sentiment(self.symbol)

        # 3. Decision Logic and Price Calculation (Unchanged)
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

# --- Main Execution Function (Unchanged) ---
def run_signal_generation_logic():
    """Initializes and runs the signal generation, backtesting, and email process."""

    # Note: We now check for the specific keys we need for the new system.
    if not TWELVEDATA_API_KEY or not FMP_API_KEY:
        print("FATAL: Price Data API Keys (TwelveData/FMP) are missing. Cannot fetch data.")
        return {"signal": "HOLD", "reason": "Missing primary Price Data API Keys."}
    
    # We proceed even if news keys are missing, as VADER will default to 0.0, 
    # but the logs will clearly state the missing keys.
    if not NEWS_API_KEY_1 and not MARKETAUX_API_KEY:
        print("WARNING: All News API Keys are missing. Sentiment will be 0.0.")
    
    try:
        # 1. Initialize the new Ensemble Data Feeds
        market_data_api = ReconcilingDataFeed()
        # Sentiment Processor now uses VADER
        sentiment_api = EnsembleFundamentalProcessor(api_key_1=NEWS_API_KEY_1, api_key_2=MARKETAUX_API_KEY)
        
        generator = SignalGenerator(data_feed=market_data_api, fundamental_processor=sentiment_api)

        # 2. Initialize historical data (runs once)
        print("\n" + "="*50)
        print("### STEP 1: INITIALIZING ENSEMBLE DATA ###")
        print("="*50)
        generator.initialize_data()

        if generator.data is None or generator.data.empty:
            print("\nFATAL: Initialization failed. Exiting.")
            return {"signal": "HOLD", "reason": "Historical data initialization failed."}

        # 3. Backtest the predicted signals
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

        # 4. Run the real-time signal generation
        print("\n" + "="*50)
        print("### STEP 3: GENERATING REAL-TIME SIGNAL ###")
        print("="*50 + "\n")

        signal_result = generator.generate_signal()

        # 5. Print and Email the result
        
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

# --- Flask Routes (Unchanged) ---
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
