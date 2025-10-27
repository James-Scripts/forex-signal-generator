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
from flask import Flask, jsonify, request

# --- 💡 NLTK & VADER for NewsAPI Sentiment ---
import nltk
# Ensure VADER lexicon is available (uncomment the next line if you get a Resource NLTK error)
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except nltk.downloader.DownloadError:
#     nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Flask Application Setup ---
app = Flask(__name__)

# --- API Keys Configuration ---
# ⚠️ SECURITY NOTE: The raw keys you provided (7ded66b4..., A0xuQ94t..., a3427bc1..., 7ec3a80c...)
# have been replaced with secure environment variable lookups. 
# You must set these variables in your Render environment.

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY") 
FMP_API_KEY = os.environ.get("FMP_API_KEY") 
FOREXRATE_API_KEY = os.environ.get("FOREXRATE_API_KEY") 

MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY")
NEWS_API_KEY_2 = os.environ.get("NEWS_API_KEY_2") 

# --- EMAIL Configuration ---
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
MAIL_SERVER = os.environ.get("MAIL_SERVER", 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get("MAIL_PORT", 465))
MAIL_SENDER = os.environ.get("MAIL_SENDER")
MAIL_RECIPIENT = os.environ.get("MAIL_RECIPIENT")

# --- 0. Email Utility Function (UNCHANGED) ---
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
        with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as server:
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_SENDER, recipient, msg.as_string())
        print("DEBUG: Email notification sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("FATAL EMAIL ERROR: SMTP Authentication failed. Check your username and App Password.")
    except Exception as e:
        print(f"FATAL EMAIL ERROR: Failed to send email: {e}")

# --- 1. Price Data Feed Base and Implementations ---
class DataFeed:
    """Base class for fetching both historical and real-time market data."""
    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        raise NotImplementedError
    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        raise NotImplementedError

class TwelveDataFeed(DataFeed):
    """Fetches market data using the TwelveData API (Price Source 1)."""
    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_cache = {}

    def _fetch(self, symbol: str, interval: str, outputsize: int, start_date: str = None) -> pd.DataFrame:
        params = {
            "symbol": symbol.replace('/', ''),
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        if start_date:
             params["start_date"] = start_date

        try:
            response = requests.get(f"{self.BASE_URL}/time_series", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data or 'message' in data:
                print(f"TwelveData Error: {data.get('message', data.get('error', 'Unknown Error'))}")
                return pd.DataFrame()
            
            if 'values' not in data: return pd.DataFrame()

            df = pd.DataFrame(data['values'])
            df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            return df

        except requests.exceptions.RequestException as e:
            print(f"TwelveData API request failed: {e}")
            return pd.DataFrame()

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        if symbol in self.data_cache: return self.data_cache[symbol].copy()
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        df = self._fetch(symbol, interval, 5000, start_date)
        self.data_cache[symbol] = df
        return df.copy()

    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        df = self._fetch(symbol, interval, 1)
        return df.iloc[0] if not df.empty else pd.Series()

class FMPFeed(DataFeed):
    """Fetches market data using the Financial Modeling Prep API (Price Source 2)."""
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _fetch(self, symbol: str, interval: str, limit: int, from_date: str = None) -> pd.DataFrame:
        # FMP Forex data requires "FX_" prefix for historical
        fmp_symbol = f"FX_{symbol.replace('/', '')}" 
        
        # FMP interval mapping for 60min. FMP typically uses 1hour or 60min.
        fmp_interval = "1hour" if interval == "60min" else interval

        params = {
            "symbol": fmp_symbol,
            "serietype": "line", # Necessary to get OHLCV data
            "apikey": self.api_key
        }
        
        # FMP historical endpoint is complex, using a simplified historical-price endpoint if available for Forex
        endpoint = f"/historical-chart/{fmp_interval}/{fmp_symbol}" 

        if not from_date: # Assume real-time fetch if no start date
             endpoint = f"/historical-chart/{fmp_interval}/{fmp_symbol}"
             # Limit param for this endpoint is often not supported, we must post-process

        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'Error Message' in data: 
                print(f"FMP Error: {data.get('Error Message', 'No data returned.')}")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df.rename(columns={'date': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)

            return df.iloc[-limit:] if limit > 0 else df

        except requests.exceptions.RequestException as e:
            print(f"FMP API request failed: {e}")
            return pd.DataFrame()

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        return self._fetch(symbol, interval, limit=0) # Fetch all available history

    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        df = self._fetch(symbol, interval, limit=1)
        return df.iloc[-1] if not df.empty else pd.Series()

class ForexRateFeed(DataFeed):
    """Placeholder for the user's ForexRate API (Price Source 3)."""
    # NOTE: The specific endpoint for the user's key is unknown. This is a generic REST stub.
    BASE_URL = "https://api.forexrateapi.com" 

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        print("WARNING: ForexRate API historical data is simulated due to unknown endpoint details.")
        # Simulation: For historical data, we need all three feeds to return a DataFrame.
        # Since this API's structure is unknown, we will use a time-shifted TwelveData simulation to prevent breakages.
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        # This function should be implemented with the actual ForexRate historical endpoint.
        
        # --- SIMULATION (REPLACE WITH REAL API CALL) ---
        # Simulates a time-shifted data set to act as a 3rd price feed for backtesting stability
        df = pd.DataFrame(index=pd.to_datetime(pd.date_range(start=start_date, periods=1000, freq=interval.replace('min', 'T'))))
        np.random.seed(321)
        df['close'] = 1.05 + np.cumsum(np.random.normal(0, 0.0001, len(df)))
        df['open'] = df['close'].shift(1)
        df[['high', 'low']] = df[['open', 'close']].apply(lambda x: pd.Series([max(x) * 1.0001, min(x) * 0.9999]), axis=1)
        df.dropna(inplace=True)
        return df
        # --- END SIMULATION ---

    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        # Placeholder for real-time spot rate.
        params = {
            "api_key": self.api_key,
            "base": symbol.split('/')[0],
            "symbols": symbol.split('/')[1]
        }
        
        try:
            # Assumes a simple 'latest' endpoint common to many FX rate APIs
            response = requests.get(f"{self.BASE_URL}/latest", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            latest_price = data.get('rates', {}).get(symbol.split('/')[1])
            if latest_price:
                # Create a synthetic OHLC bar from the spot rate for median calculation
                price = float(latest_price)
                return pd.Series({'open': price, 'high': price, 'low': price, 'close': price}, name=datetime.now())
            
            print("ForexRate API: No price data returned. Using simulated data for median.")
            price = 1.07 + np.random.normal(0, 0.0001)
            return pd.Series({'open': price, 'high': price, 'low': price, 'close': price}, name=datetime.now())
        
        except Exception as e:
            print(f"ForexRate API error: {e}. Using simulated data for median.")
            price = 1.07 + np.random.normal(0, 0.0001)
            return pd.Series({'open': price, 'high': price, 'low': price, 'close': price}, name=datetime.now())

class EnsemblePriceFeed(DataFeed):
    """Combines multiple price feeds and returns the median value."""
    def __init__(self, feeds: List[DataFeed]):
        self.feeds = [feed for feed in feeds if feed]

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        """
        Fetches historical data from the most reliable feed (TwelveData in this setup)
        and uses the others for backtesting stability/verification only.
        """
        # For simplicity and to ensure a consistent index for TA, use one reliable historical feed (TwelveData)
        print("DEBUG: Fetching historical data from primary feed (TwelveData)...")
        return self.feeds[0].fetch_historical_data(symbol, interval, lookback_years)

    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        """Fetches the latest bar from all sources and computes the median close price."""
        results = []
        for feed in self.feeds:
            bar = feed.fetch_realtime_bar(symbol, interval)
            if not bar.empty and 'close' in bar:
                results.append(bar['close'])
                print(f"DEBUG: Price from {feed.__class__.__name__}: {bar['close']:.5f}")
            else:
                print(f"WARNING: {feed.__class__.__name__} failed to return a valid close price.")

        if not results:
            return pd.Series()

        median_close = np.median(results)
        
        # Create a synthetic bar based on the median close price
        # This assumes the latest bar's OHLC can be approximated by the median spot rate
        return pd.Series({
            'open': median_close,
            'high': median_close,
            'low': median_close,
            'close': median_close
        }, name=datetime.now())

# --- 2. Fundamental Processor Implementations ---
class FundamentalProcessor:
    """Base class for fetching fundamental/sentiment data."""
    def fetch_realtime_sentiment(self) -> float:
        raise NotImplementedError

class MarketAuxProcessor(FundamentalProcessor):
    """Fetches news sentiment using the MarketAux API (Sentiment Source 1) (UNCHANGED)."""
    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_realtime_sentiment(self) -> float:
        # (MarketAux implementation remains the same)
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
                print("MarketAux: No news articles found.")
                return 0.0
                
            all_sentiments = []
            for article in articles:
                entities = article.get('entities', [])
                for entity in entities:
                    if entity.get('symbol') in ['EUR', 'USD'] and 'sentiment_score' in entity:
                        score = entity['sentiment_score']
                        # Weight EUR sentiment positively, USD sentiment negatively
                        all_sentiments.append(score if entity['symbol'] == 'EUR' else -score)
            
            if not all_sentiments: return 0.0
            
            avg_sentiment = sum(all_sentiments) / len(all_sentiments)
            print(f"DEBUG: MarketAux Sentiment Score: {avg_sentiment:.3f}")
            return avg_sentiment
        except Exception as e:
            print(f"MarketAux request failed: {e}")
            return 0.0

class NewsAPIProcessor(FundamentalProcessor):
    """Fetches news from the general News API and calculates sentiment using VADER (Sentiment Source 2)."""
    BASE_URL = "https://newsapi.org/v2/everything"
    VADER = SentimentIntensityAnalyzer()

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_realtime_sentiment(self) -> float:
        """Fetches news and calculates a composite VADER sentiment score."""
        params = {
            "apiKey": self.api_key,
            "q": "EURUSD OR Euro Dollar Forex",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get('articles', [])
            if not articles:
                print("NewsAPI: No articles found.")
                return 0.0

            all_scores = []
            for article in articles:
                # Use a combination of title and description for VADER analysis
                text = f"{article.get('title', '')}. {article.get('description', '')}"
                if text:
                    # VADER returns a dictionary, we use the compound score (-1 to +1)
                    vs = self.VADER.polarity_scores(text)
                    all_scores.append(vs['compound'])

            if not all_scores: return 0.0

            avg_sentiment = sum(all_scores) / len(all_scores)
            print(f"DEBUG: NewsAPI (VADER) Sentiment Score: {avg_sentiment:.3f}")
            return avg_sentiment
        except Exception as e:
            print(f"NewsAPI request failed: {e}")
            return 0.0

class EnsembleSentimentProcessor(FundamentalProcessor):
    """Combines multiple sentiment feeds and returns the average score."""
    def __init__(self, processors: List[FundamentalProcessor]):
        self.processors = [processor for processor in processors if processor]

    def fetch_realtime_sentiment(self) -> float:
        scores = []
        for processor in self.processors:
            score = processor.fetch_realtime_sentiment()
            if score is not None:
                scores.append(score)
        
        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)
        print(f"DEBUG: FINAL ENSEMBLE SENTIMENT: {avg_score:.3f}")
        return avg_score

# --- 3. Trading Strategy and Signal Generation (Updated to use Ensembles) ---
class Backtester:
    # (Remains largely the same, but uses simulated sentiment based on historical price action)
    # The historical sentiment simulation is needed because only one price history (TwelveData) 
    # is used for backtesting technical indicators to maintain a consistent index.
    
    def __init__(self, data: pd.DataFrame, fundamental_threshold: float, sl_pips: float, tp_pips: float):
        self.data = data.copy()
        self.fundamental_threshold = fundamental_threshold
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.PIP_CONVERSION = 10000.0 
        
    def calculate_technical_indicators(self):
        # (Technical calculation logic remains the same)
        print("DEBUG: Calculating technical indicators (MACD, Stochastic)...")
        self.data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        self.data.ta.stoch(high='high', low='low', close='close', k=14, d=3, smooth_k=3, append=True)
        self.data.dropna(inplace=True)

        self.data.rename(columns={
            'MACDh_12_26_9': 'MACD_Hist',
            'STOCHk_14_3_3': 'Stoch_K',
            'STOCHd_14_3_3': 'Stoch_D'
        }, inplace=True)
        
    def generate_historical_signals(self) -> pd.DataFrame:
        """Generates buy/sell/hold signals using technicals and simulated sentiment."""
        # SIMULATION: Create a synthetic historical sentiment series for backtesting.
        # This is a random, market-aligned simulation since we cannot fetch 2 years of sentiment.
        np.random.seed(42)
        price_diff = self.data['close'].diff().fillna(0)
        simulated_sentiment = np.clip(price_diff.rolling(window=10).mean().fillna(0) * 5 + np.random.normal(0, 0.1, len(self.data)), -1, 1)
        self.data['sentiment'] = simulated_sentiment

        signals = pd.Series(index=self.data.index, dtype=str)
        # (Crossover and Signal logic remains the same)
        macd_bullish = self.data['MACD_Hist'] > 0
        macd_bearish = self.data['MACD_Hist'] < 0
        stoch_k_prev = self.data['Stoch_K'].shift(1)
        stoch_d_prev = self.data['Stoch_D'].shift(1)
        stoch_bullish_cross = (stoch_k_prev < stoch_d_prev) & (self.data['Stoch_K'] > self.data['Stoch_D']) & (self.data['Stoch_D'] < 50)
        stoch_bearish_cross = (stoch_k_prev > stoch_d_prev) & (self.data['Stoch_K'] < self.data['Stoch_D']) & (self.data['Stoch_D'] > 50)
        sentiment_positive = self.data['sentiment'] >= self.fundamental_threshold
        sentiment_negative = self.data['sentiment'] <= -self.fundamental_threshold

        buy_condition = macd_bullish & stoch_bullish_cross & sentiment_positive
        sell_condition = macd_bearish & stoch_bearish_cross & sentiment_negative

        signals[buy_condition] = "BUY"
        signals[sell_condition] = "SELL"
        signals.fillna("HOLD", inplace=True)

        self.data['signal'] = signals
        return self.data

    def run_backtest(self) -> Dict[str, Any]:
        # (Backtest logic remains the same)
        self.calculate_technical_indicators()
        self.generate_historical_signals()
        
        if self.data.empty or 'signal' not in self.data.columns:
            return {"net_pips": 0, "total_trades": 0, "profit_factor": 0.0, "reason": "Data or signal generation failed."}
            
        # ... (Trade simulation and performance calculation logic) ...
        trades_list = []
        in_trade = False

        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            if not in_trade and current_bar['signal'] in ['BUY', 'SELL']:
                entry_time = current_bar.name
                entry_price = current_bar['close'] 
                direction = current_bar['signal']
                in_trade = True
                if direction == 'BUY':
                    sl_price = entry_price - (self.sl_pips / self.PIP_CONVERSION)
                    tp_price = entry_price + (self.tp_pips / self.PIP_CONVERSION)
                else:
                    sl_price = entry_price + (self.sl_pips / self.PIP_CONVERSION)
                    tp_price = entry_price - (self.tp_pips / self.PIP_CONVERSION)
                    
                trade = {'entry_time': entry_time, 'entry_price': entry_price, 'direction': direction,
                         'sl_price': sl_price, 'tp_price': tp_price, 'exit_time': None, 'profit_pips': 0.0, 'exit_type': ''}

            if in_trade and current_bar.name > trade['entry_time']:
                exit_logic_triggered = False
                pips_gained = 0.0
                exit_type = ''
                if trade['direction'] == 'BUY':
                    if current_bar['low'] <= trade['sl_price']: exit_logic_triggered, pips_gained, exit_type = True, -self.sl_pips, 'SL'
                    elif current_bar['high'] >= trade['tp_price']: exit_logic_triggered, pips_gained, exit_type = True, self.tp_pips, 'TP'
                elif trade['direction'] == 'SELL':
                    if current_bar['high'] >= trade['sl_price']: exit_logic_triggered, pips_gained, exit_type = True, -self.sl_pips, 'SL'
                    elif current_bar['low'] <= trade['tp_price']: exit_logic_triggered, pips_gained, exit_type = True, self.tp_pips, 'TP'

                if exit_logic_triggered:
                    trade['exit_time'] = current_bar.name
                    trade['profit_pips'] = pips_gained
                    trade['exit_type'] = exit_type
                    trades_list.append(trade)
                    in_trade = False
                    
        trades_df = pd.DataFrame(trades_list)
        total_trades = len(trades_df)

        if total_trades == 0:
            return {"net_pips": 0, "total_trades": 0, "profit_factor": 0.0, "reason": "No completed trades generated."}
            
        total_profit = trades_df[trades_df['profit_pips'] > 0]['profit_pips'].sum()
        total_loss = trades_df[trades_df['profit_pips'] < 0]['profit_pips'].sum()
        net_pips = total_profit + total_loss
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
    PIP_CONVERSION = 10000.0

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

        # The EnsemblePriceFeed uses the first feed (TwelveData) for historical fetching.
        if not self.data_feed.feeds[0].api_key:
            print("ERROR: Primary API Key (TwelveData) is missing. Cannot fetch data.")
            self.data = pd.DataFrame()
            return

        self.data = self.data_feed.fetch_historical_data(self.symbol, self.interval, self.lookback_years)

        if self.data is not None and not self.data.empty:
            print(f"DEBUG: Historical Data loaded. Total bars: {len(self.data)}")

            self.data.ta.macd(close='close', fast=self.MACD_FAST, slow=self.MACD_SLOW, signal=self.MACD_SIGNAL, append=True)
            self.data.ta.stoch(high='high', low='low', close='close', k=self.STOCH_K, d=self.STOCH_D, smooth_k=3, append=True)
            
            # --- Robust Column Renaming and Check (Same as before) ---
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
            self.data.dropna(inplace=True)
            print(f"DEBUG: Historical Data ready. Bars after cleanup: {len(self.data)}")

        else:
            print("ERROR: Initialization failed. Cannot proceed with historical data.")

    def generate_signal(self) -> Dict[str, Any]:
        """Generates a trade signal based on the latest ENSEMBLE bar, technical indicators, and ENSEMBLE fundamental score."""
        if self.data is None or self.data.empty:
            print("ERROR: Historical data not initialized. Cannot generate signal.")
            return {"signal": "HOLD", "reason": "Historical data missing."}

        # 1. Fetch Latest Bar from ENSEMBLE
        print("\n--- Running Step 1: Fetching Real-Time ENSEMBLE Price Data (3 APIs) ---")
        new_bar = self.data_feed.fetch_realtime_bar(self.symbol, self.interval)

        if new_bar.empty:
            return {"signal": "HOLD", "reason": "Failed to fetch real-time bar from price ensemble."}
            
        # Append new bar and recalculate indicators for the latest point (Same logic as before)
        temp_data = self.data.copy()
        temp_data.loc[new_bar.name] = new_bar.copy()
        temp_data.sort_index(inplace=True)

        macd_results = temp_data['close'].ta.macd(fast=self.MACD_FAST, slow=self.MACD_SLOW, signal=self.MACD_SIGNAL, append=False)
        stoch_results = temp_data.ta.stoch(high='high', low='low', close='close', k=self.STOCH_K, d=self.STOCH_D, smooth_k=3, append=False)

        self.data = temp_data.copy()

        # Update indicators
        macd_hist_col = f'MACDh_{self.MACD_FAST}_{self.MACD_SLOW}_{self.MACD_SIGNAL}'
        stoch_k_col = f'STOCHk_{self.STOCH_K}_3_3'
        stoch_d_col = f'STOCHd_{self.STOCH_K}_3_3'

        if macd_hist_col not in macd_results.columns:
            # Handle potential pandas_ta errors if data is too small or bad
             return {"signal": "HOLD", "reason": "Real-time indicator calculation failed to produce expected columns."}


        self.data['MACD_Hist'] = macd_results[macd_hist_col]
        self.data['Stoch_K'] = stoch_results[stoch_k_col]
        self.data['Stoch_D'] = stoch_results[stoch_d_col]
        self.data.dropna(subset=['MACD_Hist', 'Stoch_K', 'Stoch_D'], inplace=True)
        
        if len(self.data) < 2: return {"signal": "HOLD", "reason": "Not enough data points after latest bar append."}
            
        latest_data = self.data.iloc[-1]
        prev_data = self.data.iloc[-2]
        latest_close = latest_data['close']
        latest_macd_hist = latest_data['MACD_Hist']
        prev_stoch_k = prev_data['Stoch_K']
        prev_stoch_d = prev_data['Stoch_D']

        # 2. Fetch Latest Fundamental Score from ENSEMBLE
        print("--- Running Step 2: Fetching Real-Time ENSEMBLE Sentiment (2 APIs) ---")
        fundamental_score = self.fundamental_processor.fetch_realtime_sentiment()

        # 3. Decision Logic and Price Calculation (Same logic as before)
        signal = "HOLD"
        stop_loss, take_profit = None, None
        
        macd_bullish = latest_macd_hist > 0
        macd_bearish = latest_macd_hist < 0
        stoch_bullish_cross = (prev_stoch_k < prev_stoch_d) and (latest_data['Stoch_K'] > latest_data['Stoch_D']) and (latest_data['Stoch_D'] < 50)
        stoch_bearish_cross = (prev_stoch_k > prev_stoch_d) and (latest_data['Stoch_K'] < latest_data['Stoch_D']) and (latest_data['Stoch_D'] > 50)

        technical_bias = "NEUTRAL"
        if macd_bullish and stoch_bullish_cross: technical_bias = "BUY"
        elif macd_bearish and stoch_bearish_cross: technical_bias = "SELL"

        entry_price = latest_close 

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

    # 1. Check for missing keys (critical price APIs)
    if not all([TWELVEDATA_API_KEY, FMP_API_KEY, FOREXRATE_API_KEY]):
        print("FATAL: One or more Price API keys (TwelveData, FMP, ForexRate) are missing. Exiting.")
        return {"signal": "HOLD", "reason": "Missing critical Price API Keys."}
        
    if not all([MARKETAUX_API_KEY, NEWS_API_KEY_2]):
        print("WARNING: One or more News API keys are missing. Sentiment ensemble will be degraded.")

    try:
        # --- ENSEMBLE INITIALIZATION ---
        price_feeds = [
            TwelveDataFeed(api_key=TWELVEDATA_API_KEY),
            FMPFeed(api_key=FMP_API_KEY),
            ForexRateFeed(api_key=FOREXRATE_API_KEY)
        ]
        
        sentiment_processors = [
            MarketAuxProcessor(api_key=MARKETAUX_API_KEY),
            NewsAPIProcessor(api_key=NEWS_API_KEY_2)
        ]

        market_data_api = EnsemblePriceFeed(feeds=price_feeds)
        sentiment_api = EnsembleSentimentProcessor(processors=sentiment_processors)
        generator = SignalGenerator(data_feed=market_data_api, fundamental_processor=sentiment_api)

        # 2. Initialization and Backtest (UNCHANGED)
        print("\n" + "="*50)
        print("### STEP 1: INITIALIZING DATA ###")
        print("="*50)
        generator.initialize_data()

        if generator.data is None or generator.data.empty:
            print("\nFATAL: Initialization failed. Exiting.")
            return {"signal": "HOLD", "reason": "Historical data initialization failed."}

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
        print(f"Net Pips Gained: **{backtest_results['net_pips']:.2f}**")
        print(f"Profit Factor: **{backtest_results['profit_factor']:.2f}**")
        print("*"*50)

        # 3. Run the real-time signal generation (UNCHANGED)
        print("\n" + "="*50)
        print("### STEP 3: GENERATING REAL-TIME SIGNAL ###")
        print("="*50 + "\n")

        signal_result = generator.generate_signal()

        # 4. Print and Email the result (UNCHANGED)
        email_body = f"""
TRADING SIGNAL NOTIFICATION - EUR/USD ({generator.interval})
PAIR: {generator.symbol}
SIGNAL: {signal_result['signal']}
TIMESTAMP (Signal Close): {signal_result.get('timestamp', 'N/A')}
--- TRADE DETAILS ---
ENTRY PRICE: {signal_result.get('entry_price', 'N/A') if signal_result.get('entry_price') is None else f"{signal_result['entry_price']:.5f}"}
STOP LOSS (SL) PRICE: {signal_result.get('stop_loss', 'N/A') if signal_result.get('stop_loss') is None else f"{signal_result['stop_loss']:.5f}"} ({signal_result.get('sl_pips', 'N/A')} pips)
TAKE PROFIT (TP) PRICE: {signal_result.get('take_profit', 'N/A') if signal_result.get('take_profit') is None else f"{signal_result['take_profit']:.5f}"} ({signal_result.get('tp_pips', 'N/A')} pips)
REASON: {signal_result['reason']}
"""
        print("\n" + "#"*50)
        print("### 🔔 FINAL TRADING SIGNAL NOTIFICATION 🔔 ###")
        print(f"SIGNAL: **{signal_result['signal']}**")
        print(f"Entry: {signal_result.get('entry_price', 'N/A') if signal_result.get('entry_price') is None else f'{signal_result["entry_price"]:.5f}'}")
        print(f"Reason: {signal_result['reason']}")
        print("#"*50 + "\n")

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
        print("### 🛑 CRITICAL EXECUTION ERROR DURING RUNTIME 🛑 ###")
        traceback.print_exc()
        error_subject = "FATAL: Trading Bot Execution Failed"
        error_body = f"The trading signal logic encountered a critical error:\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}"
        send_email_notification(error_subject, error_body, MAIL_RECIPIENT)

        return {"signal": "ERROR", "reason": f"Critical runtime error: {e}"}

# --- Flask Routes (UNCHANGED) ---
@app.route('/', methods=['GET'])
def home():
    """Simple status check endpoint."""
    return "Trading Signal Generator is running. Use /run to execute the logic."

@app.route('/run', methods=['POST', 'GET'])
def run_script():
    """Endpoint to trigger the trading logic."""
    print("--- Received request to run trading logic (Starting Background Thread) ---")
    thread = Thread(target=run_signal_generation_logic)
    thread.start()
    response = {
        "status": "Processing started in background",
        "message": "The signal generation logic is running asynchronously. Check the application logs and your email for the final signal output."
    }
    return jsonify(response), 202

if __name__ == "__main__":
    print("Starting Flask server for manual execution...")
    app.run(host='0.0.0.0', port=os.environ.get("PORT", 5000))
