import requests
import pandas as pd
from ta.trend import MACD 
from ta.momentum import StochasticOscillator 
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any, Optional
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import traceback 
from threading import Thread 
import feedparser # Library for processing RSS feeds
from flask import Flask, jsonify, request

# --- Flask Application Setup ---
app = Flask(__name__)

# --- API Keys Configuration ---
TWELVESDATA_API_KEY = os.environ.get("TWELVESDATA_API_KEY") 
MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY") 
NEW_NEWS_API_KEY = os.environ.get("NEW_NEWS_API_KEY") 
TRADINGECONOMICS_API_KEY = os.environ.get("TRADINGECONOMICS_API_KEY") 

# --- RSS Feed URLs ---
REUTERS_RSS = "https://www.reuters.com/tools/rss" 

# --- Centralized Multi-Pair Configuration ---
# NOTE: The 'economic_countries' must be URL-safe slugs accepted by your Trading Economics API.
TRADING_PAIRS = {
    "EUR/USD": {
        "interval": "1h",
        "technical_config": "config_eu",
        "economic_countries": ["united states", "euro area"],
        "news_keywords": ["Eurozone inflation", "US Fed rate", "dollar", "euro"],
        "general_sentiment_threshold": 0.35,
        "sl_pips": 25.0,
        "tp_pips": 37.5 
    },
    "AUD/CAD": {
        "interval": "1h",
        "technical_config": "config_ac",
        "economic_countries": ["australia", "canada"],
        "news_keywords": ["RBA", "BoC", "Australian employment", "Canadian crude"],
        "general_sentiment_threshold": 0.40,
        "sl_pips": 20.0,
        "tp_pips": 30.0
    },
    "CAD/CHF": {
        "interval": "1h",
        "technical_config": "config_cc",
        "economic_countries": ["canada", "switzerland"],
        "news_keywords": ["BoC", "SNB", "Canadian crude", "Swiss safe-haven"],
        "general_sentiment_threshold": 0.30,
        "sl_pips": 20.0,
        "tp_pips": 25.0
    },
    "AUD/CHF": {
        "interval": "1h",
        "technical_config": "config_ah",
        "economic_countries": ["australia", "switzerland"],
        "news_keywords": ["RBA", "SNB", "commodity prices", "Swiss franc"],
        "general_sentiment_threshold": 0.45,
        "sl_pips": 25.0,
        "tp_pips": 40.0
    },
    "AUD/JPY": {
        "interval": "1h",
        "technical_config": "config_aj",
        "economic_countries": ["australia", "japan"],
        "news_keywords": ["RBA", "BoJ", "Yen intervention", "Aussie commodity"],
        "general_sentiment_threshold": 0.38,
        "sl_pips": 30.0,
        "tp_pips": 45.0
    },
    "EUR/CAD": {
        "interval": "1h",
        "technical_config": "config_ec",
        "economic_countries": ["euro area", "canada"],
        "news_keywords": ["ECB", "BoC", "Eurozone GDP", "Canadian inflation"],
        "general_sentiment_threshold": 0.35,
        "sl_pips": 25.0,
        "tp_pips": 37.5
    },
    "EUR/AUD": {
        "interval": "1h",
        "technical_config": "config_ea",
        "economic_countries": ["euro area", "australia"],
        "news_keywords": ["ECB", "RBA", "European growth", "Aussie housing"],
        "general_sentiment_threshold": 0.42,
        "sl_pips": 30.0,
        "tp_pips": 45.0
    },
    "CAD/JPY": {
        "interval": "1h",
        "technical_config": "config_cj",
        "economic_countries": ["canada", "japan"],
        "news_keywords": ["BoC", "BoJ", "Crude oil", "Yen safe-haven"],
        "general_sentiment_threshold": 0.30,
        "sl_pips": 20.0,
        "tp_pips": 30.0
    },
    "CHF/JPY": {
        "interval": "1h",
        "technical_config": "config_hj",
        "economic_countries": ["switzerland", "japan"],
        "news_keywords": ["SNB", "BoJ", "Yen intervention", "Swiss inflation"],
        "general_sentiment_threshold": 0.25,
        "sl_pips": 20.0,
        "tp_pips": 25.0
    },
    "AUD/USD": {
        "interval": "1h",
        "technical_config": "config_au",
        "economic_countries": ["australia", "united states"],
        "news_keywords": ["RBA", "Fed", "Australian CPI", "US NFP"],
        "general_sentiment_threshold": 0.35,
        "sl_pips": 25.0,
        "tp_pips": 37.5
    },
}


# --- EMAIL Configuration (Retained) ---
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
MAIL_SERVER = os.environ.get("MAIL_SERVER", 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get("MAIL_PORT", 465))
MAIL_SENDER = os.environ.get("MAIL_SENDER")
MAIL_RECIPIENT = os.environ.get("MAIL_RECIPIENT")

# --- 0. Utility Functions (Retained) ---
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
    except Exception as e:
        print(f"FATAL EMAIL ERROR: Failed to send email: {e}")

# --- 1. Data Feed (Twelves Data) ---
class TwelvesDataFeed:
    BASE_URL = "https://api.twelvedata.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_cache = {}

    def fetch_historical_data(self, symbol: str, interval: str, lookback_years: int) -> pd.DataFrame:
        """ Fetches historical Forex data using Twelves Data. """
        print(f"DEBUG: Fetching historical {interval} data for {symbol} (Twelves Data)...")

        # Simplified cache check for brevity
        if symbol in self.data_cache:
            return self.data_cache[symbol].copy()

        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": 5000,
            "apikey": self.api_key
        }
        url = f"{self.BASE_URL}/time_series"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'error' or 'values' not in data:
                print(f"Twelves Data API Error for {symbol}: {data.get('message', 'No historical data returned.')}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data['values'])
            df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            
            self.data_cache[symbol] = df[['open', 'high', 'low', 'close']]
            print(f"DEBUG: {symbol} Historical Data loaded. Total bars: {len(df)}")
            return self.data_cache[symbol].copy()

        except requests.exceptions.RequestException as e:
            print(f"Twelves Data historical request for {symbol} failed: {e}")
            return pd.DataFrame()

    def fetch_realtime_bar(self, symbol: str, interval: str) -> pd.Series:
        """ Fetches the latest intraday bar data using Twelves Data. """
        print(f"DEBUG: Fetching real-time bar for {symbol} (Twelves Data)...")

        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": 1,
            "apikey": self.api_key
        }
        url = f"{self.BASE_URL}/time_series"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'error' or 'values' not in data or not data['values']:
                error_msg = data.get('message', 'No real-time bar returned.')
                print(f"ERROR fetching real-time bar for {symbol}: {error_msg}")
                return pd.Series()

            latest_data = data['values'][0]
            latest_time = latest_data['datetime']
            
            latest_bar = pd.Series({
                'open': float(latest_data['open']),
                'high': float(latest_data['high']),
                'low': float(latest_data['low']),
                'close': float(latest_data['close'])
            }, name=pd.to_datetime(latest_time))

            return latest_bar
        except Exception as e:
            print(f"Failed to process real-time data for {symbol}: {e}")
            return pd.Series()

# --- MultiSourceSentimentProcessor (UPDATED) ---
class MultiSourceSentimentProcessor:
    """Combines sentiment from MarketAux and the new News API."""
    MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
    NEWS_API_URL = "https://newsapi.org/v2/everything" 

    def __init__(self, marketaux_key: str, news_api_key: str):
        self.marketaux_key = marketaux_key
        self.news_api_key = news_api_key

    def _get_marketaux_sentiment(self, symbol: str, search_query: str) -> List[float]:
        if not self.marketaux_key:
            return []

        params = {
            "api_token": self.marketaux_key,
            "search": f"{symbol} OR {search_query}", # Search by symbol and keywords
            "filter_entities": "true",
            "language": "en",
            "limit": 10
        }
        try:
            response = requests.get(self.MARKETAUX_URL, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get('data', [])
            sentiments = []
            
            base_currency = symbol.split('/')[0]
            counter_currency = symbol.split('/')[1]
            
            for article in articles:
                entities = article.get('entities', [])
                for entity in entities:
                    # Focus on entities that match the base or counter currency
                    entity_symbol = entity.get('symbol')
                    if entity_symbol in [base_currency, counter_currency] and 'sentiment_score' in entity:
                        score = entity['sentiment_score']
                        # Counter-currency (like USD) sentiment is inverted for the pair.
                        sentiments.append(score if entity_symbol == base_currency else -score) 
            
            return sentiments
        except Exception:
            return []

    def _get_newsapi_sentiment(self, symbol: str, search_query: str) -> List[float]:
        if not self.news_api_key:
            return []
            
        params = {
            "apiKey": self.news_api_key,
            "q": f"({symbol}) AND ({search_query})", # Target query to the specific pair and keywords
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }
        try:
            response = requests.get(self.NEWS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            
            positive_words = ['gains', 'rise', 'strong', 'growth', 'up', 'outperform']
            negative_words = ['drops', 'falls', 'weak', 'plunge', 'down', 'underperform']
            
            sentiment_scores = []
            for article in articles:
                title = article.get('title', '').lower()
                pos_count = sum(1 for word in positive_words if word in title)
                neg_count = sum(1 for word in negative_words if word in title)
                
                if pos_count > neg_count:
                    sentiment_scores.append(0.5)
                elif neg_count > pos_count:
                    sentiment_scores.append(-0.5)
                else:
                    sentiment_scores.append(0.0)
            
            return sentiment_scores
        except Exception:
            return []

    def fetch_realtime_sentiment(self, symbol: str, search_query: str) -> float:
        """Combines sentiment from all available API sources."""
        all_sentiments = []
        all_sentiments.extend(self._get_marketaux_sentiment(symbol, search_query))
        all_sentiments.extend(self._get_newsapi_sentiment(symbol, search_query))
        
        if not all_sentiments:
            return 0.0

        avg_sentiment = sum(all_sentiments) / len(all_sentiments)
        return avg_sentiment

# --- 2. News and Fundamental Processor (UPGRADED) ---
class NewsProcessor:
    HIGH_IMPACT_EVENTS = ['Nonfarm Payrolls', 'Interest Rate Decision', 'CPI', 'GDP', 'Unemployment Rate', 'Retail Sales']
    SURPRISE_MULTIPLIER = 5.0 

    def __init__(self):
        # Existing API-based sentiment processors
        self.sentiment_processor = MultiSourceSentimentProcessor(
            marketaux_key=MARKETAUX_API_KEY,
            news_api_key=NEW_NEWS_API_KEY
        )

    def _calculate_surprise(self, actual_str: str, forecast_str: str, base_currency: str, counter_currency: str, country: str) -> Optional[float]:
        """Converts strings to float, calculates deviation, and applies polarity for the pair."""
        try:
            def clean_value(s):
                if not s: return None
                s = str(s).replace(',', '').replace('%', '').strip()
                return float(s)

            actual = clean_value(actual_str)
            forecast = clean_value(forecast_str)

            if actual is None or forecast is None or forecast == 0:
                return None
            
            surprise_percent = (actual - forecast)
            score = np.clip(surprise_percent * self.SURPRISE_MULTIPLIER, -1.0, 1.0)
            
            # Polarity Adjustment: If the event is from the counter currency's country, invert the score.
            country_map = {
                "united states": "USD", "euro area": "EUR", "australia": "AUD",
                "canada": "CAD", "switzerland": "CHF", "japan": "JPY"
            }
            
            currency_from_event = country_map.get(country.lower())

            if currency_from_event == counter_currency:
                score *= -1 # Invert score for counter-currency strength
                
            return score
            
        except Exception:
            return None

    def fetch_economic_calendar_surprise(self, symbol: str, countries: List[str]) -> Dict[str, Any]:
        """Fetches and calculates a combined News Surprise Score for the given pair's countries."""
        print(f"DEBUG: Checking Trading Economics API for High-Impact news surprise for {symbol}...")

        base_currency = symbol.split('/')[0]
        counter_currency = symbol.split('/')[1]
        
        if not TRADINGECONOMICS_API_KEY:
            print("ERROR: TRADINGECONOMICS_API_KEY missing. Skipping calendar fetch.")
            return {"score": 0.0, "reason": "API Key missing."}

        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Dynamic URL construction using the pair's countries
        country_list_str = ','.join(countries).replace(' ', '%20')
        url = (
            f"https://api.tradingeconomics.com/calendar/country/{country_list_str}"
            f"?c={TRADINGECONOMICS_API_KEY}&d1={start_date}&d2={end_date}"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"ERROR fetching Trading Economics data for {symbol}: {e}")
            return {"score": 0.0, "reason": f"API fetch failed: {e}"}

        latest_surprise_score = 0.0
        active_event_details = "None"
        time_limit = datetime.utcnow() - timedelta(hours=1)
        
        for event in data:
            if event.get('Importance') not in ['High', 'Medium']: continue
            
            event_date_str = event.get('Date')
            event_country_lower = event.get('Country', '').lower()
            if not event_date_str or event_country_lower not in countries: continue

            try:
                event_date = datetime.strptime(event_date_str[:19], '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                continue

            if event_date < time_limit: continue
            
            event_name = event.get('Event')
            actual = event.get('Actual')
            forecast = event.get('Forecast')
            
            if any(keyword in event_name for keyword in self.HIGH_IMPACT_EVENTS) and actual and forecast:
                score = self._calculate_surprise(actual, forecast, base_currency, counter_currency, event_country_lower)
                
                if score is not None and abs(score) > abs(latest_surprise_score):
                    latest_surprise_score = score
                    active_event_details = f"{event_name} ({event.get('Country')}): Actual={actual}, Forecast={forecast}, Score={latest_surprise_score:.2f}"

        return {
            "score": latest_surprise_score,
            "reason": active_event_details
        }
        
    def fetch_general_news_sentiment(self, symbol: str, keywords: List[str]) -> float:
        """Fetches general market sentiment from multiple sources including Reuters RSS and APIs."""
        search_query = ' OR '.join(keywords)
        print(f"DEBUG: Fetching General Sentiment for {symbol} using keywords: ({search_query})...")
        
        all_sentiment_scores = []
        
        # 1. Existing API-based sentiment (MarketAux/News API)
        api_sentiment = self.sentiment_processor.fetch_realtime_sentiment(symbol, search_query)
        if api_sentiment != 0.0:
            all_sentiment_scores.append(api_sentiment)
            
        # 2. Reuters RSS Feed (Filtered by keywords)
        reuters_feed = feedparser.parse(REUTERS_RSS)
        reuters_scores = []
        
        positive_words = ['gains', 'rise', 'strong', 'growth', 'up', 'outperform', 'rally', 'beats']
        negative_words = ['drops', 'falls', 'weak', 'plunge', 'down', 'underperform', 'slips', 'misses']

        keyword_check = [k.lower() for k in keywords + [symbol.replace('/', ' ')]]

        for entry in reuters_feed.entries[:20]:
            title = entry.title.lower()
            
            # Filter for relevance to the current pair
            if not any(k in title for k in keyword_check):
                continue
            
            pos_count = sum(1 for word in positive_words if word in title)
            neg_count = sum(1 for word in negative_words if word in title)
            
            if pos_count > neg_count:
                reuters_scores.append(0.2)
            elif neg_count > pos_count:
                reuters_scores.append(-0.2)

        if reuters_scores:
            all_sentiment_scores.append(sum(reuters_scores) / len(reuters_scores))
            
        if not all_sentiment_scores:
            return 0.0
            
        combined_score = sum(all_sentiment_scores) / len(all_sentiment_scores)
        print(f"DEBUG: Combined General Sentiment: {combined_score:.3f} (from {len(all_sentiment_scores)} calculated scores)")
        return combined_score

# --- 3. Trading Strategy and Signal Generation ---
class Backtester:
    def __init__(self, data: pd.DataFrame, fundamental_threshold: float, sl_pips: float, tp_pips: float):
        self.data = data.copy()
        self.fundamental_threshold = fundamental_threshold
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.PIP_CONVERSION = 10000.0 if not self.is_jpy_pair(data) else 100.0
        
    def is_jpy_pair(self, data: pd.DataFrame) -> bool:
        """Heuristic check for JPY pairs based on price scale (if symbol isn't available)."""
        if data.empty:
            return False
        # If price is typically over 100, assume JPY pair (or similar high-value pair)
        return data['close'].iloc[-1] > 10.0 

    def calculate_technical_indicators(self):
        macd_instance = MACD(close=self.data['close'], window_fast=12, window_slow=26, window_sign=9, fillna=False) 
        stoch_instance = StochasticOscillator(high=self.data['high'], low=self.data['low'], close=self.data['close'], window=14, smooth_window=3, fillna=False) 
        self.data['MACD_Hist'] = macd_instance.macd_diff() 
        self.data['Stoch_K'] = stoch_instance.stoch() 
        self.data['Stoch_D'] = stoch_instance.stoch_signal() 
        self.data.dropna(inplace=True)
        
    def generate_historical_signals(self) -> pd.DataFrame:
        self.calculate_technical_indicators()
        np.random.seed(42)
        price_diff = self.data['close'].diff().fillna(0)
        simulated_sentiment = np.clip(price_diff.rolling(window=10).mean().fillna(0) * 5 + np.random.normal(0, 0.1, len(self.data)), -1, 1)
        self.data['sentiment'] = simulated_sentiment
        signals = pd.Series(index=self.data.index, dtype=str)
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
        if self.data.empty:
            return {"net_pips": 0, "total_trades": 0, "winning_trades": 0, "profit_factor": 0.0, "reason": "Initial data is empty."}
            
        self.calculate_technical_indicators()
        self.generate_historical_signals()

        if self.data.empty or 'signal' not in self.data.columns:
            return {"net_pips": 0, "total_trades": 0, "winning_trades": 0, "profit_factor": 0.0, "reason": "Data or signal generation failed after indicators/signals."}
            
        trades_list = []
        in_trade = False

        # ... (Rest of backtest logic remains the same, using self.PIP_CONVERSION)
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

            if in_trade and current_bar.name > trade['entry_time']:

                exit_logic_triggered = False
                pips_gained = 0.0
                exit_type = ''

                if trade['direction'] == 'BUY':
                    if current_bar['low'] <= trade['sl_price']:
                        exit_logic_triggered = True
                        pips_gained = -self.sl_pips 
                        exit_type = 'SL'
                    elif current_bar['high'] >= trade['tp_price']:
                        exit_logic_triggered = True
                        pips_gained = self.tp_pips 
                        exit_type = 'TP'

                elif trade['direction'] == 'SELL':
                    if current_bar['high'] >= trade['sl_price']:
                        exit_logic_triggered = True
                        pips_gained = -self.sl_pips 
                        exit_type = 'SL'
                    elif current_bar['low'] <= trade['tp_price']:
                        exit_logic_triggered = True
                        pips_gained = self.tp_pips 
                        exit_type = 'TP'

                if exit_logic_triggered:
                    trade['exit_time'] = current_bar.name
                    trade['profit_pips'] = pips_gained
                    trade['exit_type'] = exit_type
                    trades_list.append(trade)
                    in_trade = False 
                    
        trades_df = pd.DataFrame(trades_list)
        total_trades = len(trades_df)

        if total_trades == 0:
            return {"net_pips": 0, "total_trades": 0, "winning_trades": 0, "profit_factor": 0.0, "reason": "No completed trades generated."}
            
        total_profit = trades_df[trades_df['profit_pips'] > 0]['profit_pips'].sum()
        total_loss = trades_df[trades_df['profit_pips'] < 0]['profit_pips'].sum()
        net_pips = total_profit + total_loss

        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else np.inf
        return {
            "net_pips": round(net_pips, 2),
            "total_trades": total_trades,
            "winning_trades": len(trades_df[trades_df['profit_pips'] > 0]),
            "profit_factor": round(profit_factor, 2),
            "reason": f"Backtested over {len(self.data)} bars. SL: {self.sl_pips} pips, TP: {self.tp_pips} pips (R:R 1:{round(self.tp_pips/self.sl_pips, 1)}).",
            "pip_conversion": self.PIP_CONVERSION
        }


class SignalGenerator:
    # --- Strategy Tuning Parameters (Kept as default) ---
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    STOCH_K = 14
    STOCH_D = 3
    SURPRISE_SCORE_THRESHOLD = 0.5   # Threshold for Economic Calendar Surprise (triggers News Trade)
    PIP_CONVERSION_DEFAULT = 10000.0

    def __init__(self, data_feed: TwelvesDataFeed, news_processor: NewsProcessor, symbol: str, pair_config: Dict[str, Any]):
        self.data_feed = data_feed
        self.news_processor = news_processor 
        self.symbol = symbol
        self.interval = pair_config["interval"]
        self.lookback_years = 2
        self.data: Optional[pd.DataFrame] = None
        
        # Dynamic Risk Management from config
        self.GENERAL_SENTIMENT_THRESHOLD = pair_config["general_sentiment_threshold"]
        self.STOP_LOSS_PIPS = pair_config["sl_pips"]
        self.TAKE_PROFIT_PIPS = pair_config["tp_pips"]
        # PIP conversion will be determined dynamically
        self.PIP_CONVERSION = self.PIP_CONVERSION_DEFAULT 

    def initialize_data(self):
        """Fetches historical data to build the model's foundation."""
        print(f"\n--- Initializing Historical Data for {self.symbol} ---")
        if not self.data_feed.api_key:
            print("ERROR: Data Feed API Key is missing. Cannot fetch data.")
            self.data = pd.DataFrame()
            return
            
        self.data = self.data_feed.fetch_historical_data(self.symbol, self.interval, self.lookback_years)
        if self.data is not None and not self.data.empty:
            # Determine PIP conversion dynamically (Heuristic: JPY pairs usually have prices > 100)
            if self.data['close'].iloc[-1] > 10.0:
                 self.PIP_CONVERSION = 100.0
            else:
                 self.PIP_CONVERSION = 10000.0
                 
            macd_instance = MACD(close=self.data['close'], window_fast=self.MACD_FAST, window_slow=self.MACD_SLOW, window_sign=self.MACD_SIGNAL, fillna=False) 
            stoch_instance = StochasticOscillator(high=self.data['high'], low=self.data['low'], close=self.data['close'], window=self.STOCH_K, smooth_window=self.STOCH_D, fillna=False) 
            self.data['MACD_Hist'] = macd_instance.macd_diff() 
            self.data['Stoch_K'] = stoch_instance.stoch() 
            self.data['Stoch_D'] = stoch_instance.stoch_signal() 
            
            if not all(col in self.data.columns for col in ['MACD_Hist', 'Stoch_K', 'Stoch_D']): 
                self.data = pd.DataFrame()
                return
            self.data.dropna(inplace=True)
            print(f"DEBUG: {self.symbol} Historical Data ready. Bars after cleanup: {len(self.data)}. PIPs Conversion: {self.PIP_CONVERSION}")
        else:
            print(f"ERROR: {self.symbol} Initialization failed. Cannot proceed with historical data.")

    def _determine_technical_bias(self, latest_data: pd.Series, prev_data: pd.Series) -> str:
        """Determines technical bias based on MACD and Stochastic."""
        latest_macd_hist = latest_data['MACD_Hist']
        prev_stoch_k = prev_data['Stoch_K']
        prev_stoch_d = prev_data['Stoch_D']

        macd_bullish = latest_macd_hist > 0
        macd_bearish = latest_macd_hist < 0
        stoch_bullish_cross = (prev_stoch_k < prev_stoch_d) and \
                              (latest_data['Stoch_K'] > latest_data['Stoch_D']) and \
                              (latest_data['Stoch_D'] < 50)
        stoch_bearish_cross = (prev_stoch_k > prev_stoch_d) and \
                              (latest_data['Stoch_K'] < latest_data['Stoch_D']) and \
                              (latest_data['Stoch_D'] > 50)

        if macd_bullish and stoch_bullish_cross:
            return "BUY"
        elif macd_bearish and stoch_bearish_cross:
            return "SELL"
        else:
            return "NEUTRAL"

    def generate_signal(self, pair_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a trade signal using Technical + Economic Calendar Surprise + General Sentiment.
        """
        if self.data is None or self.data.empty:
            return {"signal": "HOLD", "reason": "Historical data missing."}

        # 1. Prepare Latest Price Data
        new_bar = self.data_feed.fetch_realtime_bar(self.symbol, self.interval)
        if new_bar.empty:
            return {"signal": "HOLD", "reason": f"{self.symbol}: Failed to fetch real-time bar."}
            
        temp_data = self.data.copy()
        temp_data.loc[new_bar.name] = new_bar.copy()
        temp_data.sort_index(inplace=True)
        # Recalculate indicators on the combined dataset
        macd_instance = MACD(close=temp_data['close'], window_fast=self.MACD_FAST, window_slow=self.MACD_SLOW, window_sign=self.MACD_SIGNAL, fillna=False) 
        stoch_instance = StochasticOscillator(high=temp_data['high'], low=temp_data['low'], close=temp_data['close'], window=self.STOCH_K, smooth_window=self.STOCH_D, fillna=False) 
        self.data = temp_data.copy()
        self.data['MACD_Hist'] = macd_instance.macd_diff() 
        self.data['Stoch_K'] = stoch_instance.stoch() 
        self.data['Stoch_D'] = stoch_instance.stoch_signal() 
        self.data.dropna(subset=['MACD_Hist', 'Stoch_K', 'Stoch_D'], inplace=True)
        
        if len(self.data) < 2:
            return {"signal": "HOLD", "reason": f"{self.symbol}: Not enough data points after latest bar append."}
            
        latest_data = self.data.iloc[-1]
        prev_data = self.data.iloc[-2]
        latest_close = latest_data['close']

        # 2. Determine Technical Bias
        technical_bias = self._determine_technical_bias(latest_data, prev_data)
            
        # 3. Fetch Fundamental Data (using pair-specific config)
        print(f"\n--- Running Step 3.2.1: Checking Economic Calendar Surprise for {self.symbol} ---")
        calendar_data = self.news_processor.fetch_economic_calendar_surprise(self.symbol, pair_config["economic_countries"])
        surprise_score = calendar_data['score']
        surprise_reason = calendar_data['reason']
        
        print(f"\n--- Running Step 3.2.2: Fetching General News Sentiment for {self.symbol} ---")
        search_query = ' OR '.join(pair_config["news_keywords"])
        general_sentiment = self.news_processor.fetch_general_news_sentiment(self.symbol, pair_config["news_keywords"])
        
        # 4. Final Strategy Decision
        signal = "HOLD"
        stop_loss, take_profit = None, None
        entry_price = latest_close 
        reason = ""
        
        # --- LAYER 1: NEWS TRADING MODE (Economic Surprise Override) ---
        if abs(surprise_score) >= self.SURPRISE_SCORE_THRESHOLD:
            signal = "BUY" if surprise_score > 0 else "SELL"
            reason = f"***NEWS TRADE OVERRIDE***: Extreme Economic Surprise detected. Signal based on: {surprise_reason}"
        
        # --- LAYER 2: CONFIRMATION TRADING MODE (Tech + General Sentiment) ---
        elif technical_bias != "NEUTRAL":
            sentiment_direction = "BUY" if general_sentiment >= self.GENERAL_SENTIMENT_THRESHOLD else \
                                  "SELL" if general_sentiment <= -self.GENERAL_SENTIMENT_THRESHOLD else \
                                  "NEUTRAL"
            
            if technical_bias == sentiment_direction:
                signal = technical_bias
                reason = f"**CONFIRMED {signal}**: Technical signal ({technical_bias}) aligns with strong General Sentiment ({general_sentiment:.3f})."
                
            elif sentiment_direction == "NEUTRAL":
                signal = technical_bias
                reason = f"**{signal} (Caution)**: Technical signal is clear, but General Sentiment is neutral ({general_sentiment:.3f})."
                
            else: # Technical and Sentiment Contradict
                signal = "HOLD"
                reason = f"HOLD: Technical bias ({technical_bias}) strongly contradicts General Sentiment ({sentiment_direction} - {general_sentiment:.3f})."

        # --- LAYER 3: DEFAULT HOLD ---
        else:
            signal = "HOLD"
            reason = f"HOLD: Technical bias is NEUTRAL, and General Sentiment is neutral/conflicting ({general_sentiment:.3f}). No high-impact news."

        # 5. Price Calculation
        if signal == "BUY":
            stop_loss = entry_price - (self.STOP_LOSS_PIPS / self.PIP_CONVERSION)
            take_profit = entry_price + (self.TAKE_PROFIT_PIPS / self.PIP_CONVERSION)
            
        elif signal == "SELL":
            stop_loss = entry_price + (self.STOP_LOSS_PIPS / self.PIP_CONVERSION)
            take_profit = entry_price - (self.TAKE_PROFIT_PIPS / self.PIP_CONVERSION)

        return {
            "symbol": self.symbol,
            "signal": signal,
            "reason": reason,
            "timestamp": latest_data.name.strftime('%Y-%m-%d %H:%M:%S') if signal != "HOLD" else "N/A",
            "entry_price": entry_price if signal != "HOLD" else None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_pips": self.STOP_LOSS_PIPS,
            "tp_pips": self.TAKE_PROFIT_PIPS
        }

# --- Main Execution Function (UPDATED FOR MULTI-PAIR LOOP) ---
def run_signal_generation_logic():
    """Initializes and runs the signal generation, backtesting, and email process for ALL pairs."""
    
    global TRADING_PAIRS # Access the global configuration

    if not TWELVESDATA_API_KEY:
        print("FATAL: TWELVESDATA_API_KEY is not set. Cannot fetch data.")
        return {"signal": "HOLD", "reason": "Missing primary data feed API Key."}
    
    market_data_api = TwelvesDataFeed(api_key=TWELVESDATA_API_KEY)
    news_processor = NewsProcessor()
    
    all_results = {}

    for symbol, config in TRADING_PAIRS.items():
        try:
            print("\n" + "="*80)
            print(f"### STARTING FULL TRADING CYCLE FOR {symbol} ###")
            print("="*80)

            generator = SignalGenerator(
                data_feed=market_data_api, 
                news_processor=news_processor, 
                symbol=symbol,
                pair_config=config
            )
            
            # 1. Initialize historical data (runs once per pair)
            print("\n" + "="*50)
            print(f"### STEP 1: INITIALIZING HISTORICAL DATA for {symbol} ###")
            print("="*50)
            generator.initialize_data()

            if generator.data is None or generator.data.empty:
                print(f"\nFATAL: {symbol} Initialization failed. Skipping pair.")
                all_results[symbol] = {"signal": "HOLD", "reason": "Historical data initialization failed."}
                continue
            
            # 2. Backtest the predicted signals
            print("\n" + "="*50)
            print(f"### STEP 2: RUNNING BACKTESTING for {symbol} ###")
            print("="*50)

            backtester = Backtester(
                data=generator.data.copy(),
                fundamental_threshold=generator.GENERAL_SENTIMENT_THRESHOLD, 
                sl_pips=generator.STOP_LOSS_PIPS,
                tp_pips=generator.TAKE_PROFIT_PIPS
            )
            backtest_results = backtester.run_backtest()
            
            print("\n" + "*"*50)
            print(f"### BACKTEST RESULTS FOR {symbol} ###")
            print(f"Total Trades Analyzed: {backtest_results.get('total_trades', 0)}")
            print(f"Net Pips Gained: **{backtest_results.get('net_pips', 0.0):.2f}**")
            print(f"Profit Factor: **{backtest_results.get('profit_factor', 0.0):.2f}**")
            print(f"Details: {backtest_results.get('reason', 'Unknown failure.')}")
            print("*"*50)
            
            # ⚠️ CRITICAL FIX: Rate Limit Delay (Per Pair)
            print("\n" + "="*50)
            print(f"### PAUSING FOR 60 SECONDS TO AVOID TWELVES DATA RATE LIMIT for {symbol} ###")
            print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(60)
            print(f"Resuming execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50 + "\n")

            # 3. Run the real-time signal generation
            print("\n" + "="*50)
            print(f"### STEP 3: GENERATING UPGRADED REAL-TIME SIGNAL for {symbol} ###")
            print("="*50 + "\n")

            signal_result = generator.generate_signal(config)
            all_results[symbol] = signal_result

            # 4. Print and Email the result (Updated for multi-pair)

            email_body = f"""
TRADING SIGNAL NOTIFICATION - {signal_result['symbol']} ({generator.interval})
PAIR: {signal_result['symbol']}
SIGNAL: {signal_result['signal']}
TIMESTAMP (Signal Close): {signal_result.get('timestamp', 'N/A')}
--- TRADE DETAILS ---
ENTRY PRICE: {signal_result.get('entry_price', 'N/A') if signal_result.get('entry_price') is None else f"{signal_result['entry_price']:.5f}"}
STOP LOSS (SL) PRICE: {signal_result.get('stop_loss', 'N/A') if signal_result.get('stop_loss') is None else f"{signal_result['stop_loss']:.5f}"} ({signal_result.get('sl_pips', 'N/A')} pips)
TAKE PROFIT (TP) PRICE: {signal_result.get('take_profit', 'N/A') if signal_result.get('take_profit') is None else f"{signal_result['take_profit']:.5f}"} ({signal_result.get('tp_pips', 'N/A')} pips)
Risk/Reward Ratio: 1:{round(signal_result['tp_pips']/signal_result['sl_pips'], 1)}
REASON: {signal_result['reason']}

--- BACKTEST PERFORMANCE ---
Net Pips Gained: {backtest_results.get('net_pips', 0.0):.2f}
Profit Factor: {backtest_results.get('profit_factor', 0.0):.2f}
"""
            # Console output
            print("\n" + "#"*50)
            print("### 🔔 FINAL TRADING SIGNAL NOTIFICATION 🔔 ###")
            print(f"PAIR: {signal_result['symbol']}")
            print(f"SIGNAL: **{signal_result['signal']}**")
            print(f"Reason: {signal_result['reason']}")
            print("#"*50 + "\n")
            
            if signal_result['signal'] != "HOLD":
                subject = f"**{signal_result['signal']}** Signal Generated for {signal_result['symbol']}!"
            else:
                subject = f"HOLD Signal Generated for {signal_result['symbol']}."

            send_email_notification(
                subject=subject,
                body=email_body,
                recipient=MAIL_RECIPIENT
            )

        except Exception as e:
            error_msg = f"Critical error for {symbol}: {e}"
            print("\n" + "!"*50)
            print(f"### 🛑 CRITICAL EXECUTION ERROR FOR {symbol} 🛑 ###")
            print(error_msg)
            print("\n--- Detailed Stack Trace ---")
            traceback.print_exc()
            print("!"*50 + "\n")

            all_results[symbol] = {"signal": "ERROR", "reason": error_msg}
            error_subject = f"FATAL: Bot Failed for {symbol}"
            error_body = f"The trading signal logic encountered a critical error for {symbol}:\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}"
            send_email_notification(error_subject, error_body, MAIL_RECIPIENT)
            
    return all_results

# --- Flask Routes (Retained) ---
@app.route('/', methods=['GET'])
def home():
    """Simple status check endpoint."""
    return "Trading Signal Generator is running. Use /run to execute the logic."
@app.route('/run', methods=['POST', 'GET'])
def run_script():
    """
    Endpoint to trigger the trading logic.
    """
    print("--- Received request to run trading logic (Starting Background Thread) ---")

    thread = Thread(target=run_signal_generation_logic)
    thread.start()

    response = {
        "status": "Processing started in background",
        "message": f"The signal generation logic is running asynchronously for {len(TRADING_PAIRS)} pairs. Check the application logs and your email for the final signal output."
    }

    return jsonify(response), 202
if __name__ == "__main__":
    print("Starting Flask server for manual execution...")
    app.run(host='0.0.0.0', port=os.environ.get("PORT", 5000))
