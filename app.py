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
import feedparser 
from flask import Flask, jsonify, request

# --- Flask Application Setup ---
app = Flask(__name__)

# --- API Keys Configuration ---
TWELVESDATA_API_KEY = os.environ.get("TWELVESDATA_API_KEY") 
MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY") 
NEW_NEWS_API_KEY = os.environ.get("NEW_NEWS_API_KEY") 

# 💡 CHANGE 1: Using the provided FMP key (A0xuQ94tqyjfAKitVIGoNKPNnBX2K0JT)
# Note: Reusing the TRADINGECONOMICS_API_KEY variable to store the FMP key for minimal impact.
TRADINGECONOMICS_API_KEY = "A0xuQ94tqyjfAKitVIGoNKPNnBX2K0JT" 

# --- RSS Feed URLs ---
REUTERS_RSS = "https://www.reuters.com/tools/rss" 

# --- API Endpoints ---
# 💡 CHANGE 2: New FMP Economic Calendar Endpoint
FMP_CALENDAR_URL = "https://financialmodelingprep.com/api/v3/economic_calendar"

# --- EMAIL Configuration (Retained) ---
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
MAIL_RECIPIENT = os.environ.get("MAIL_RECIPIENT", "jamesnwoke880@gmail.com") 

# --- Trading Configuration ---
# 💡 CHANGE 3: Decrease general_sentiment_threshold to 0.15 for both pairs
TRADING_PAIRS = {
    "EUR/USD": {
        "timeframe": "1h", 
        "historical_bars": 100,
        "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCH_WINDOW": 14, "STOCH_SMOOTH": 3,
        "SURPRISE_SCORE_THRESHOLD": 0.5,
        "general_sentiment_threshold": 0.15, # DECREASED to increase signals
        "sl_pips": 25.0,
        "tp_pips": 37.5 
    },
    "GBP/USD": {
        "timeframe": "1h",
        "historical_bars": 100,
        "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCH_WINDOW": 14, "STOCH_SMOOTH": 3,
        "SURPRISE_SCORE_THRESHOLD": 0.5,
        "general_sentiment_threshold": 0.15, # DECREASED to increase signals
        "sl_pips": 25.0,
        "tp_pips": 37.5 
    }
}
# --- Helper Functions ---

def send_email_notification(subject: str, body: str, recipient: str):
    """Sends an email notification."""
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        print("FATAL EMAIL ERROR: MAIL_USERNAME or MAIL_PASSWORD is not set.")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = MAIL_USERNAME
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
        server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_USERNAME, recipient, msg.as_string())
        server.quit()
        print(f"DEBUG: Email sent successfully to {recipient} with subject: {subject}")
    except Exception as e:
        print(f"FATAL EMAIL ERROR: Failed to send email: {e}")

def get_realtime_bar(pair: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Fetches real-time price data using Twelves Data."""
    print(f"DEBUG: Fetching real-time bar for {pair} (Twelves Data)...")
    url = "https://api.twelvedata.com/time_series"
    
    end_time = datetime.now()
    start_time = (end_time - timedelta(hours=config['historical_bars'] + 1)).strftime("%Y-%m-%d %H:%M:%S")

    params = {
        'symbol': pair,
        'interval': config['timeframe'],
        'outputsize': config['historical_bars'] + 1,
        'apikey': TWELVESDATA_API_KEY,
        'format': 'JSON'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'error':
            print(f"ERROR fetching Twelve Data for {pair}: {data.get('message')}")
            return None
        
        df = pd.DataFrame(data['values'])
        df = df.iloc[::-1].reset_index(drop=True) 
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        return df

    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching Twelve Data for {pair}: {e}")
        return None
    except KeyError:
        print(f"ERROR: Twelve Data response format unexpected for {pair}. Response: {data}")
        return None


def calculate_technical_bias(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Calculates MACD and Stochastic bias."""
    if df is None or len(df) < config['historical_bars']:
        return "NEUTRAL"
    
    # 1. MACD (Momentum)
    macd_instance = MACD(
        close=df['close'],
        window_fast=config['MACD_FAST'],
        window_slow=config['MACD_SLOW'],
        window_sign=config['MACD_SIGNAL'],
        fillna=True
    )
    macd = macd_instance.macd()
    macd_signal = macd_instance.macd_signal()
    
    # Check for crossover on the last closed bar
    if macd.iloc[-2] < macd_signal.iloc[-2] and macd.iloc[-1] > macd_signal.iloc[-1]:
        macd_bias = "BUY"
    elif macd.iloc[-2] > macd_signal.iloc[-2] and macd.iloc[-1] < macd_signal.iloc[-1]:
        macd_bias = "SELL"
    else:
        macd_bias = "NEUTRAL"

    # 2. Stochastic Oscillator (Overbought/Oversold)
    stoch_instance = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=config['STOCH_WINDOW'],
        smooth_window=config['STOCH_SMOOTH'],
        fillna=True
    )
    stoch_k = stoch_instance.stoch()
    stoch_d = stoch_instance.stoch_signal()
    
    # Check for crossover
    if stoch_k.iloc[-2] < stoch_d.iloc[-2] and stoch_k.iloc[-1] > stoch_d.iloc[-1]:
        stoch_bias = "BUY"
    elif stoch_k.iloc[-2] > stoch_d.iloc[-2] and stoch_k.iloc[-1] < stoch_d.iloc[-1]:
        stoch_bias = "SELL"
    else:
        stoch_bias = "NEUTRAL"
        
    # 3. Combine Biases
    if macd_bias == "BUY" and stoch_bias == "BUY":
        return "BUY"
    elif macd_bias == "SELL" and stoch_bias == "SELL":
        return "SELL"
    else:
        return "NEUTRAL"

# 💡 START OF CHANGE 4: Refactored function to use FMP API
def check_economic_calendar_surprise(pair: str) -> float:
    """
    Checks the Financial Modeling Prep (FMP) economic calendar for a high-impact news surprise.
    
    Returns a float score: >0.5 for a strong positive surprise, <-0.5 for strong negative, 
    and 0.0 otherwise.
    """
    print(f"--- Running Step 3.2.1: Checking Economic Calendar Surprise for {pair} (using FMP) ---")
    
    # 1. Define time range (Last 48 hours for safety)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d") 

    # 2. Construct API URL using FMP constants
    global TRADINGECONOMICS_API_KEY, FMP_CALENDAR_URL
    
    url = f"{FMP_CALENDAR_URL}?from={start_date}&to={end_date}&apikey={TRADINGECONOMICS_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        calendar_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching FMP economic calendar data for {pair}: {e}")
        return 0.0 

    # 3. Filter and Process Data
    
    # Relevant countries for EUR/USD and GBP/USD.
    relevant_countries = ['US', 'United States', 'Eurozone', 'Germany', 'France', 'UK', 'United Kingdom']
    
    # We only care about events in the last hour
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    max_surprise = 0.0
    
    for event in calendar_data:
        try:
            event_date = datetime.strptime(event['date'], '%Y-%m-%d %H:%M:%S')
            
            # Check if event is recent, relevant, and has necessary data
            if event_date >= one_hour_ago and event.get('country') in relevant_countries:
                actual = event.get('actual')
                forecast = event.get('forecast')
                impact = event.get('impact') 

                if actual is None or forecast is None or impact is None:
                    continue
                
                # Filter strictly for High/Medium impact 
                if 'High' not in impact and 'Medium' not in impact:
                    continue

                # Clean values
                actual = str(actual).replace('%', '').replace('K', '000').replace('M', '000000').replace('B', '000000000')
                forecast = str(forecast).replace('%', '').replace('K', '000').replace('M', '000000').replace('B', '000000000')

                try:
                    actual = float(actual)
                    forecast = float(forecast)
                except ValueError:
                    continue
                
                # Calculate surprise score 
                if abs(forecast) < 0.0001:
                    surprise_score = 1.0 if abs(actual) > 0.01 else 0.0
                else:
                    surprise_score = (actual - forecast) / abs(forecast)
                
                
                is_quote_currency_event = event['country'] in ['US', 'United States']
                final_score = surprise_score
                
                if pair.endswith('/USD') and is_quote_currency_event:
                    # USD news (quote currency) has inverse impact on EUR/USD or GBP/USD
                    final_score = -surprise_score
                
                # We are looking for the maximum shock 
                if abs(final_score) > abs(max_surprise):
                    max_surprise = final_score
                    
        except Exception:
            # Silently skip events that fail to parse
            continue

    print(f"DEBUG: Max News Surprise Score for {pair}: {max_surprise:.3f}")

    return max_surprise
# 💡 END OF CHANGE 4


def fetch_general_sentiment(pair: str) -> float:
    """Fetches general market sentiment score using MarketAux, NewsAPI, and Reuters feeds."""
    if pair == "EUR/USD":
        keywords = "(Eurozone inflation OR US Fed rate OR dollar OR euro OR ECB)"
    elif pair == "GBP/USD":
        keywords = "(UK Interest Rate OR Bank of England OR sterling OR pound OR BOE)"
    else:
        keywords = pair.replace('/', ' OR ')

    print(f"DEBUG: Fetching General Sentiment for {pair} using keywords: {keywords}...")
    
    sentiment_scores = []

    # 1. MarketAux API (Focused Financial News Sentiment)
    if MARKETAUX_API_KEY:
        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                'filter_entities': keywords,
                'api_token': MARKETAUX_API_KEY,
                'limit': 5  
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('data'):
                ma_sentiment = np.mean([item['sentiment_score'] for item in data['data'] if item.get('sentiment_score') is not None])
                sentiment_scores.append(ma_sentiment)
        except requests.exceptions.RequestException as e:
            print(f"ERROR fetching MarketAux data: {e}")

    # 2. NewsAPI (General News Coverage Sentiment)
    if NEW_NEWS_API_KEY:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': keywords,
                'apiKey': NEW_NEWS_API_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            positive_words = ['gain', 'rise', 'strong', 'optimistic', 'beat', 'growth']
            negative_words = ['drop', 'fall', 'weak', 'pessimistic', 'miss', 'recession']
            
            score = 0
            count = 0
            for article in data.get('articles', []):
                text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
                for word in positive_words:
                    if word in text:
                        score += 0.1
                        count += 1
                for word in negative_words:
                    if word in text:
                        score -= 0.1
                        count += 1
            if count > 0:
                sentiment_scores.append(score / count)
                
        except requests.exceptions.RequestException as e:
            print(f"ERROR fetching NewsAPI data: {e}")

    # 3. Reuters RSS Feed (Placeholder)
    try:
        feedparser.parse(REUTERS_RSS)
    except Exception as e:
        print(f"ERROR processing Reuters RSS: {e}")


    if not sentiment_scores:
        combined_sentiment = 0.0
    else:
        combined_sentiment = np.mean(sentiment_scores)
        
    print(f"DEBUG: Combined General Sentiment: {combined_sentiment:.3f} (from {len(sentiment_scores)} calculated scores)")
    
    return combined_sentiment


def generate_signal(pair: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Main logic for generating the trading signal based on a three-layer confirmation strategy.
    """
    
    # 1. Fetch Historical Data (Needed for Technical Analysis)
    df_history = get_realtime_bar(pair, config)
    if df_history is None or len(df_history) < config['historical_bars']:
        return {"signal": "HOLD", "reason": f"HOLD: Insufficient historical data for {pair}."}
        
    # 2. Layer 1: Economic Calendar Surprise (News Trade Override)
    surprise_score = check_economic_calendar_surprise(pair)
    surprise_threshold = config['SURPRISE_SCORE_THRESHOLD']

    if abs(surprise_score) >= surprise_threshold:
        signal = "BUY" if surprise_score > 0 else "SELL"
        # The logic below handles the inverse relationship for quote currency news
        return {"signal": signal, 
                "reason": f"NEWS TRADE OVERRIDE: Extreme news surprise detected (Score: {surprise_score:.3f})."}
    
    # 3. Layer 2: Technical Bias
    technical_bias = calculate_technical_bias(df_history, config)
    
    if technical_bias == "NEUTRAL":
        return {"signal": "HOLD", "reason": f"HOLD: Technical bias is NEUTRAL, and General Sentiment is neutral/conflicting (Score: {surprise_score:.3f}). No high-impact news."}

    # 4. Layer 3: General Sentiment Confirmation
    general_sentiment = fetch_general_sentiment(pair)
    sentiment_threshold = config['general_sentiment_threshold']
    
    if technical_bias == "BUY":
        if general_sentiment >= sentiment_threshold:
            return {"signal": "BUY", "reason": f"CONFIRMATION: Technical BUY confirmed by positive sentiment ({general_sentiment:.3f})."}
        elif general_sentiment < -sentiment_threshold:
             # Contradiction: Tech says BUY, Sentiment says SELL. Hold for safety.
            return {"signal": "HOLD", "reason": f"HOLD: Technical bias (BUY) strongly contradicts General Sentiment (SELL) ({general_sentiment:.3f})."}
        else:
            # Sentiment too neutral for confirmation. Hold.
            return {"signal": "HOLD", "reason": f"HOLD: Technical BUY bias lacks strong sentiment confirmation ({general_sentiment:.3f})."}

    elif technical_bias == "SELL":
        if general_sentiment <= -sentiment_threshold:
            return {"signal": "SELL", "reason": f"CONFIRMATION: Technical SELL confirmed by negative sentiment ({general_sentiment:.3f})."}
        elif general_sentiment > sentiment_threshold:
            # Contradiction: Tech says SELL, Sentiment says BUY. Hold for safety.
            return {"signal": "HOLD", "reason": f"HOLD: Technical bias (SELL) strongly contradicts General Sentiment (BUY) ({general_sentiment:.3f})."}
        else:
            # Sentiment too neutral for confirmation. Hold.
            return {"signal": "HOLD", "reason": f"HOLD: Technical SELL bias lacks strong sentiment confirmation ({general_sentiment:.3f})."}
            
    return {"signal": "HOLD", "reason": "HOLD: Unknown reason or processing error."}


def run_signal_generation_logic():
    """Main execution loop for all trading pairs."""
    print("==================================================\n")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("--- Starting Full Signal Generation Cycle ---\n")
    
    all_results = {}
    completed_trades_count = 0

    for pair, config in TRADING_PAIRS.items():
        print("================================================================================")
        print(f"### STARTING FULL TRADING CYCLE FOR {pair} ###")
        print("================================================================================\n")
        
        try:
            print("==================================================")
            print(f"### STEP 1: INITIALIZING HISTORICAL DATA for {pair} ###")
            print("==================================================\n")
            print(f"--- Initializing Historical Data for {pair} ---")
            
            # The get_realtime_bar function handles historical fetching
            df_history = get_realtime_bar(pair, config)
            if df_history is None:
                 raise Exception(f"Failed to fetch historical data for {pair}.")

            # STEP 2: GENERATE REAL-TIME SIGNAL
            print("==================================================")
            print(f"### STEP 2: GENERATING UPGRADED REAL-TIME SIGNAL for {pair} ###")
            print("==================================================\n")
            
            result = generate_signal(pair, config)
            
            # STEP 3: NOTIFICATION AND LOGGING
            all_results[pair] = result
            
            print("##################################################")
            print("### 🔔 FINAL TRADING SIGNAL NOTIFICATION 🔔 ###")
            print(f"PAIR: {pair}")
            print(f"SIGNAL: **{result['signal']}**")
            print(f"Reason: {result['reason']}")
            print("##################################################\n")

            subject = f"Signal: {result['signal']} for {pair}"
            body = f"Pair: {pair}\nSignal: {result['signal']}\nReason: {result['reason']}\n\nThis signal was generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            
            send_email_notification(subject, body, MAIL_RECIPIENT)
            
            if result['signal'] != 'HOLD':
                 completed_trades_count += 1
            
            # Rate limit avoidance
            print("==================================================")
            print(f"### PAUSING FOR 60 SECONDS TO AVOID TWELVES DATA RATE LIMIT for {pair} ###")
            time.sleep(60) 
            print(f"Resuming execution at: {(datetime.now() + timedelta(seconds=60)).strftime('%Y-%m-%d %H:%M:%S')}")
            print("==================================================\n")

        except Exception as e:
            error_message = f"An error occurred while processing {pair}: {e}"
            print(f"ERROR: {error_message}")
            send_email_notification(f"ERROR processing {pair}", error_message, MAIL_RECIPIENT)

    print("**************************************************")
    print("--- Full Cycle Complete ---")
    print(f"Details: {completed_trades_count} completed trades generated.")
    print(f"Final Results: {json.dumps(all_results, indent=4)}")
    print("**************************************************")
    
    return all_results

# --- Wrapper for background execution (Retained) ---
def run_signal_generation_logic_wrapper():
    """Wrapper to handle critical errors during thread execution."""
    try:
        run_signal_generation_logic()
    except Exception as e:
        print("!"*50)
        print("### 🛑 CRITICAL EXECUTION ERROR DURING RUNTIME 🛑 ###")
        print(f"An unexpected error occurred: {e}")
        print("\n--- Detailed Stack Trace ---")
        traceback.print_exc()
        print("!"*50 + "\n")

        error_subject = "FATAL: Trading Bot Execution Failed"
        error_body = f"The trading signal logic encountered a critical error:\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}"
        send_email_notification(error_subject, error_body, MAIL_RECIPIENT)

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

    thread = Thread(target=run_signal_generation_logic_wrapper) 
    thread.start()

    response = {
        "status": "Processing started in background",
        "message": "The signal generation logic is running asynchronously. Check the application logs and your email for the final signal output."
    }

    return jsonify(response), 202
if __name__ == "__main__":
    print("Starting Flask server for manual execution...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
