import os
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger("NewsClient")

# Curated to major market-moving entities
AUTHORITY_SOURCES = [
    "Federal Reserve", "Federal Open Market Committee", "Bureau of Labor Statistics", 
    "Department of Labor", "Bureau of Economic Analysis", "Census Bureau",
    "European Central Bank", "Eurostat", "Bank of England", "Office for National Statistics",
    "Bank of Canada", "Statistics Canada", "Reserve Bank of Australia", "Australian Bureau of Statistics",
    "Reserve Bank of New Zealand", "Statistics New Zealand", "Bank of Japan", "Ministry of Finance",
    "Cabinet Office Japan", "Swiss National Bank", "State Secretariat for Economic Affairs"
]

# Robust macro target checklist 
AUTHORIZED_KEYWORDS = [
    "cpi", "core cpi", "ppi", "pce", "core pce", "nonfarm payroll", "nfp", "payroll",
    "unemployment", "jobless", "retail sales", "gdp", "interest rate", "rate decision",
    "fomc", "minutes", "powell", "ecb", "lagarde", "boe", "bailey", "manufacturing pmi",
    "services pmi", "ism", "consumer confidence", "durable goods", "trade balance", "inflation"
]

class NewsCalendarClient:
    def __init__(self, api_key: str = None, news_source: str = None):
        self.api_key = api_key or os.getenv("JBLANKED_API_KEY")
        self.news_source = news_source or os.getenv("NEWS_SOURCE", "forex-factory")
        self.fmp_key = os.getenv("FMP_API_KEY")
        self.newsapi_key = os.getenv("NEW_NEWS_API_KEY") or os.getenv("NEWS_API_KEY_2")
        # Added Finnhub token fallback as a free alternative for the economic calendar
        self.finnhub_key = os.getenv("FINNHUB_API_KEY") 

    def fetch_high_impact_events(self) -> list:
        """
        Fetches high-impact economic indicators. Swapped to Finnhub as an alternative
        since FMP's economic calendar returns a 403 error on the Free Plan.
        """
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY missing. Skipping free economic calendar stream to prevent FMP 403 blocks.")
            return []
            
        # Finnhub provides a free economic calendar endpoint
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {"token": self.finnhub_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Finnhub Calendar API returned status code {response.status_code}")
                return []
                
            vetted_events = []
            # Finnhub wraps data inside an 'economicCalendar' array
            events = response.json().get("economicCalendar", [])
            
            for event in events:
                # Finnhub impact uses integer rating matrix scales or strings. Filter high-impact:
                # Typically, Finnhub uses 1-3 stars or 'high' strings depending on version.
                impact = str(event.get("impact", "")).lower()
                if "high" not in impact and impact != "3":
                    continue
                    
                # Standardize to your existing architecture 
                currency = event.get("currency")
                if currency not in ["USD", "EUR", "GBP"]:
                    continue
                    
                name = event.get("event") or ""
                country = event.get("country") or ""
                text = f"{name} {country}".lower()
                
                is_authorized = (
                    any(src.lower() in text for src in AUTHORITY_SOURCES) or
                    any(keyword in text for keyword in AUTHORIZED_KEYWORDS)
                )
                    
                if not is_authorized:
                    continue

                date_str = event.get("time") # Finnhub uses 'time' key for execution timestamps
                try:
                    # Finnhub time strings typically map natively via ISO format parsers
                    scheduled_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except Exception:
                    continue

                vetted_events.append({
                    "id": f"free_cal_{name}_{date_str}".replace(" ", "_"),
                    "type": "economic_calendar",
                    "title": name,
                    "currency": currency,
                    "scheduled_time": scheduled_time,
                    "forecast": self._clean_numeric_value(event.get("estimate")),
                    "previous": self._clean_numeric_value(event.get("previous")),
                    "deviation_threshold": self._determine_threshold(name)
                })
            return vetted_events
            
        except Exception as e:
            logger.error(f"Economic Calendar fallback aggregation error: {e}")
            return []

    def fetch_fmp_news(self) -> list:
        """Fetches real-time market news from Financial Modeling Prep using Free Tier rules."""
        if not self.fmp_key:
            return []
            
        # Cleaned URL up for FMP Free Tier compatibility, avoiding nested server-side page structures
        url = "https://financialmodelingprep.com/api/v3/fmp/articles"
        params = {
            "limit": 5,
            "apikey": self.fmp_key
        }
        try:
            res = requests.get(url, params=params, timeout=8)
            if res.status_code != 200:
                logger.debug(f"FMP Articles returned {res.status_code}. Might be limited on free accounts.")
                return []
                
            # Free tier direct response payload arrays vs paginated objects
            data = res.json()
            articles = data if isinstance(data, list) else data.get("content", [])
            
            events = []
            for art in articles:
                title = art.get("title", "")
                text = art.get("content", "")
                full_text = f"{title} {text}".lower()
                
                currency = None
                if any(x in full_text for x in ["fed", "fomc", "powell", "dollar", "usd"]):
                    currency = "USD"
                elif any(x in full_text for x in ["ecb", "lagarde", "euro", "eur"]):
                    currency = "EUR"
                elif any(x in full_text for x in ["boe", "bailey", "pound", "gbp"]):
                    currency = "GBP"
                
                if not currency:
                    continue
                    
                sentiment = 0
                if any(w in full_text for w in ["hike", "hawkish", "bullish", "strong", "growth"]):
                    sentiment = 1
                elif any(w in full_text for w in ["cut", "dovish", "bearish", "weak", "recession"]):
                    sentiment = -1
                    
                if sentiment == 0:
                    continue
                    
                events.append({
                    "id": f"fmp_{art.get('id', title)}",
                    "type": "breaking_news",
                    "title": title,
                    "currency": currency,
                    "scheduled_time": datetime.now(timezone.utc),
                    "sentiment": sentiment
                })
            return events
        except Exception as e:
            logger.error(f"FMP News Processing error: {e}")
            return []

    def fetch_newsapi_signals(self) -> list:
        """Fetches breaking global news from NewsAPI."""
        if not self.newsapi_key:
            return []
        url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={self.newsapi_key}"
        try:
            res = requests.get(url, timeout=8)
            if res.status_code != 200:
                return []
            articles = res.json().get("articles", [])
            events = []
            for art in articles:
                title = art.get("title", "")
                desc = art.get("description", "") or ""
                full_text = f"{title} {desc}".lower()
                
                currency = None
                if "usd" in full_text or "fed" in full_text or "wall street" in full_text:
                    currency = "USD"
                elif "euro" in full_text or "ecb" in full_text:
                    currency = "EUR"
                elif "inflation" in full_text or "boe" in full_text or "uk" in full_text:
                    currency = "GBP"
                    
                if not currency:
                    continue
                    
                sentiment = 0
                if any(w in full_text for w in ["surge", "higher", "positive", "gains", "beat"]):
                    sentiment = 1
                elif any(w in full_text for w in ["drop", "plunge", "lower", "losses", "missed"]):
                    sentiment = -1
                    
                if sentiment == 0:
                    continue

                events.append({
                    "id": f"newsapi_{title}",
                    "type": "breaking_news",
                    "title": title,
                    "currency": currency,
                    "scheduled_time": datetime.now(timezone.utc),
                    "sentiment": sentiment
                })
            return events
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

    def _clean_numeric_value(self, val) -> float:
        if val is None or val == "": return 0.0
        if isinstance(val, (int, float)): return float(val)
        try:
            val_str = str(val).upper().strip().replace("%", "").replace(",", "")
            m = 1.0
            if "K" in val_str: m = 1000.0; val_str = val_str.replace("K", "")
            if "M" in val_str: m = 1000000.0; val_str = val_str.replace("M", "")
            return float(val_str) * m
        except ValueError: return 0.0

    def _determine_threshold(self, name: str) -> float:
        n = name.lower()
        if "rate" in n: return 0.25
        if "cpi" in n: return 0.2
        if "nfp" in n or "payrolls" in n: return 30000.0
        return 0.1
