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

    def fetch_high_impact_events(self) -> list:
        """
        Fetches high-impact economic indicators from FMP and verifies them 
        using a dual Authority/Keyword matrix framework.
        """
        if not self.fmp_key:
            logger.error("FMP_API_KEY is missing. Skipping economic calendar aggregation.")
            return []
            
        url = "https://financialmodelingprep.com/stable/economic-calendar"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        params = {
            "apikey": self.fmp_key,
            "from": today_str,
            "to": today_str
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"FMP Economic Calendar API returned status code {response.status_code}")
                return []
                
            vetted_events = []
            for event in response.json():
                # Primary filter: Trust FMP's high impact classification metadata
                if event.get("impact") != "High":
                    continue
                    
                currency = event.get("currency")
                if currency not in ["USD", "EUR", "GBP"]:
                    continue
                    
                name = event.get("event") or ""
                country = event.get("country") or ""
                
                # Broad text matching context evaluation
                text = f"{name} {country}".lower()
                
                # Check for either specific institutional authority OR matching macro indicators
                is_authorized = (
                    any(src.lower() in text for src in AUTHORITY_SOURCES) or
                    any(keyword in text for keyword in AUTHORIZED_KEYWORDS)
                )
                    
                if not is_authorized:
                    continue

                date_str = event.get("date")
                try:
                    scheduled_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                vetted_events.append({
                    "id": f"fmp_cal_{name}_{date_str}".replace(" ", "_"),
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
            logger.error(f"FMP Economic Calendar error: {e}")
            return []

    def fetch_fmp_news(self) -> list:
        """Fetches real-time market news from Financial Modeling Prep."""
        if not self.fmp_key:
            return []
        url = f"https://financialmodelingprep.com/api/v3/fmp/articles?page=0&size=5&apikey={self.fmp_key}"
        try:
            res = requests.get(url, timeout=8)
            if res.status_code != 200:
                return []
            articles = res.json().get("content", [])
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
            logger.error(f"FMP error: {e}")
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
