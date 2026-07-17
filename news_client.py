import os
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger("NewsClient")

AUTHORITY_SOURCES = [
    "Bureau of Labor Statistics", "Federal Reserve", "Department of Labor",
    "Census Bureau", "Bureau of Economic Analysis", "Bank of England",
    "Office for National Statistics", "European Central Bank", "Eurostat",
    "Deutsche Bundesbank", "Bank of Canada", "Statistics Canada", 
    "Reserve Bank of Australia", "Australian Bureau of Statistics"
]

class NewsCalendarClient:
    def __init__(self, api_key: str = None, news_source: str = None):
        self.api_key = api_key or os.getenv("JBLANKED_API_KEY")
        self.news_source = news_source or os.getenv("NEWS_SOURCE", "forex-factory")
        self.fmp_key = os.getenv("FMP_API_KEY")
        self.newsapi_key = os.getenv("NEW_NEWS_API_KEY") or os.getenv("NEWS_API_KEY_2")
        
        self.base_url = f"https://www.jblanked.com/news/api/{self.news_source}/calendar/today/"

    def fetch_high_impact_events(self) -> list:
        """Fetches structured economic events from JBlanked."""
        if not self.api_key:
            return []
        headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {self.api_key}"}
        try:
            response = requests.get(self.base_url, headers=headers, timeout=8)
            if response.status_code != 200:
                logger.error(f"JBlanked API returned status code {response.status_code}")
                return []
            
            vetted_events = []
            for event in response.json():
                impact = event.get("Impact") or event.get("impact")
                if impact != "High":
                    continue
                currency = event.get("Currency") or event.get("currency")
                if currency not in ["USD", "EUR", "GBP"]:
                    continue
                    
                name = event.get("Name") or event.get("name") or ""
                category = event.get("Category") or event.get("category") or ""
                
                is_authorized = any(src.lower() in name.lower() or src.lower() in category.lower() for src in AUTHORITY_SOURCES)
                if not is_authorized and any(kw in name.lower() for kw in ["rate", "fomc", "cpi", "nfp", "gdp"]):
                    is_authorized = True
                    
                if not is_authorized:
                    continue

                date_str = event.get("Date") or event.get("date")
                try:
                    scheduled_time = datetime.fromisoformat(date_str.replace(".", "-")).replace(tzinfo=timezone.utc)
                except Exception:
                    try:
                        scheduled_time = datetime.strptime(date_str, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except Exception:
                        continue

                vetted_events.append({
                    "id": str(event.get("ID") or name),
                    "type": "economic_calendar",
                    "title": name,
                    "currency": currency,
                    "scheduled_time": scheduled_time,
                    "forecast": self._clean_numeric_value(event.get("Forecast")),
                    "previous": self._clean_numeric_value(event.get("Previous")),
                    "deviation_threshold": self._determine_threshold(name)
                })
            return vetted_events
        except Exception as e:
            logger.error(f"JBlanked error: {e}")
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
