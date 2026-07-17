import os
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger("NewsClient")

# Strict list of authority sources representing central banks and national statistics bureaus
AUTHORITY_SOURCES = [
    "Bureau of Labor Statistics", "Federal Reserve", "Department of Labor",
    "Census Bureau", "Bureau of Economic Analysis", "Bank of England",
    "Office for National Statistics", "European Central Bank", "Eurostat",
    "Deutsche Bundesbank", "Bank of Canada", "Statistics Canada", 
    "Reserve Bank of Australia", "Australian Bureau of Statistics"
]

class NewsCalendarClient:
    def __init__(self, api_key: str = None, news_source: str = None):
        """
        Supports JBlanked API sources: 'forex-factory', 'mql5', or 'fxstreet'
        Pulls dynamically from environment if parameters are omitted.
        """
        self.api_key = api_key or os.getenv("JBLANKED_API_KEY")
        self.news_source = news_source or os.getenv("NEWS_SOURCE", "forex-factory")
        self.base_url = f"https://www.jblanked.com/news/api/{self.news_source}/calendar/today/"
        
        if not self.api_key:
            logger.error("JBLANKED_API_KEY environment variable is missing!")

    def fetch_high_impact_events(self) -> list:
        """
        Fetches today's economic events and filters down to high-impact, 
        authority-vetted releases for EUR, USD, and GBP.
        """
        if not self.api_key:
            logger.error("Cannot fetch news; JBlanked API Key is unconfigured.")
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        try:
            logger.info(f"Fetching today's economic calendar from JBlanked ({self.news_source})...")
            response = requests.get(self.base_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch calendar. Status Code: {response.status_code}")
                return []
                
            raw_events = response.json()
            vetted_events = []
            
            for event in raw_events:
                # 1. Filter by High Impact
                impact = event.get("Impact") or event.get("impact")
                if impact != "High":
                    continue
                    
                # 2. Filter by Core Currencies
                currency = event.get("Currency") or event.get("currency")
                if currency not in ["USD", "EUR", "GBP"]:
                    continue
                    
                # 3. Source Authority Evaluation
                category = event.get("Category") or event.get("category") or ""
                # Check if event name or category points to a trusted official institution
                name = event.get("Name") or event.get("name") or ""
                is_authorized = any(src.lower() in name.lower() or src.lower() in category.lower() for src in AUTHORITY_SOURCES)
                
                # If source is not explicitly known but is standard high impact central bank event (e.g. Rate Decision)
                if not is_authorized and any(kw in name.lower() for kw in ["rate", "fomc", "cpi", "nfp", "employment", "gdp"]):
                    is_authorized = True
                    
                if not is_authorized:
                    logger.warning(f"Skipping event '{name}' - Failed source authority verification.")
                    continue
                
                # Normalize time format into standard ISO datetime
                date_str = event.get("Date") or event.get("date")
                try:
                    clean_date_str = date_str.replace(".", "-")
                    scheduled_time = datetime.fromisoformat(clean_date_str).replace(tzinfo=timezone.utc)
                except Exception:
                    try:
                        scheduled_time = datetime.strptime(date_str, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except Exception as parse_err:
                        logger.error(f"Could not parse event datetime '{date_str}': {parse_err}")
                        continue

                vetted_events.append({
                    "id": event.get("ID") or event.get("event_id") or name,
                    "title": name,
                    "currency": currency,
                    "scheduled_time": scheduled_time,
                    "forecast": self._clean_numeric_value(event.get("Forecast") or event.get("forecast")),
                    "previous": self._clean_numeric_value(event.get("Previous") or event.get("previous")),
                    "deviation_threshold": self._determine_threshold(name)
                })
                
            logger.info(f"Successfully processed and vetted {len(vetted_events)} high-impact events for today.")
            return vetted_events
            
        except Exception as e:
            logger.error(f"Error executing news calendar fetch sequence: {e}")
            return []

    def _clean_numeric_value(self, val) -> float:
        """Strip percent symbols, 'K', 'M' and convert raw strings to clean floats"""
        if val is None or val == "":
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            val_str = str(val).upper().strip()
            multiplier = 1.0
            if "K" in val_str:
                multiplier = 1000.0
                val_str = val_str.replace("K", "")
            elif "M" in val_str:
                multiplier = 1000000.0
                val_str = val_str.replace("M", "")
            val_str = val_str.replace("%", "").replace(",", "").strip()
            return float(val_str) * multiplier
        except ValueError:
            return 0.0

    def _determine_threshold(self, name: str) -> float:
        """Determines the minimum surprise deviation needed to trigger execution based on event type."""
        name_lower = name.lower()
        if "rate" in name_lower:
            return 0.25
        elif "cpi" in name_lower:
            return 0.2 
        elif "unemployment" in name_lower:
            return 0.2
        elif "nfp" in name_lower or "payrolls" in name_lower:
            return 30000.0
        elif "gdp" in name_lower:
            return 0.3
        return 0.1
