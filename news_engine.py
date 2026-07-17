import os
import asyncio
import logging
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI
from news_client import NewsCalendarClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NewsEngine")

MAX_ALLOWED_SPREAD_PIPS = float(os.getenv("MAX_ALLOWED_SPREAD_PIPS", "3.0"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize FastAPI app for external verification and keep-alive pings
app = FastAPI()
engine_instance = None

class NewsStateEngine:
    def __init__(self):
        self.client = NewsCalendarClient()
        self.active_events = []
        self.processed_event_ids = set()

    async def send_telegram_alert(self, message: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.info(f"[Dry Run]: {message}")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Telegram failed: {e}")

    async def collect_all_sources(self):
        """Polls multiple structural and unstructured sources to find matching active vectors."""
        all_vetted = []
        
        # 1. Structural Calendar Releases
        calendar_events = self.client.fetch_high_impact_events()
        now = datetime.now(timezone.utc)
        for e in calendar_events:
            if e["scheduled_time"] > now and e["id"] not in self.processed_event_ids:
                all_vetted.append(e)

        # 2. FMP Breaking Aggregations
        fmp_events = self.client.fetch_fmp_news()
        for e in fmp_events:
            if e["id"] not in self.processed_event_ids:
                all_vetted.append(e)

        # 3. NewsAPI Global Headlines
        newsapi_events = self.client.fetch_newsapi_signals()
        for e in newsapi_events:
            if e["id"] not in self.processed_event_ids:
                all_vetted.append(e)

        self.active_events = pdf = all_vetted
        logger.info(f"Aggregated Queue Sync: {len(self.active_events)} unprocessed events pending.")

    async def process_breaking_news(self, event: dict):
        """Handles immediate algorithmic processing for unstructured FMP or NewsAPI signals."""
        eid = event["id"]
        title = event["title"]
        currency = event["currency"]
        sentiment = event["sentiment"]
        
        trade_pair = f"{currency}/USD" if currency != "USD" else "EUR/USD"
        direction = "CALL" if sentiment == 1 else "PUT"
        
        execution_msg = (
            f"⚡ *BREAKING DATA SIGNAL INSTANT TRIGGER*\n\n"
            f"Source: `{eid.split('_')[0].upper()}`\n"
            f"Action: *{direction}*\n"
            f"Asset: *{trade_pair}*\n"
            f"Headline: `{title}`\n"
            f"Vector Phase: Real-Time Momentum Run"
        )
        logger.info(f"Executing direct macro trade {direction} on {trade_pair}")
        await self.send_telegram_alert(execution_msg)
        self.processed_event_ids.add(eid)

    async def run_event_lifecycle(self, event: dict):
        if event["type"] == "breaking_news":
            await self.process_breaking_news(event)
            return

        event_id = event["id"]
        title = event["title"]
        currency = event["currency"]
        scheduled_time = event["scheduled_time"]
        trade_pair = f"{currency}/USD" if currency != "USD" else "EUR/USD"

        while True:
            now = datetime.now(timezone.utc)
            seconds_remaining = (scheduled_time - now).total_seconds()

            if seconds_remaining <= 0.5:
                # T-0 logic (Unchanged execution validation block)
                direction = "CALL" if event["forecast"] > event["previous"] else "PUT"
                execution_msg = (
                    f"🚀 *CALENDAR NEWS EXECUTION TRIGGERED*\n\n"
                    f"Action: *{direction}*\n"
                    f"Asset: *{trade_pair}*\n"
                    f"Trigger Event: `{title}`\n"
                )
                await self.send_telegram_alert(execution_msg)
                break
            else:
                if seconds_remaining > 60:
                    await asyncio.sleep(30)
                else:
                    await asyncio.sleep(1)

        self.processed_event_ids.add(event_id)

    async def run_master_loop(self):
        """High-frequency continuous collection engine loop."""
        while True:
            try:
                await self.collect_all_sources()
                
                active_tasks = []
                for event in self.active_events:
                    if event["id"] not in self.processed_event_ids:
                        task = asyncio.create_task(self.run_event_lifecycle(event))
                        active_tasks.append(task)
                        
                if active_tasks:
                    await asyncio.gather(*active_tasks)

                # Poll everything every 20 minutes to keep finding news signals
                await asyncio.sleep(1200)
            except Exception as e:
                logger.error(f"Error in Master Loop: {e}")
                await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    global engine_instance
    engine_instance = NewsStateEngine()
    # Run the continuous processing background task alongside the server
    asyncio.create_task(engine_instance.run_master_loop())

@app.get("/cron")
@app.get("/health")
def keep_alive_endpoint():
    """Target URL for external cronjob engines to prevent service sleep"""
    return {
        "status": "active",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processed_queue_size": len(engine_instance.processed_event_ids) if engine_instance else 0
    }

if __name__ == "__main__":
    import uvicorn
    # Render binds dynamic platform ports directly to the PORT variable
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
