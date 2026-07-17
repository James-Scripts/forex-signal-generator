import os
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI
from news_client import NewsCalendarClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NewsEngine")

MAX_ALLOWED_SPREAD_PIPS = float(os.getenv("MAX_ALLOWED_SPREAD_PIPS", "3.0"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
        all_vetted = []
        
        calendar_events = self.client.fetch_high_impact_events()
        now = datetime.now(timezone.utc)
        for e in calendar_events:
            if e["scheduled_time"] > now and e["id"] not in self.processed_event_ids:
                all_vetted.append(e)

        fmp_events = self.client.fetch_fmp_news()
        for e in fmp_events:
            if e["id"] not in self.processed_event_ids:
                all_vetted.append(e)

        newsapi_events = self.client.fetch_newsapi_signals()
        for e in newsapi_events:
            if e["id"] not in self.processed_event_ids:
                all_vetted.append(e)

        self.active_events = all_vetted
        logger.info(f"Aggregated Queue Sync: {len(self.active_events)} unprocessed events pending.")

    async def process_breaking_news(self, event: dict):
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

                await asyncio.sleep(1200)
            except Exception as e:
                logger.error(f"Error in Master Loop: {e}")
                await asyncio.sleep(60)

engine_instance = None

# Using the modern lifespan context manager pattern to silence deprecation warnings
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance
    engine_instance = NewsStateEngine()
    asyncio.create_task(engine_instance.run_master_loop())
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root_endpoint():
    """Default landing route to satisfy standard platform health checks"""
    return {"status": "online", "service": "Forex News Signal Engine"}



@app.get("/cron")
@app.get("/health")
def keep_alive_endpoint():
    return {
        "status": "active",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processed_queue_size": len(engine_instance.processed_event_ids) if engine_instance else 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
