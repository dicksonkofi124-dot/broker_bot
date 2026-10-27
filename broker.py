import asyncio
import random
import logging
import sys
import math
import httpx
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from datetime import datetime, timedelta, timezone
from collections import deque
from supabase import create_client, Client

# ============================================================
# CONFIGURATION
# ============================================================
SUPABASE_URL = "https://supabase.co"
SUPABASE_KEY = "sb_publishable_MOAgTjmJK9hOtB36yYrylA_sYxufidf"

TELEGRAM_BOT_TOKEN = "8705718355:AAFNWWMrmAacQIu876J4kyopXAp_Twcnj44"
TELEGRAM_CHAT_ID = "1059602487"

SEQUENCE_LENGTH = 20
OBSERVATION_SECONDS = 12
TRADE_EXPIRATION_SECONDS = 60
RESOLUTION_POLL_INTERVAL = 3.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EO_Signal_Engine")


# ============================================================
# BROADCAST TELEGRAM FUNCTION
# ============================================================
async def broadcast_telegram(message: str, db_client: Client):
    if TELEGRAM_BOT_TOKEN.startswith("YOUR_"):
        logger.warning("Telegram token not configured.")
        return

    target_chats = [TELEGRAM_CHAT_ID]

    try:
        response = db_client.table("bot_users").select("chat_id").execute()
        if response.data:
            for user in response.data:
                chat_str = str(user["chat_id"])
                if chat_str not in target_chats:
                    target_chats.append(chat_str)
    except Exception as e:
        logger.error(f"Error fetching subscriber list from Supabase: {e}")

    url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    async with httpx.AsyncClient() as client:
        for chat_id in target_chats:
            try:
                await client.post(url, json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }, timeout=10)
            except Exception as e:
                logger.error(f"Failed sending to chat {chat_id}: {e}")


# ============================================================
# Adaptive Model
# ============================================================
class AdaptiveStatisticalPredictor:
    def __init__(self, memory_size=80):
        self.outcome_memory = deque(maxlen=memory_size)
        self.bias = 0.0
        self.base_threshold = 0.012
        self.total_wins = 0
        self.total_trades = 0

    def evaluate_sequence(self, prices):
        n = len(prices)
        if n < 10:
            return 0, 0.0, 60, "Low"

        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(prices)
        sum_xx = sum(i * i for i in x)
        sum_xy = sum(x[i] * prices[i] for i in range(n))

        denominator = (n * sum_xx) - (sum_x ** 2)
        if denominator == 0:
            return 0, 0.0, 60, "Low"

        slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator

        returns = [math.log(prices[i] / prices[i-1]) for i in range(1, n) if prices[i-1] > 0]
        mean_return = sum(returns) / len(returns) if returns else 0.0
        variance = sum((r - mean_return)**2 for r in returns) / len(returns) if returns else 0.0
        volatility = math.sqrt(variance) if variance > 0 else 0.0

        raw_score = slope / (volatility + 1e-6)
        adjusted_score = raw_score + self.bias

        threshold = self.base_threshold
        if len(self.outcome_memory) > 30:
            threshold *= 0.75

        if adjusted_score > threshold:
            direction = 1
        elif adjusted_score < -threshold:
            direction = -1
        else:
            return 0, adjusted_score, 60, "Low"

        abs_score = abs(adjusted_score)

        if abs_score > 0.45:
            expiration = 5 if volatility > 0.00015 else 15
        elif abs_score > 0.28:
            expiration = 15 if volatility > 0.00012 else 30
        elif abs_score > 0.15:
            expiration = 30 if volatility > 0.00009 else 60
        else:
            expiration = 60 if volatility > 0.00007 else 120

        if abs_score >= 0.38:
            confidence = "High"
        elif abs_score >= 0.22:
            confidence = "Medium"
        else:
            confidence = "Low"

        return direction, adjusted_score, expiration, confidence

    def learn_from_outcome(self, score, was_win):
        self.outcome_memory.append((score, was_win))
        self.total_trades += 1
        if was_win:
            self.total_wins += 1

        if len(self.outcome_memory) < 8:
            return

        wins = sum(1 for _, w in self.outcome_memory if w)
        win_rate = wins / len(self.outcome_memory)

        if win_rate >= 0.58:
            self.bias += 0.0008
        elif win_rate <= 0.42:
            self.bias -= 0.0015

        self.bias = max(-0.06, min(0.06, self.bias))
        logger.info(f"[LEARNING] WR: {win_rate:.1%} | Bias: {self.bias:+.4f}")


# ============================================================
# Data Feed
# ============================================================
class MarketDataFeed:
    def __init__(self):
        self.base_price = 1.120991
        self.price_buffer = deque(maxlen=SEQUENCE_LENGTH + 10)

    async def start(self):
        while True:
            self.base_price += random.uniform(-0.00004, 0.00004)
            self.price_buffer.append(self.base_price)
            await asyncio.sleep(0.25)

    def get_latest_prices(self, n):
        return list(self.price_buffer)[-n:]

    def get_current_price(self):
        return self.price_buffer[-1] if self.price_buffer else None


# ============================================================
# Main Bot Engine
# ============================================================
class EOLivePredictorBot:
    def __init__(self):
        self.feed = MarketDataFeed()
        self.model = AdaptiveStatisticalPredictor()
        self.is_running = True
        self.trading_unlocked = False
        self.unlock_time = datetime.now() + timedelta(seconds=OBSERVATION_SECONDS)
        self.last_signal_time = datetime.now() - timedelta(seconds=60)

        try:
            self.db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Connected to Supabase")
        except Exception as e:
            logger.critical(f"DB Error: {e}")
            sys.exit(1)

    async def monitor_phase(self):
        while self.is_running:
            if not self.trading_unlocked and datetime.now() >= self.unlock_time:
                self.trading_unlocked = True
                await broadcast_telegram("🚀 <b>Bot is now LIVE</b>\nOnly Medium & High confidence signals will be sent.", self.db_client)
                logger.warning("SIGNALS ACTIVE - Low confidence ignored")
            await asyncio.sleep(0.5)

    async def telegram_listener_loop(self):
        offset = 0
        url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/getUpdates"
        
        while self.is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = client.get(url, params={"offset": offset, "timeout": 20}, timeout=25)
                    data = response.json()
                    
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            offset = update["update_id"] + 1
                            message = update.get("message")
                            
                            if not message:
                                continue
                                
                            text = message.get("text", "")
                            chat_id = message["chat"]["id"]
                            
                            if text.startswith("/broadcast ") and str(chat_id) == TELEGRAM_CHAT_ID:
                                announcement = text.replace("/broadcast ", "", 1).strip()
                                if announcement:
                                    formatted_announcement = f"📢 <b>ADMIN ANNOUNCEMENT</b>\n\n{announcement}"
                                    await broadcast_telegram(formatted_announcement, self.db_client)
                                continue

                            if text == "/start":
                                first_name = message["chat"].get("first_name", "Trader")
                                try:
                                    self.db_client.table("bot_users").upsert({
                                        "chat_id": str(chat_id),
                                        "username": first_name
                                    }, on_conflict="chat_id").execute()
                                    
                                    welcome_url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/sendMessage"
                                    await client.post(welcome_url, json={
                                        "chat_id": chat_id,
                                        "text": f"👋 Hello {first_name}! Welcome to the Live Broker Signals channel. You will now receive live Medium & High confidence alerts."
                                    })
                                except Exception as dbe:
                                    logger.error(f"Database insertion failed for user {chat_id}: {dbe}")
            except Exception as e:
                logger.error(f"Telegram listener error: {e}")
            
            await asyncio.sleep(2)

    async def log_trade(self, direction, price, score):
        try:
            self.db_client.table("paper_trades").insert({
                "asset_pair": "EUR/USD (OTC)",
                "direction": direction,
                "entry_price": round(price, 5),
                "pytorch_output": round(score, 4),
                "status": "PENDING"
            }).execute()
