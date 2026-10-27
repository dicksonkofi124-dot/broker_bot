import asyncio
import json
import random
import logging
import os
import math
import httpx
import websockets
from contextlib import asynccontextmanager
from fastapi import FastAPI
from datetime import datetime, timedelta, timezone
from collections import deque
from supabase import create_client, Client

# ============================================================
# CONFIGURATION
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_SYMBOL = os.getenv("FINNHUB_SYMBOL", "OANDA:EUR_USD")
FINNHUB_WS_URL = "wss://ws.finnhub.io"

SEQUENCE_LENGTH = 20
OBSERVATION_SECONDS = 12
TRADE_EXPIRATION_SECONDS = 30
RESOLUTION_POLL_INTERVAL = 3.0
# Rows older than this were scored by the OLD formula, so they are ignored.
LEARNING_HISTORY_START_AT = datetime(2026, 10, 27, 13, tzinfo=timezone.utc)
LEARNING_HISTORY_LIMIT = 200

# Learning settings
LEARNING_SAMPLE_SECONDS = 5     # take a training sample every 5s
MODEL_STATE_ID = "eurusd_online_v1"   # row id in the model_state table
MODEL_SAVE_EVERY = 50            # save learned weights every 50 new samples
MIN_TRAIN_SAMPLES = 300         # model stays silent until it has learned from this many
MIN_EDGE = 0.03                 # P(up) must be at least 3% away from 50% to signal
MEDIUM_EDGE = 0.05
HIGH_EDGE = 0.10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# httpx logs full URLs at INFO level, which would print your Telegram bot token in the logs
for noisy_logger in ("httpx", "httpcore", "websockets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger("EO_Signal_Engine")


# ============================================================
# BROADCAST TELEGRAM FUNCTION
# ============================================================
async def broadcast_telegram(
    message: str,
    db_client: Client,
    subscribers: set[str] | None = None,
):
    if TELEGRAM_BOT_TOKEN.startswith("YOUR_"):
        logger.warning("Telegram token not configured.")
        return

    target_chats = {TELEGRAM_CHAT_ID}
    if subscribers:
        target_chats.update(subscribers)

    try:
        response = db_client.table("bot_users").select("chat_id").execute()
        if response.data:
            for user in response.data:
                target_chats.add(str(user["chat_id"]))
    except Exception as e:
        logger.error(f"Error fetching subscriber list from Supabase: {e}")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    async with httpx.AsyncClient() as client:
        for chat_id in target_chats:
            try:
                response = await client.post(url, json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }, timeout=10)
                response.raise_for_status()
                result = response.json()
                if not result.get("ok"):
                    logger.error(
                        "Telegram rejected message for chat %s: %s",
                        chat_id,
                        result.get("description", "unknown error"),
                    )
            except Exception as e:
                logger.error(f"Failed sending to chat {chat_id}: {e}")


# ============================================================
# Adaptive Model
# ============================================================
class RunningStats:
    """Keeps a running mean/std per feature so inputs stay on the same scale."""

    def __init__(self, size):
        self.n = 0
        self.mean = [0.0] * size
        self.m2 = [0.0] * size

    def update(self, x):
        self.n += 1
        for i, v in enumerate(x):
            d = v - self.mean[i]
            self.mean[i] += d / self.n
            self.m2[i] += d * (v - self.mean[i])

    def standardize(self, x):
        if self.n < 30:
            return [0.0] * len(x)
        out = []
        for i, v in enumerate(x):
            var = self.m2[i] / (self.n - 1)
            std = math.sqrt(var) if var > 0 else 1.0
            out.append(max(-5.0, min(5.0, (v - self.mean[i]) / (std + 1e-12))))
        return out


class OnlineLogisticModel:
    """Tiny logistic regression that learns one example at a time (no extra libraries)."""

    def __init__(self, n_features, lr=0.02, l2=1e-3):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.stats = RunningStats(n_features)
        self.n_updates = 0

    @staticmethod
    def _sigmoid(z):
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, features):
        x = self.stats.standardize(features)
        return self._sigmoid(self.b + sum(w * v for w, v in zip(self.w, x)))

    def update(self, features, label):
        self.stats.update(features)
        x = self.stats.standardize(features)
        p = self._sigmoid(self.b + sum(w * v for w, v in zip(self.w, x)))
        err = p - label
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (err * x[i] + self.l2 * self.w[i])
        self.b -= self.lr * err
        self.n_updates += 1

    def to_dict(self):
        return {
            "w": self.w,
            "b": self.b,
            "n_updates": self.n_updates,
            "stats_n": self.stats.n,
            "stats_mean": self.stats.mean,
            "stats_m2": self.stats.m2,
        }

    def load_dict(self, d):
        size = len(self.w)
        w, mean, m2 = d["w"], d["stats_mean"], d["stats_m2"]
        if not (len(w) == len(mean) == len(m2) == size):
            raise ValueError("saved model has a different number of features")
        values = list(w) + list(mean) + list(m2) + [d["b"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("saved model contains invalid numbers")
        self.w = [float(v) for v in w]
        self.b = float(d["b"])
        self.n_updates = int(d["n_updates"])
        self.stats.n = int(d["stats_n"])
        self.stats.mean = [float(v) for v in mean]
        self.stats.m2 = [float(v) for v in m2]


class JarvisLearningModule:
    def __init__(self, memory_size=200):
        self.outcome_memory = deque(maxlen=memory_size)
        self.total_wins = 0
        self.total_trades = 0
        self.model = OnlineLogisticModel(n_features=5)

    @staticmethod
    def extract_features(prices):
        n = len(prices)
        if n < 10 or min(prices) <= 0:
            return None

        returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, n)]
        mean_r = sum(returns) / len(returns)
        vol = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
        if vol == 0:
            return None

        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n
        sxx = sum((i - x_mean) ** 2 for i in range(n))
        slope = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n)) / sxx
        price_std = math.sqrt(sum((p - y_mean) ** 2 for p in prices) / n)

        trend = slope / (vol * prices[-1])
        momentum = sum(returns[-5:]) / (vol * math.sqrt(5))
        last_move = returns[-1] / vol
        zscore = (prices[-1] - y_mean) / (price_std + 1e-12)
        vol_bps = min(vol * 1e4, 50.0)

        feats = [trend, momentum, last_move, zscore, vol_bps]
        return [max(-10.0, min(10.0, f)) for f in feats]

    def evaluate_sequence(self, prices):
        features = self.extract_features(prices)
        if features is None or self.model.n_updates < MIN_TRAIN_SAMPLES:
            return 0, 0.0, TRADE_EXPIRATION_SECONDS, "Low"

        edge = self.model.predict_proba(features) - 0.5  # + means "price likely up"
        if abs(edge) < MIN_EDGE:
            return 0, edge, TRADE_EXPIRATION_SECONDS, "Low"

        direction = 1 if edge > 0 else -1
        confidence = self.confidence_band(edge)

        if confidence != "Low":
            band_outcomes = [
                won
                for score, won in self.outcome_memory
                if self.confidence_band(score) == confidence
                and (1 if score > 0 else -1) == direction
            ]
            if (
                len(band_outcomes) >= 20
                and self.wilson_lower_bound(band_outcomes) < 0.45
            ):
                confidence = "Low"

        return direction, edge, TRADE_EXPIRATION_SECONDS, confidence

    @staticmethod
    def confidence_band(score):
        abs_score = abs(score)
        if abs_score >= HIGH_EDGE:
            return "High"
        if abs_score >= MEDIUM_EDGE:
            return "Medium"
        return "Low"

    @staticmethod
    def wilson_lower_bound(outcomes):
        sample_size = len(outcomes)
        if sample_size == 0:
            return 0.0

        win_rate = sum(outcomes) / sample_size
        z_score = 1.96
        denominator = 1 + z_score**2 / sample_size
        center = win_rate + z_score**2 / (2 * sample_size)
        margin = z_score * math.sqrt(
            (win_rate * (1 - win_rate) + z_score**2 / (4 * sample_size))
            / sample_size
        )
        return (center - margin) / denominator

    def _record_outcome(self, score, was_win):
        self.outcome_memory.append((score, was_win))
        self.total_trades += 1
        if was_win:
            self.total_wins += 1

    def learn_from_outcome(self, score, was_win):
        self._record_outcome(score, was_win)
        wins = sum(won for _, won in self.outcome_memory)
        logger.info(
            "[JARVIS] recent_signals=%s signal_win_rate=%.1f%%",
            len(self.outcome_memory),
            100 * wins / len(self.outcome_memory),
        )


# ============================================================
# Data Feed
# ============================================================
class MarketDataFeed:
    """Live EUR/USD prices from Finnhub's WebSocket (needs a free API key)."""

    def __init__(self):
        if not FINNHUB_API_KEY:
            raise RuntimeError("Missing required environment variable: FINNHUB_API_KEY")
        self.price_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.latest_price = None
        self.last_update_at = None
        self._first_tick_logged = False

    async def _sampler(self):
        """Adds the latest price to the buffer once per second (keeps spacing even)."""
        while True:
            await asyncio.sleep(1)
            if self.latest_price is not None and self.has_fresh_price():
                self.price_buffer.append(self.latest_price)

    def _handle_message(self, raw):
        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        message_type = message.get("type")
        if message_type == "error":
            logger.warning("Finnhub error message: %s", message.get("msg"))
            return
        if message_type != "trade":
            return  # ping and anything else

        for trade in message.get("data") or []:
            try:
                if trade.get("s") != FINNHUB_SYMBOL:
                    continue
                price = float(trade["p"])
                if not math.isfinite(price) or price <= 0:
                    raise ValueError("invalid price")
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            self.latest_price = price
            self.last_update_at = datetime.now(timezone.utc)
            if not self._first_tick_logged:
                self._first_tick_logged = True
                logger.info("First %s price received from Finnhub: %.5f", FINNHUB_SYMBOL, price)

    async def start(self):
        sampler = asyncio.create_task(self._sampler())
        delay = 5
        try:
            while True:
                try:
                    url = f"{FINNHUB_WS_URL}?token={FINNHUB_API_KEY}"
                    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": FINNHUB_SYMBOL}))
                        logger.info("Connected to Finnhub; subscribed to %s", FINNHUB_SYMBOL)
                        delay = 5
                        while True:
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=60)
                            except asyncio.TimeoutError:
                                logger.warning(
                                    "No %s prices for 60s (market closed, or forex streaming "
                                    "is not included in your Finnhub plan)",
                                    FINNHUB_SYMBOL,
                                )
                                continue
                            self._handle_message(raw)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    error_text = str(e).replace(FINNHUB_API_KEY, "***")
                    logger.error("Finnhub stream error: %s; reconnecting in %ss", error_text, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        finally:
            sampler.cancel()

    def get_latest_prices(self, n):
        return list(self.price_buffer)[-n:]

    def get_current_price(self):
        return self.price_buffer[-1] if self.price_buffer else None

    def has_fresh_price(self, max_age_seconds=15):
        if self.last_update_at is None:
            return False
        age_seconds = (datetime.now(timezone.utc) - self.last_update_at).total_seconds()
        return 0 <= age_seconds <= max_age_seconds

    def has_full_price_window(self):
        return len(self.price_buffer) == SEQUENCE_LENGTH


# ============================================================
# Main Bot Engine
# ============================================================
class EOLivePredictorBot:
    def __init__(self):
        self.feed = MarketDataFeed()
        self.jarvis = JarvisLearningModule()
        self.subscribers: set[str] = set()
        self.is_running = True
        self.trading_unlocked = False
        self.unlock_time = datetime.now() + timedelta(seconds=OBSERVATION_SECONDS)
        self.last_signal_time = datetime.now() - timedelta(seconds=60)

        missing_settings = [
            name
            for name, value in (
                ("SUPABASE_URL", SUPABASE_URL),
                ("SUPABASE_KEY", SUPABASE_KEY),
                ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
                ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
                ("FINNHUB_API_KEY", FINNHUB_API_KEY),
            )
            if not value
        ]
        if missing_settings:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing_settings)
            )
        if SUPABASE_KEY.startswith(("sb_publishable_", "sb_secret_")):
            raise RuntimeError(
                "supabase==2.7.4 requires a legacy JWT-format API key. "
                "Set SUPABASE_KEY to the project's legacy anon key from "
                "Supabase API settings."
            )

        self.db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase")

    def restore_model_state(self):
        try:
            response = (
                self.db_client.table("model_state")
                .select("state")
                .eq("id", MODEL_STATE_ID)
                .execute()
            )
            if not response.data:
                logger.info("No saved model found; starting fresh")
                return
            self.jarvis.model.load_dict(response.data[0]["state"])
            logger.info(
                "Restored learned model (%s samples)", self.jarvis.model.n_updates
            )
        except Exception as e:
            logger.error("Could not restore saved model, starting fresh: %s", e)

    def save_model_state(self):
        try:
            self.db_client.table("model_state").upsert(
                {
                    "id": MODEL_STATE_ID,
                    "state": self.jarvis.model.to_dict(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="id",
            ).execute()
        except Exception as e:
            logger.error("Could not save model state: %s", e)

    def restore_learning_history(self):
        try:
            response = (
                self.db_client.table("paper_trades")
                .select("asset_pair,pytorch_output,status,executed_at")
                .eq("asset_pair", "EUR/USD (Finnhub reference)")
                .in_("status", ["WIN", "LOSS"])
                .gte("executed_at", LEARNING_HISTORY_START_AT.isoformat())
                .order("executed_at", desc=True)
                .limit(LEARNING_HISTORY_LIMIT)
                .execute()
            )
        except Exception as e:
            logger.error("Could not restore learning history from Supabase: %s", e)
            return

        restored = 0
        for trade in reversed(response.data or []):
            if trade.get("status") not in {"WIN", "LOSS"}:
                continue
            try:
                score = float(trade["pytorch_output"])
                executed_at = datetime.fromisoformat(
                    trade["executed_at"].replace("Z", "+00:00")
                )
                if (
                    executed_at < LEARNING_HISTORY_START_AT
                    or not math.isfinite(score)
                ):
                    continue
                self.jarvis._record_outcome(score, trade["status"] == "WIN")
                restored += 1
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping invalid historical learning record: %s", e)

        logger.info("Restored %s resolved outcomes for the learning window", restored)

    async def monitor_phase(self):
        while self.is_running:
            if not self.trading_unlocked and datetime.now() >= self.unlock_time:
                self.trading_unlocked = True
                await broadcast_telegram(
                    "🚀 <b>Bot is now LIVE</b>\nOnly Medium & High confidence signals will be sent.",
                    self.db_client,
                    self.subscribers,
                )
                logger.warning("SIGNALS ACTIVE - Low confidence ignored")
            await asyncio.sleep(0.5)

    async def telegram_listener_loop(self):
        offset = 0
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"


        while self.is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params={"offset": offset, "timeout": 20}, timeout=25)
                    data = response.json()

                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            offset = update["update_id"] + 1
                            message = update.get("message")

                            if not message:
                                continue

                            text = message.get("text", "")
                            chat_id = message["chat"]["id"]

                            command = text.split(maxsplit=1)[0].split("@", 1)[0].lower() if text else ""

                            if text.startswith("/broadcast ") and str(chat_id) == TELEGRAM_CHAT_ID:
                                announcement = text.replace("/broadcast ", "", 1).strip()
                                if announcement:
                                    formatted_announcement = f"📢 <b>ADMIN ANNOUNCEMENT</b>\n\n{announcement}"
                                    await broadcast_telegram(
                                        formatted_announcement,
                                        self.db_client,
                                        self.subscribers,
                                    )
                                continue

                            if command == "/start":
                                chat_id_str = str(chat_id)
                                self.subscribers.add(chat_id_str)
                                first_name = message["chat"].get("first_name", "Trader")
                                try:
                                    self.db_client.table("bot_users").upsert({
                                        "chat_id": chat_id_str,
                                        "username": first_name
                                    }, on_conflict="chat_id").execute()
                                except Exception as dbe:
                                    logger.error(
                                        "Failed to persist Telegram subscriber %s: %s",
                                        chat_id,
                                        dbe,
                                    )

                                try:
                                    welcome_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                                    response = await client.post(welcome_url, json={
                                        "chat_id": chat_id,
                                        "text": f"👋 Hello {first_name}! You are subscribed to broker signals. Send /stop to unsubscribe."
                                    })
                                    response.raise_for_status()
                                    result = response.json()
                                    if not result.get("ok"):
                                        logger.error(
                                            "Telegram rejected welcome message for chat %s: %s",
                                            chat_id,
                                            result.get("description", "unknown error"),
                                        )
                                except Exception as send_error:
                                    logger.error(
                                        "Failed to send subscription confirmation to chat %s: %s",
                                        chat_id,
                                        send_error,
                                    )
                            elif command == "/stop":
                                chat_id_str = str(chat_id)
                                self.subscribers.discard(chat_id_str)
                                try:
                                    self.db_client.table("bot_users").delete().eq(
                                        "chat_id", chat_id_str
                                    ).execute()
                                except Exception as dbe:
                                    logger.error(
                                        "Failed to remove Telegram subscriber %s: %s",
                                        chat_id,
                                        dbe,
                                    )
                                try:
                                    response = await client.post(
                                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                                        json={
                                            "chat_id": chat_id,
                                            "text": "You have been unsubscribed from broker signals. Send /start to subscribe again.",
                                        },
                                    )
                                    response.raise_for_status()
                                    result = response.json()
                                    if not result.get("ok"):
                                        logger.error(
                                            "Telegram rejected unsubscribe confirmation for chat %s: %s",
                                            chat_id,
                                            result.get("description", "unknown error"),
                                        )
                                except Exception as send_error:
                                    logger.error(
                                        "Failed to send unsubscribe confirmation to chat %s: %s",
                                        chat_id,
                                        send_error,
                                    )
            except Exception as e:
                logger.error(f"Telegram listener error: {e}")

            await asyncio.sleep(2)

    async def log_trade(self, direction, price, score):
        try:
            self.db_client.table("paper_trades").insert({
                "asset_pair": "EUR/USD (Finnhub reference)",
                "direction": direction,
                "entry_price": round(price, 5),
                "pytorch_output": round(score, 4),
                "status": "PENDING"
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Trade insertion failed: {e}")
            return False

    async def execution_loop(self):
        while self.is_running:
            if self.trading_unlocked:
                if not self.feed.has_fresh_price():
                    await asyncio.sleep(1)
                    continue
                if not self.feed.has_full_price_window():
                    await asyncio.sleep(1)
                    continue
                prices = self.feed.get_latest_prices(SEQUENCE_LENGTH)
                if len(prices) >= SEQUENCE_LENGTH:
                    decision, score, expiration, confidence = self.jarvis.evaluate_sequence(prices)
                    price = prices[-1]

                    time_since_last = (datetime.now() - self.last_signal_time).total_seconds()

                    if decision != 0 and confidence != "Low" and time_since_last >= 55:
                        direction = "BUY" if decision == 1 else "SELL"
                        if not await self.log_trade(direction, price, score):
                            await asyncio.sleep(1)
                            continue

                        emoji = "🟢" if decision == 1 else "🔴"
                        msg = (
                            f"{emoji} <b>{direction} SIGNAL</b>\n\n"
                            f"Asset: <b>EUR/USD (Finnhub reference; may differ from EO OTC)</b>\n"
                            f"Entry: <code>{price:.5f}</code>\n"
                            f"Score: <code>{score:.4f}</code>\n"
                            f"⏱️ Time: <b>{expiration}s</b>\n"
                            f"📊 Confidence: <b>{confidence}</b>\n\n"
                            f"{datetime.now().strftime('%H:%M:%S')}"
                        )
                        await broadcast_telegram(msg, self.db_client, self.subscribers)
                        self.last_signal_time = datetime.now()

                        delay = random.uniform(55, 95)
                        await asyncio.sleep(delay)
                        continue

                await asyncio.sleep(1)
            else:
                await asyncio.sleep(1)

    async def learning_loop(self):
        """Learns from EVERY price window (not just trades): predict, wait 30s, check, learn."""
        pending = deque()
        recent_hits = deque(maxlen=500)
        last_sample_at = None
        while self.is_running:
            try:
                if self.feed.has_fresh_price() and self.feed.has_full_price_window():
                    now = datetime.now(timezone.utc)
                    price = self.feed.get_current_price()

                    if (
                        last_sample_at is None
                        or (now - last_sample_at).total_seconds() >= LEARNING_SAMPLE_SECONDS
                    ):
                        features = self.jarvis.extract_features(
                            self.feed.get_latest_prices(SEQUENCE_LENGTH)
                        )
                        if features is not None:
                            model = self.jarvis.model
                            # remember the guess made NOW, so accuracy is measured honestly
                            guess_up = (
                                model.predict_proba(features) >= 0.5
                                if model.n_updates >= 30
                                else None
                            )
                            pending.append((now, features, price, guess_up))
                        last_sample_at = now

                    while pending:
                        sampled_at, features, entry, guess_up = pending[0]
                        age = (now - sampled_at).total_seconds()
                        if age < TRADE_EXPIRATION_SECONDS:
                            break
                        pending.popleft()
                        if age > TRADE_EXPIRATION_SECONDS + 5 or price == entry:
                            continue  # stale sample or no move: skip

                        label = 1 if price > entry else 0
                        model = self.jarvis.model
                        if guess_up is not None:
                            recent_hits.append(guess_up == (label == 1))
                        model.update(features, label)

                        if model.n_updates % MODEL_SAVE_EVERY == 0:
                            self.save_model_state()

                        if model.n_updates % 50 == 0 and recent_hits:
                            logger.info(
                                "[LEARNING] samples=%s honest_accuracy=%.1f%% (last %s predictions)",
                                model.n_updates,
                                100 * sum(recent_hits) / len(recent_hits),
                                len(recent_hits),
                            )
            except Exception:
                logger.exception("Learning loop error")
            await asyncio.sleep(1)

    async def resolve_trades(self):
        while self.is_running:
            try:
                response = self.db_client.table("paper_trades").select("*").eq("status", "PENDING").execute()
                pending = response.data or []
                current = (
                    self.feed.get_current_price()
                    if self.feed.has_fresh_price()
                    else None
                )

                if pending and current:
                    now = datetime.now(timezone.utc)
                    for trade in pending:
                        executed_at = datetime.fromisoformat(trade["executed_at"].replace("Z", "+00:00"))
                        if (now - executed_at).total_seconds() < TRADE_EXPIRATION_SECONDS:
                            continue

                        direction = trade["direction"]
                        entry = float(trade["entry_price"])
                        score = float(trade.get("pytorch_output", 0))
                        was_win = (direction == "BUY" and current > entry) or (direction == "SELL" and current < entry)

                        try:
                            self.db_client.table("paper_trades").update({
                                "status": "WIN" if was_win else "LOSS",
                                "exit_price": round(current, 5)
                            }).eq("id", trade["id"]).execute()

                            if trade.get("asset_pair") == "EUR/USD (Finnhub reference)":
                                self.jarvis.learn_from_outcome(score, was_win)
                        except Exception as update_err:
                            logger.error(f"Failed inside resolution row: {update_err}")
            except Exception as e:
                logger.error(f"Resolve error: {e}")
            await asyncio.sleep(RESOLUTION_POLL_INTERVAL)

    async def start(self):
        self.restore_model_state()
        self.restore_learning_history()
        await broadcast_telegram(
            "🤖 <b>Signal bot started</b>\n"
            "Signals use Finnhub EUR/USD reference prices, which may differ from EO Broker OTC.\n"
            "JARVIS keeps adapting its signal filter from recent resolved Finnhub paper trades.",
            self.db_client,
            self.subscribers,
        )
        await asyncio.gather(
            self.feed.start(),
            self.monitor_phase(),
            self.telegram_listener_loop(),
            self.execution_loop(),
            self.learning_loop(),
            self.resolve_trades()
        )


# ============================================================
# FASTAPI & LIFESPAN MANAGEMENT (CRITICAL FOR RENDER)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    bot_engine = EOLivePredictorBot()
    bot_task = asyncio.create_task(bot_engine.start())
    try:
        yield
    finally:
        bot_engine.is_running = False
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Bot is alive and running!", "timestamp": datetime.now(timezone.utc).isoformat()}