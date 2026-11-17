import time
import asyncio
import logging
from typing import Dict, Any, Optional, AsyncIterator, Tuple
import httpx

from config_loader import LocalBackendConfig

logger = logging.getLogger("gateway.local")


class LocalManager:
    """
    Управление локальным инференсом (llama-server, http://127.0.0.1:8080):
    - Мониторинг слота 0 (n_past, скорость t/s, занятость)
    - Session-Aware Queueing (FIFO-очередь с таймаутами)
    - Мгновенное прерывание слота 0 при обрыве клиента (action=erase)
    - Сохранение/восстановление и автоочистка KV-кэша по таймеру
    """

    def __init__(self, config: LocalBackendConfig):
        self.config = config
        self._slot_lock = asyncio.Lock()
        self._current_session_id: Optional[str] = None
        self._current_request_start: float = 0.0
        self._last_activity_time: float = time.time()
        self._queue_waiters: int = 0
        self._idle_task: Optional[asyncio.Task] = None

    def start_background_tasks(self) -> None:
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_monitor_loop())

    async def stop_background_tasks(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass

    async def _idle_monitor_loop(self) -> None:
        """Фоновый таймер сброса KV-кэша при простое."""
        while True:
            try:
                await asyncio.sleep(60)
                idle_limit = self.config.cache.idle_clear_sec
                if idle_limit > 0:
                    now = time.time()
                    if not self._slot_lock.locked() and (now - self._last_activity_time) >= idle_limit:
                        logger.info(f"Local backend idle for {idle_limit}s. Auto-clearing KV-cache...")
                        await self.clear_cache()
                        self._last_activity_time = now
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in idle monitor loop: {e}")

    async def check_health(self) -> Dict[str, Any]:
        """Проверяет доступность локального llama-server и состояние слота 0."""
        url = f"{self.config.url.rstrip('/')}/slots"
        timeout_val = self.config.health_timeout_sec
        client_timeout = httpx.Timeout(timeout_val, connect=timeout_val, read=timeout_val, write=timeout_val)
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    # llama-server возвращает массив слотов
                    slot0 = data[0] if isinstance(data, list) and len(data) > 0 else {}
                    n_past = slot0.get("n_past", 0)
                    is_processing = slot0.get("is_processing", False)
                    # Скорость токенов/сек если передается
                    t_s = 0.0
                    if "n_decoded" in slot0 and "t_token" in slot0 and slot0.get("t_token", 0) > 0:
                        t_s = round(1000.0 / slot0["t_token"], 1)

                    return {
                        "online": True,
                        "model": self.config.model_name,
                        "slot_occupied": self._slot_lock.locked() or is_processing,
                        "active_session": self._current_session_id,
                        "n_past": n_past,
                        "tokens_per_sec": t_s,
                        "queue_length": self._queue_waiters,
                        "max_context": self.config.max_context,
                    }
                else:
                    return {"online": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {
                "online": False,
                "error": str(e),
                "model": self.config.model_name,
                "slot_occupied": False,
                "queue_length": self._queue_waiters,
            }

    async def acquire_slot(self, session_id: str, timeout_sec: float) -> bool:
        """
        Попытка захватить слот. Если занят — ждёт в очереди до timeout_sec.
        Возвращает True если слот успешно захвачен, иначе False.
        """
        if self._queue_waiters >= self.config.queue.max_size:
            logger.warning("Local queue max size exceeded.")
            return False

        self._queue_waiters += 1
        try:
            # Ожидание захвата блокировки слота
            acquired = False
            start_wait = time.time()
            while time.time() - start_wait < timeout_sec:
                try:
                    await asyncio.wait_for(self._slot_lock.acquire(), timeout=min(1.0, timeout_sec))
                    acquired = True
                    break
                except asyncio.TimeoutError:
                    continue

            if acquired:
                self._current_session_id = session_id
                self._current_request_start = time.time()
                self._last_activity_time = time.time()
                return True
            return False
        finally:
            self._queue_waiters = max(0, self._queue_waiters - 1)

    def release_slot(self) -> None:
        """Освобождает слот после генерации."""
        self._current_session_id = None
        self._last_activity_time = time.time()
        if self._slot_lock.locked():
            try:
                self._slot_lock.release()
            except RuntimeError:
                pass

    def is_slot_busy(self) -> bool:
        return self._slot_lock.locked()

    def get_current_session(self) -> Optional[str]:
        return self._current_session_id

    async def abort_slot(self, slot_id: int = 0) -> bool:
        """
        Экстренно прерывает вычисления на слоте llama-server.
        Освобождает GPU <500 мс.
        """
        url = f"{self.config.url.rstrip('/')}/slots/{slot_id}?action=erase"
        logger.warning(f"Aborting slot {slot_id} in llama-server...")
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(url)
                logger.info(f"Slot {slot_id} abort response: {resp.status_code}")
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to abort slot {slot_id}: {e}")
            return False
        finally:
            self.release_slot()

    async def clear_cache(self, slot_id: int = 0) -> bool:
        """Сброс KV-кэша слота (action=erase)."""
        url = f"{self.config.url.rstrip('/')}/slots/{slot_id}?action=erase"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Error clearing cache on slot {slot_id}: {e}")
            return False

    async def save_cache(self, filepath: str, slot_id: int = 0) -> bool:
        """Сохранение дампа KV-кэша слота на диск."""
        url = f"{self.config.url.rstrip('/')}/slots/{slot_id}?action=save&filepath={filepath}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Error saving cache for slot {slot_id}: {e}")
            return False

    async def restore_cache(self, filepath: str, slot_id: int = 0) -> bool:
        """Восстановление состояния KV-кэша слота с диска."""
        url = f"{self.config.url.rstrip('/')}/slots/{slot_id}?action=restore&filepath={filepath}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Error restoring cache for slot {slot_id}: {e}")
            return False


_local_manager: Optional[LocalManager] = None


def get_local_manager(config: Optional[LocalBackendConfig] = None) -> LocalManager:
    global _local_manager
    if _local_manager is None:
        if config is None:
            raise ValueError("LocalBackendConfig must be provided on first call")
        _local_manager = LocalManager(config)
    return _local_manager
