import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class KeyState(str, Enum):
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"
    DEAD = "DEAD"


@dataclass
class KeyInfo:
    key: str
    status: KeyState = KeyState.ACTIVE
    cooldown_until: float = 0.0
    last_error: Optional[str] = None
    request_count: int = 0
    fail_count: int = 0

    @property
    def masked_key(self) -> str:
        if len(self.key) <= 8:
            return "***"
        prefix = self.key[:4]
        suffix = self.key[-4:]
        return f"{prefix}...{suffix}"


class KeyManager:
    """
    Пул API-ключей провайдеров с поддержкой ротации (Round-Robin),
    карантина (COOLDOWN) при 429/5xx и выбывания (DEAD) при 401/403.
    """

    def __init__(self, default_cooldown_sec: float = 60.0):
        self.default_cooldown_sec = default_cooldown_sec
        self._provider_keys: Dict[str, List[KeyInfo]] = {}
        self._rr_indices: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def register_provider_keys(self, provider_id: str, keys: List[str]) -> None:
        async with self._lock:
            existing = {k.key: k for k in self._provider_keys.get(provider_id, [])}
            new_list = []
            for k in keys:
                if k in existing:
                    new_list.append(existing[k])
                else:
                    new_list.append(KeyInfo(key=k))
            self._provider_keys[provider_id] = new_list
            if provider_id not in self._rr_indices:
                self._rr_indices[provider_id] = 0

    async def unregister_provider(self, provider_id: str) -> None:
        async with self._lock:
            self._provider_keys.pop(provider_id, None)
            self._rr_indices.pop(provider_id, None)

    async def get_active_key(self, provider_id: str) -> Optional[str]:
        async with self._lock:
            keys = self._provider_keys.get(provider_id, [])
            if not keys:
                return None

            now = time.time()
            # Обновляем статус тех, чей карантин истек
            available: List[KeyInfo] = []
            for k in keys:
                if k.status == KeyState.COOLDOWN and now >= k.cooldown_until:
                    k.status = KeyState.ACTIVE
                    k.last_error = None
                if k.status == KeyState.ACTIVE:
                    available.append(k)

            if not available:
                return None

            # Round-Robin выбор
            idx = self._rr_indices.get(provider_id, 0) % len(available)
            chosen = available[idx]
            self._rr_indices[provider_id] = (idx + 1) % len(available)
            chosen.request_count += 1
            return chosen.key

    async def mark_cooldown(
        self,
        provider_id: str,
        key: str,
        cooldown_sec: Optional[float] = None,
        error_msg: str = "Rate limited (429)",
    ) -> None:
        async with self._lock:
            sec = cooldown_sec if (cooldown_sec is not None and cooldown_sec > 0) else self.default_cooldown_sec
            for k in self._provider_keys.get(provider_id, []):
                if k.key == key:
                    k.status = KeyState.COOLDOWN
                    k.cooldown_until = time.time() + sec
                    k.last_error = error_msg
                    k.fail_count += 1
                    break

    async def mark_dead(self, provider_id: str, key: str, error_msg: str = "Auth failed (401/403)") -> None:
        async with self._lock:
            for k in self._provider_keys.get(provider_id, []):
                if k.key == key:
                    k.status = KeyState.DEAD
                    k.last_error = error_msg
                    k.fail_count += 1
                    break

    async def reset_all_cooldowns(self) -> None:
        async with self._lock:
            for keys in self._provider_keys.values():
                for k in keys:
                    if k.status == KeyState.COOLDOWN:
                        k.status = KeyState.ACTIVE
                        k.last_error = None

    async def get_provider_status(self, provider_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            keys = self._provider_keys.get(provider_id, [])
            now = time.time()
            result = []
            for k in keys:
                cooldown_remaining = max(0.0, k.cooldown_until - now) if k.status == KeyState.COOLDOWN else 0.0
                status_str = k.status.value
                if k.status == KeyState.COOLDOWN:
                    status_str = f"COOLDOWN ({int(cooldown_remaining)}s)"

                result.append({
                    "masked_key": k.masked_key,
                    "status": status_str,
                    "cooldown_remaining": round(cooldown_remaining, 1),
                    "last_error": k.last_error,
                    "request_count": k.request_count,
                    "fail_count": k.fail_count,
                })
            return result

    async def get_all_status(self) -> Dict[str, List[Dict[str, Any]]]:
        async with self._lock:
            result = {}
            for pid in self._provider_keys.keys():
                # Call internal without re-acquiring lock
                keys = self._provider_keys[pid]
                now = time.time()
                provider_list = []
                for k in keys:
                    cooldown_remaining = max(0.0, k.cooldown_until - now) if k.status == KeyState.COOLDOWN else 0.0
                    status_str = k.status.value
                    if k.status == KeyState.COOLDOWN:
                        status_str = f"COOLDOWN ({int(cooldown_remaining)}s)"
                    provider_list.append({
                        "masked_key": k.masked_key,
                        "status": status_str,
                        "cooldown_remaining": round(cooldown_remaining, 1),
                        "last_error": k.last_error,
                        "request_count": k.request_count,
                        "fail_count": k.fail_count,
                    })
                result[pid] = provider_list
            return result


# Singleton instance
_key_manager: Optional[KeyManager] = None


def get_key_manager(default_cooldown_sec: float = 60.0) -> KeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager(default_cooldown_sec)
    return _key_manager
