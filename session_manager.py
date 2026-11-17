import time
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    active_provider: Optional[str] = None
    active_model: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    local_last_used: float = 0.0
    total_requests: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    In-memory реестр сессий для Sticky Session Affinity и защиты горячего слота.
    Не пишет на диск, работает со скоростью наносекунд.
    """

    def __init__(self, session_ttl_sec: int = 7200, grace_period_sec: float = 60.0):
        self.session_ttl_sec = session_ttl_sec
        self.grace_period_sec = grace_period_sec
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._active_local_session: Optional[str] = None

    async def get_or_create(self, session_id: str) -> SessionState:
        async with self._lock:
            self._cleanup_expired()
            now = time.time()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    created_at=now,
                    last_activity=now,
                )
            session = self._sessions[session_id]
            session.last_activity = now
            session.total_requests += 1
            return session

    async def bind_backend(self, session_id: str, provider: str, model: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.active_provider = provider
                session.active_model = model
                session.last_activity = time.time()
                if provider == "local":
                    session.local_last_used = time.time()
                    self._active_local_session = session_id

    async def touch_local_lease(self, session_id: str) -> None:
        """Обновляет таймштамп использования локального слота для Grace Period."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.local_last_used = time.time()
            self._active_local_session = session_id

    async def is_local_slot_reserved_for_other(self, session_id: str) -> bool:
        """
        Проверяет, удерживается ли локальный слот другой сессией в рамках grace_period_sec.
        Если да, то запрос чужой сессии должен уйти в облако, чтобы не сбить горячий KV-кэш.
        """
        async with self._lock:
            if not self._active_local_session:
                return False
            if self._active_local_session == session_id:
                return False

            other_session = self._sessions.get(self._active_local_session)
            if not other_session:
                self._active_local_session = None
                return False

            now = time.time()
            elapsed = now - other_session.local_last_used
            if elapsed < self.grace_period_sec:
                return True
            else:
                # Grace period истек
                return False

    async def clear_affinity(self, session_id: str) -> None:
        """Сбрасывает привязку к модели при аварийном переключении."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.active_provider = None
                session.active_model = None

    async def get_active_sessions_list(self) -> List[Dict[str, Any]]:
        async with self._lock:
            self._cleanup_expired()
            now = time.time()
            result = []
            for s in sorted(self._sessions.values(), key=lambda x: x.last_activity, reverse=True):
                result.append({
                    "session_id": s.session_id,
                    "active_provider": s.active_provider or "none",
                    "active_model": s.active_model or "none",
                    "idle_sec": round(now - s.last_activity, 1),
                    "total_requests": s.total_requests,
                    "is_holding_local_slot": (
                        self._active_local_session == s.session_id and
                        (now - s.local_last_used) < self.grace_period_sec
                    )
                })
            return result

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_activity) > self.session_ttl_sec
        ]
        for sid in expired:
            if self._active_local_session == sid:
                self._active_local_session = None
            del self._sessions[sid]


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(session_ttl_sec: int = 7200, grace_period_sec: float = 60.0) -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(session_ttl_sec, grace_period_sec)
    return _session_manager
