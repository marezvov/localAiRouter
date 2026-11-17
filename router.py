import time
import json
import re
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

from config_loader import AppConfig
from key_manager import KeyManager
from session_manager import SessionManager
from local_manager import LocalManager
from discovery import ModelDiscovery
from image_pipeline import process_messages_images, has_images

logger = logging.getLogger("gateway.router")

# Регекс для поиска XML tool call вида <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*({.*?})\s*<\/tool_call>", re.DOTALL)


def extract_session_id(request: Request, body: Dict[str, Any]) -> str:
    """Извлекает session_id из заголовков, extra_body или вычисляет хэш."""
    # 1. Заголовок X-Session-ID
    header_sid = request.headers.get("X-Session-ID") or request.headers.get("x-session-id")
    if header_sid and header_sid.strip():
        return header_sid.strip()

    # 2. extra_body.session_id или session_id в теле
    extra = body.get("extra_body")
    if isinstance(extra, dict) and "session_id" in extra:
        return str(extra["session_id"]).strip()
    if "session_id" in body:
        return str(body["session_id"]).strip()

    # 3. Fallback: SHA256 от клиента и системного промпта
    client_ip = request.client.host if request.client else "127.0.0.1"
    system_text = ""
    messages = body.get("messages", [])
    if isinstance(messages, list) and len(messages) > 0:
        first_msg = messages[0]
        if first_msg.get("role") == "system":
            content = first_msg.get("content", "")
            system_text = str(content)[:300]

    raw = f"{client_ip}:{system_text}".encode("utf-8")
    return f"sess_{hashlib.sha256(raw).hexdigest()[:16]}"


def normalize_qwen_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Парсит возможные XML-теги <tool_call> и формирует структуру OpenAI tool_calls."""
    matches = QWEN_TOOL_CALL_RE.findall(text)
    if not matches:
        return text, []

    tool_calls = []
    clean_text = QWEN_TOOL_CALL_RE.sub("", text).strip()
    for idx, match_json in enumerate(matches):
        try:
            parsed = json.loads(match_json)
            tool_calls.append({
                "id": f"call_local_{int(time.time())}_{idx}",
                "type": "function",
                "function": {
                    "name": parsed.get("name", "unknown"),
                    "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False)
                    if isinstance(parsed.get("arguments"), dict) else str(parsed.get("arguments", "{}"))
                }
            })
        except Exception:
            continue

    return clean_text, tool_calls


class Router:
    """
    Главный диспетчер запросов:
    - Sticky Session Affinity
    - Защита горячего слота и Session-Aware очереди
    - Отказоустойчивый Failover в RAM (обход 429/5xx)
    - Потоковый SSE-стриминг с мгновенным перехватом abort
    - Предобработка картинок
    """

    def __init__(
        self,
        config: AppConfig,
        key_manager: KeyManager,
        session_manager: SessionManager,
        local_manager: LocalManager,
        discovery: ModelDiscovery,
        outbound_proxy: Optional[str] = None,
    ):
        self.config = config
        self.key_manager = key_manager
        self.session_manager = session_manager
        self.local_manager = local_manager
        self.discovery = discovery
        self.outbound_proxy = outbound_proxy

    async def route_chat_completion(self, request: Request, body: Dict[str, Any]) -> Response:
        session_id = extract_session_id(request, body)
        session = await self.session_manager.get_or_create(session_id)

        requested_model = body.get("model", "auto-executor")
        stream = body.get("stream", False)
        req_tools = bool(body.get("tools"))
        messages = body.get("messages", [])
        req_vision = has_images(messages)

        logger.info(
            f"Routing request: model='{requested_model}', session='{session_id}', "
            f"tools={req_tools}, vision={req_vision}, stream={stream}"
        )

        # Составляем список кандидатов (провайдер, модель)
        candidates = self._build_candidate_list(requested_model, session, req_tools, req_vision)
        if not candidates:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"No available models or providers found for '{requested_model}'"}}
            )

        # Выполняем попытки с Failover в RAM
        last_error = "Unknown error"
        for provider_id, model_id in candidates:
            logger.info(f"Attempting backend: provider='{provider_id}', model='{model_id}'")

            # 1. Если локальный провайдер
            if provider_id == "local":
                resp = await self._handle_local_request(
                    request=request,
                    body=body,
                    session_id=session_id,
                    requested_alias=requested_model,
                    stream=stream,
                )
                if resp is not None:
                    # Успешно выполнено
                    await self.session_manager.bind_backend(session_id, "local", self.config.local_backend.model_name)
                    return resp
                else:
                    logger.warning("Local backend busy or unavailable. Falling back to next candidate...")
                    continue

            # 2. Если облачный провайдер
            pcfg = self.config.providers.get(provider_id)
            if not pcfg or not pcfg.enabled:
                continue

            resp, err = await self._handle_cloud_request(
                request=request,
                body=body,
                provider_id=provider_id,
                pcfg=pcfg,
                model_id=model_id,
                session_id=session_id,
                stream=stream,
            )
            if resp is not None:
                await self.session_manager.bind_backend(session_id, provider_id, model_id)
                return resp

            last_error = err or "Cloud provider failed"
            logger.warning(f"Failover from {provider_id}/{model_id}: {last_error}")

        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"All fallback providers failed. Last error: {last_error}"}}
        )

    def _build_candidate_list(
        self,
        requested_model: str,
        session: Any,
        req_tools: bool,
        req_vision: bool,
    ) -> List[Tuple[str, str]]:
        """Формирует цепочку провайдеров и моделей с учетом Model-First, Sticky Affinity и режимов выбора."""
        candidates: List[Tuple[str, str]] = []
        mode = self.config.routing.model_selection_mode
        global_prio = self.config.routing.provider_priority

        # 1. Проверяем Sticky Session Affinity
        if self.config.routing.sticky_sessions and session.active_provider and session.active_model:
            candidates.append((session.active_provider, session.active_model))

        # 2. Если это алиас из config.yaml
        if requested_model in self.config.aliases:
            alias_cfg = self.config.aliases[requested_model]
            provider_prio = alias_cfg.provider_priority or alias_cfg.priority or global_prio

            # 2.1. Model-First: Пользовательские модели, явно заданные в alias.models
            if alias_cfg.models:
                for target_m in alias_cfg.models:
                    matches = self.discovery.find_providers_for_model(target_m, provider_prio)
                    for pid, actual_m in matches:
                        if (pid, actual_m) not in candidates:
                            candidates.append((pid, actual_m))

            # 2.2. Если режим "user_first" (или у алиаса нет списка models) — дополняем эвристикой
            if mode == "user_first" or not alias_cfg.models:
                for pid in provider_prio:
                    if pid == "local":
                        loc_tuple = ("local", self.config.local_backend.model_name)
                        if loc_tuple not in candidates:
                            candidates.append(loc_tuple)
                    else:
                        ranked_models = self.discovery.rank_models_heuristically(
                            provider_id=pid,
                            require_tools=alias_cfg.require_tools or req_tools,
                            require_vision=alias_cfg.require_vision or req_vision,
                            prefer_tags=alias_cfg.prefer_tags,
                        )
                        for m in ranked_models:
                            pair = (pid, m)
                            if pair not in candidates:
                                candidates.append(pair)

        # 3. Если запрашивается конкретная локальная модель
        elif requested_model.lower() == self.config.local_backend.model_name.lower():
            loc_tuple = ("local", self.config.local_backend.model_name)
            if loc_tuple not in candidates:
                candidates.append(loc_tuple)

        # 4. Если запрашивается неизвестная или конкретная облачная модель
        else:
            matches = self.discovery.find_providers_for_model(requested_model, global_prio)
            for pid, actual_m in matches:
                if (pid, actual_m) not in candidates:
                    candidates.append((pid, actual_m))

            # Если модель не обнаружена в каталогах
            if not candidates:
                if mode == "user_first":
                    # В режиме user_first пробуем сквозную отправку на провайдеры в порядке приоритета
                    for pid in global_prio:
                        if pid != "local":
                            pcfg = self.config.providers.get(pid)
                            if pcfg and pcfg.enabled:
                                candidates.append((pid, requested_model))

        return candidates

    async def _handle_local_request(
        self,
        request: Request,
        body: Dict[str, Any],
        session_id: str,
        requested_alias: str,
        stream: bool,
    ) -> Optional[Response]:
        """Обрабатывает запрос к локальному llama-server."""
        # Проверяем: зарезервирован ли слот чужой сессией в рамках Grace Period
        is_reserved = await self.session_manager.is_local_slot_reserved_for_other(session_id)
        if is_reserved and requested_alias != "local-only":
            logger.info(f"Local slot is holding grace period for another session. Bypassing to cloud.")
            return None

        timeout_sec = (
            self.config.local_backend.queue.local_only_timeout_sec
            if requested_alias == "local-only"
            else self.config.local_backend.queue.same_session_timeout_sec
        )

        # Захват слота
        acquired = await self.local_manager.acquire_slot(session_id, timeout_sec)
        if not acquired:
            if requested_alias == "local-only":
                return JSONResponse(
                    status_code=504,
                    content={"error": {"message": f"Local slot busy. Timed out waiting after {timeout_sec}s"}}
                )
            return None

        try:
            # Предобработка картинок для локального CPU
            processed_messages = await process_messages_images(
                messages=body.get("messages", []),
                target_backend="local",
                local_cfg=self.config.image_processing.local,
                cloud_cfg=self.config.image_processing.cloud,
                master_enabled=self.config.image_processing.enabled,
            )

            local_payload = dict(body)
            local_payload["messages"] = processed_messages
            local_payload["model"] = self.config.local_backend.model_name

            # Применяем default_sampling из алиаса если есть
            alias_cfg = self.config.aliases.get(requested_alias)
            if alias_cfg and alias_cfg.default_sampling:
                for k, v in alias_cfg.default_sampling.items():
                    if k not in local_payload:
                        local_payload[k] = v

            url = f"{self.config.local_backend.url.rstrip('/')}/v1/chat/completions"

            if stream:
                return await self._stream_local(request, url, local_payload, session_id)
            else:
                return await self._call_local(url, local_payload, session_id)

        except Exception as e:
            logger.error(f"Error executing local request: {e}")
            self.local_manager.release_slot()
            return None

    async def _stream_local(
        self,
        request: Request,
        url: str,
        payload: Dict[str, Any],
        session_id: str,
    ) -> Response:
        """Потоковая передача от локального бэкенда с отслеживанием обрыва клиента."""
        client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=10.0))

        async def stream_generator() -> AsyncIterator[bytes]:
            try:
                async with client:
                    async with client.stream("POST", url, json=payload) as upstream:
                        async for chunk in upstream.aiter_bytes():
                            # Проверяем: оборвал ли клиент соединение (нажал Stop)
                            if await request.is_disconnected():
                                logger.warning(f"Client disconnected during local stream. Aborting slot 0...")
                                await self.local_manager.abort_slot(0)
                                break
                            yield chunk
            except asyncio.CancelledError:
                logger.warning("CancelledError caught. Aborting local slot 0...")
                await self.local_manager.abort_slot(0)
                raise
            except Exception as e:
                logger.error(f"Error in local stream generator: {e}")
            finally:
                self.local_manager.release_slot()
                await self.session_manager.touch_local_lease(session_id)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    async def _call_local(
        self,
        url: str,
        payload: Dict[str, Any],
        session_id: str,
    ) -> Response:
        """Непотоковый вызов локального бэкенда."""
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    # Нормализация Tool Calls если Qwen вернул их как XML
                    choices = data.get("choices", [])
                    if choices and isinstance(choices, list):
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "")
                        if content and "<tool_call>" in content:
                            clean_text, tool_calls = normalize_qwen_tool_calls(content)
                            msg["content"] = clean_text
                            if tool_calls:
                                msg["tool_calls"] = tool_calls

                    return JSONResponse(status_code=200, content=data)
                else:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
        finally:
            self.local_manager.release_slot()
            await self.session_manager.touch_local_lease(session_id)

    async def _handle_cloud_request(
        self,
        request: Request,
        body: Dict[str, Any],
        provider_id: str,
        pcfg: Any,
        model_id: str,
        session_id: str,
        stream: bool,
    ) -> Tuple[Optional[Response], Optional[str]]:
        """Отправка запроса в облачный API с обработкой ключей и стримингом."""
        key = await self.key_manager.get_active_key(provider_id)
        if not key:
            return None, f"No active keys for provider '{provider_id}' (all in cooldown or dead)"

        # Предобработка картинок для облака
        processed_messages = await process_messages_images(
            messages=body.get("messages", []),
            target_backend="cloud",
            local_cfg=self.config.image_processing.local,
            cloud_cfg=self.config.image_processing.cloud,
            master_enabled=self.config.image_processing.enabled,
        )

        cloud_payload = dict(body)
        cloud_payload["messages"] = processed_messages
        cloud_payload["model"] = model_id

        # Удаляем session_id и специфичные поля роутера перед отправкой в облако
        cloud_payload.pop("extra_body", None)
        cloud_payload.pop("session_id", None)

        url = f"{pcfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            **pcfg.extra_headers,
        }
        proxies = self.outbound_proxy if pcfg.use_proxy else None

        if stream:
            try:
                client = httpx.AsyncClient(timeout=pcfg.timeout_sec, proxy=proxies)
                req = client.build_request("POST", url, headers=headers, json=cloud_payload)
                upstream = await client.send(req, stream=True)

                if upstream.status_code == 200:
                    async def cloud_stream_generator() -> AsyncIterator[bytes]:
                        try:
                            async with client:
                                async for chunk in upstream.aiter_bytes():
                                    if await request.is_disconnected():
                                        break
                                    yield chunk
                        except Exception as e:
                            logger.error(f"Error in cloud stream: {e}")

                    return StreamingResponse(cloud_stream_generator(), media_type="text/event-stream"), None

                # Обработка ошибок статуса
                error_body = await upstream.aread()
                await client.aclose()
                err_text = error_body.decode("utf-8", errors="replace")

                if upstream.status_code == 429:
                    retry_after = upstream.headers.get("Retry-After")
                    sec = float(retry_after) if retry_after and retry_after.isdigit() else self.config.routing.key_cooldown_sec
                    await self.key_manager.mark_cooldown(provider_id, key, sec, f"Rate limited: {err_text[:100]}")
                    return None, f"HTTP 429 Rate Limit from {provider_id}"
                elif upstream.status_code in (401, 403):
                    await self.key_manager.mark_dead(provider_id, key, f"Auth error: {err_text[:100]}")
                    return None, f"HTTP {upstream.status_code} Auth Error from {provider_id}"
                else:
                    return None, f"HTTP {upstream.status_code} from {provider_id}: {err_text[:100]}"

            except Exception as e:
                await self.key_manager.mark_cooldown(provider_id, key, self.config.routing.key_cooldown_sec, str(e))
                return None, str(e)
        else:
            try:
                async with httpx.AsyncClient(timeout=pcfg.timeout_sec, proxy=proxies) as client:
                    resp = await client.post(url, headers=headers, json=cloud_payload)
                    if resp.status_code == 200:
                        return JSONResponse(status_code=200, content=resp.json()), None

                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        sec = float(retry_after) if retry_after and retry_after.isdigit() else self.config.routing.key_cooldown_sec
                        await self.key_manager.mark_cooldown(provider_id, key, sec, f"429: {resp.text[:100]}")
                        return None, f"HTTP 429 Rate Limit from {provider_id}"
                    elif resp.status_code in (401, 403):
                        await self.key_manager.mark_dead(provider_id, key, f"Auth {resp.status_code}: {resp.text[:100]}")
                        return None, f"HTTP {resp.status_code} Auth error from {provider_id}"
                    else:
                        return None, f"HTTP {resp.status_code}: {resp.text[:100]}"
            except Exception as e:
                await self.key_manager.mark_cooldown(provider_id, key, self.config.routing.key_cooldown_sec, str(e))
                return None, str(e)
