import time
import asyncio
import logging
import re
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config_loader
from key_manager import get_key_manager, KeyManager
from session_manager import get_session_manager, SessionManager
from local_manager import get_local_manager, LocalManager
from discovery import ModelDiscovery
from router import Router
import system_monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gateway.main")

START_TIME = time.time()

# Глобальные сервисы
key_mgr: Optional[KeyManager] = None
session_mgr: Optional[SessionManager] = None
local_mgr: Optional[LocalManager] = None
discovery_svc: Optional[ModelDiscovery] = None
router_svc: Optional[Router] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_mgr, session_mgr, local_mgr, discovery_svc, router_svc
    logger.info("Initializing AI Gateway...")

    cfg = config_loader.get_config()
    secrets = config_loader.get_secrets()

    key_mgr = get_key_manager(cfg.routing.key_cooldown_sec)
    # Регистрируем ключи провайдеров из .env
    for pid, pcfg in cfg.providers.items():
        keys = secrets.provider_keys.get(pcfg.api_key_env, [])
        await key_mgr.register_provider_keys(pid, keys)

    session_mgr = get_session_manager(cfg.routing.session_ttl_sec, cfg.local_backend.queue.grace_period_sec)
    local_mgr = get_local_manager(cfg.local_backend)
    local_mgr.start_background_tasks()
    discovery_svc = ModelDiscovery(cfg, key_mgr, secrets.outbound_proxy)

    router_svc = Router(
        config=cfg,
        key_manager=key_mgr,
        session_manager=session_mgr,
        local_manager=local_mgr,
        discovery=discovery_svc,
        outbound_proxy=secrets.outbound_proxy,
    )

    # Фоновое начальное обнаружение моделей
    asyncio.create_task(discovery_svc.refresh_all(force=False))
    logger.info(f"AI Gateway started successfully on {cfg.server.host}:{cfg.server.port}")

    yield

    logger.info("Shutting down AI Gateway...")
    if local_mgr:
        await local_mgr.stop_background_tasks()


app = FastAPI(title="Local-Cloud AI Gateway", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Авторизация по Bearer токену (мастер-ключ шлюза или пароль дашборда)
async def verify_auth(authorization: Optional[str] = Header(None)) -> bool:
    secrets = config_loader.get_secrets()
    valid_tokens = [secrets.gateway_api_key]
    if secrets.dashboard_password:
        valid_tokens.append(secrets.dashboard_password)

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format. Expected 'Bearer <KEY>'")

    token = parts[1]
    if token not in valid_tokens:
        raise HTTPException(status_code=401, detail="Invalid Gateway API Key or Dashboard Password")

    return True


# -------------------------------------------------------------
# OpenAI Compatible Endpoints
# -------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _auth: bool = Depends(verify_auth)):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not router_svc:
        raise HTTPException(status_code=503, detail="Gateway initializing")

    return await router_svc.route_chat_completion(request, body)


@app.get("/v1/models")
async def list_models(_auth: bool = Depends(verify_auth)):
    if not discovery_svc:
        return {"object": "list", "data": []}
    return {
        "object": "list",
        "data": discovery_svc.get_all_models_list(),
    }


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, _auth: bool = Depends(verify_auth)):
    if not discovery_svc:
        raise HTTPException(status_code=404, detail="Model not found")
    all_m = discovery_svc.get_all_models_list()
    for m in all_m:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# -------------------------------------------------------------
# Local Backend Control Endpoints
# -------------------------------------------------------------

@app.get("/v1/local/status")
async def local_status(_auth: bool = Depends(verify_auth)):
    if not local_mgr:
        raise HTTPException(status_code=503, detail="Local manager not initialized")
    return await local_mgr.check_health()


@app.post("/v1/local/cache/clear")
async def clear_local_cache(_auth: bool = Depends(verify_auth)):
    if not local_mgr:
        raise HTTPException(status_code=503, detail="Local manager not initialized")
    success = await local_mgr.clear_cache(0)
    return {"status": "ok" if success else "failed", "action": "clear_cache", "slot": 0}


@app.post("/v1/local/cache/save")
async def save_local_cache(request: Request, _auth: bool = Depends(verify_auth)):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    filepath = body.get("filepath", "/tmp/slot0_cache.bin")
    if not local_mgr:
        raise HTTPException(status_code=503, detail="Local manager not initialized")
    success = await local_mgr.save_cache(filepath, 0)
    return {"status": "ok" if success else "failed", "filepath": filepath}


@app.post("/v1/local/cache/restore")
async def restore_local_cache(request: Request, _auth: bool = Depends(verify_auth)):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    filepath = body.get("filepath", "/tmp/slot0_cache.bin")
    if not local_mgr:
        raise HTTPException(status_code=503, detail="Local manager not initialized")
    success = await local_mgr.restore_cache(filepath, 0)
    return {"status": "ok" if success else "failed", "filepath": filepath}


@app.post("/v1/local/abort")
async def abort_local_slot(_auth: bool = Depends(verify_auth)):
    if not local_mgr:
        raise HTTPException(status_code=503, detail="Local manager not initialized")
    success = await local_mgr.abort_slot(0)
    return {"status": "ok" if success else "failed", "slot": 0}


@app.post("/v1/admin/reload-config")
async def reload_configuration(_auth: bool = Depends(verify_auth)):
    global router_svc, discovery_svc
    new_cfg, new_sec = config_loader.reload_config()

    if key_mgr:
        key_mgr.default_cooldown_sec = new_cfg.routing.key_cooldown_sec
        await key_mgr.reset_all_cooldowns()
        for pid, pcfg in new_cfg.providers.items():
            keys = new_sec.provider_keys.get(pcfg.api_key_env, [])
            await key_mgr.register_provider_keys(pid, keys)

    if local_mgr:
        local_mgr.config = new_cfg.local_backend

    if discovery_svc:
        discovery_svc.config = new_cfg
        discovery_svc.outbound_proxy = new_sec.outbound_proxy

    if router_svc:
        router_svc.config = new_cfg
        router_svc.outbound_proxy = new_sec.outbound_proxy

    return {
        "status": "reloaded",
        "aliases": list(new_cfg.aliases.keys()),
        "providers": list(new_cfg.providers.keys()),
    }


@app.post("/v1/admin/discovery")
async def force_discovery(_auth: bool = Depends(verify_auth)):
    if not discovery_svc:
        raise HTTPException(status_code=503, detail="Discovery not initialized")
    cache = await discovery_svc.refresh_all(force=True)
    return {"status": "ok", "updated_at": cache.get("updated_at")}


async def check_all_keys_live() -> Dict[str, Any]:
    """Выполняет реальные тестовые пинги (/models) для всех ключей и обновляет их статусы."""
    cfg = config_loader.get_config()
    secrets = config_loader.get_secrets()
    results = {}

    for pid, pcfg in cfg.providers.items():
        if not pcfg.enabled:
            continue
        keys = secrets.provider_keys.get(pcfg.api_key_env, [])
        prov_results = []
        for key in keys:
            start = time.time()
            headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", **pcfg.extra_headers}
            proxies = secrets.outbound_proxy if pcfg.use_proxy else None
            try:
                async with httpx.AsyncClient(timeout=8.0, proxy=proxies) as client:
                    resp = await client.get(f"{pcfg.base_url.rstrip('/')}/models", headers=headers)
                    lat = round((time.time() - start) * 1000, 1)
                    if resp.status_code == 200:
                        # Снимаем с карантина если был
                        await key_mgr.mark_cooldown(pid, key, 0, "")
                        prov_results.append({"masked_key": f"{key[:4]}...{key[-4:]}", "status": "ACTIVE", "latency_ms": lat})
                    elif resp.status_code in (401, 403):
                        await key_mgr.mark_dead(pid, key, f"Auth Error {resp.status_code}")
                        prov_results.append({"masked_key": f"{key[:4]}...{key[-4:]}", "status": "DEAD", "error": f"HTTP {resp.status_code}"})
                    elif resp.status_code == 429:
                        await key_mgr.mark_cooldown(pid, key, cfg.routing.key_cooldown_sec, "Rate limited 429")
                        prov_results.append({"masked_key": f"{key[:4]}...{key[-4:]}", "status": "COOLDOWN", "error": "429"})
                    else:
                        prov_results.append({"masked_key": f"{key[:4]}...{key[-4:]}", "status": f"HTTP {resp.status_code}", "latency_ms": lat})
            except Exception as e:
                lat = round((time.time() - start) * 1000, 1)
                await key_mgr.mark_cooldown(pid, key, cfg.routing.key_cooldown_sec, str(e))
                prov_results.append({"masked_key": f"{key[:4]}...{key[-4:]}", "status": "COOLDOWN", "error": str(e)[:30]})
        results[pid] = prov_results
    return results


@app.post("/v1/admin/check-keys")
async def admin_check_keys(_auth: bool = Depends(verify_auth)):
    results = await check_all_keys_live()
    return {"status": "ok", "results": results}


@app.post("/v1/admin/login")
async def admin_login(request: Request):
    """Авторизация в панели администратора по логину и паролю / мастер-ключу."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()

    secrets = config_loader.get_secrets()
    valid_passwords = [secrets.gateway_api_key]
    if secrets.dashboard_password:
        valid_passwords.append(secrets.dashboard_password)

    expected_user = secrets.dashboard_username or "admin"
    if username and username.lower() != expected_user.lower() and username.lower() != "admin":
        raise HTTPException(status_code=401, detail="Неверное имя пользователя")

    if password not in valid_passwords:
        raise HTTPException(status_code=401, detail="Неверный пароль или мастер-ключ шлюза")

    token = secrets.dashboard_password or secrets.gateway_api_key
    return {"status": "ok", "token": token, "username": expected_user}


@app.get("/v1/admin/keys-config")
async def get_keys_config(_auth: bool = Depends(verify_auth)):
    """Возвращает полную конфигурацию ключей и провайдеров для администратора."""
    cfg = config_loader.get_config()
    secrets = config_loader.get_secrets()

    providers_res = {}
    for pid, pcfg in cfg.providers.items():
        keys = secrets.provider_keys.get(pcfg.api_key_env, [])
        providers_res[pid] = {
            "name": pcfg.name,
            "base_url": pcfg.base_url,
            "api_key_env": pcfg.api_key_env,
            "enabled": pcfg.enabled,
            "use_proxy": pcfg.use_proxy,
            "timeout_sec": pcfg.timeout_sec,
            "keys": keys,
            "models": [m.id for m in pcfg.models],
        }

    return {
        "status": "ok",
        "gateway_api_key": secrets.gateway_api_key,
        "dashboard_username": secrets.dashboard_username,
        "has_dashboard_password": bool(secrets.dashboard_password),
        "outbound_proxy": secrets.outbound_proxy or "",
        "providers": providers_res,
    }


@app.post("/v1/admin/save-keys")
async def save_keys(request: Request, _auth: bool = Depends(verify_auth)):
    """Сохраняет измененные ключи и сетевые настройки в .env и обновляет реестр на лету."""
    body = await request.json()
    updates = {}

    if "gateway_api_key" in body and body["gateway_api_key"].strip():
        updates["GATEWAY_API_KEY"] = body["gateway_api_key"].strip()

    if "outbound_proxy" in body:
        updates["OUTBOUND_PROXY"] = body["outbound_proxy"].strip()

    if "dashboard_password" in body and body["dashboard_password"].strip():
        updates["DASHBOARD_PASSWORD"] = body["dashboard_password"].strip()

    provider_keys = body.get("provider_keys", {})
    for env_name, klist in provider_keys.items():
        if isinstance(klist, list):
            updates[env_name] = ",".join([k.strip() for k in klist if k.strip()])
        elif isinstance(klist, str):
            updates[env_name] = klist.strip()

    config_loader.update_env_file(updates)
    new_cfg, new_sec = config_loader.reload_config()

    if key_mgr:
        for pid, pcfg in new_cfg.providers.items():
            keys = new_sec.provider_keys.get(pcfg.api_key_env, [])
            await key_mgr.register_provider_keys(pid, keys)

    if discovery_svc:
        discovery_svc.config = new_cfg
        discovery_svc.outbound_proxy = new_sec.outbound_proxy

    if router_svc:
        router_svc.config = new_cfg
        router_svc.outbound_proxy = new_sec.outbound_proxy

    return {"status": "ok", "message": "Ключи и настройки успешно сохранены в .env"}


@app.post("/v1/admin/save-provider")
async def save_or_delete_provider(request: Request, _auth: bool = Depends(verify_auth)):
    """Добавляет, редактирует или удаляет кастомного поставщика в config.yaml и .env."""
    body = await request.json()
    action = body.get("action", "save")
    provider_id = body.get("provider_id", "").strip().lower()

    if not provider_id or not re.match(r"^[a-z0-9_-]+$", provider_id):
        raise HTTPException(
            status_code=400,
            detail="ID поставщика должен содержать только латинские буквы, цифры, дефис или подчеркивание."
        )

    if action == "delete":
        config_loader.save_provider_to_config(provider_id, {}, remove=True)
        new_cfg, new_sec = config_loader.reload_config()
        if key_mgr:
            await key_mgr.unregister_provider(provider_id)
        if router_svc:
            router_svc.config = new_cfg
        if discovery_svc:
            discovery_svc.config = new_cfg
        return {"status": "ok", "message": f"Поставщик '{provider_id}' успешно удален"}

    name = body.get("name", "").strip() or provider_id.capitalize()
    base_url = body.get("base_url", "").strip().rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="Base URL обязателен")

    api_key_env = body.get("api_key_env", "").strip() or f"{provider_id.upper()}_API_KEYS"
    enabled = bool(body.get("enabled", True))
    use_proxy = bool(body.get("use_proxy", False))
    timeout_sec = float(body.get("timeout_sec", 60.0))

    p_dict = {
        "name": name,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "enabled": enabled,
        "use_proxy": use_proxy,
        "timeout_sec": timeout_sec,
        "models": [],
    }

    config_loader.save_provider_to_config(provider_id, p_dict, remove=False)

    keys = body.get("keys", [])
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    if keys is not None:
        config_loader.update_env_file({api_key_env: ",".join(keys)})

    new_cfg, new_sec = config_loader.reload_config()

    if key_mgr:
        p_keys = new_sec.provider_keys.get(api_key_env, [])
        await key_mgr.register_provider_keys(provider_id, p_keys)

    if router_svc:
        router_svc.config = new_cfg
        router_svc.outbound_proxy = new_sec.outbound_proxy
    if discovery_svc:
        discovery_svc.config = new_cfg
        discovery_svc.outbound_proxy = new_sec.outbound_proxy
        asyncio.create_task(discovery_svc.refresh_all(force=True))

    return {"status": "ok", "message": f"Поставщик '{provider_id}' успешно сохранен", "provider_id": provider_id}


@app.post("/v1/admin/test-key")
async def test_single_key(request: Request, _auth: bool = Depends(verify_auth)):
    """Выполняет прямой микро-тест указанного ключа к указанному эндпоинту."""
    body = await request.json()
    base_url = body.get("base_url", "").strip().rstrip("/")
    key = body.get("key", "").strip()
    use_proxy = bool(body.get("use_proxy", False))

    if not base_url or not key:
        raise HTTPException(status_code=400, detail="base_url and key are required")

    secrets = config_loader.get_secrets()
    proxies = secrets.outbound_proxy if use_proxy else None
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=8.0, proxy=proxies) as client:
            resp = await client.get(f"{base_url}/models", headers=headers)
            lat = round((time.time() - start) * 1000, 1)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    m_count = len(data.get("data", [])) if isinstance(data, dict) else 0
                except Exception:
                    m_count = 0
                return {"ok": True, "status": "ACTIVE", "status_code": 200, "latency_ms": lat, "models_count": m_count}
            elif resp.status_code in (401, 403):
                return {"ok": False, "status": "UNAUTHORIZED", "status_code": resp.status_code, "latency_ms": lat, "error": f"HTTP {resp.status_code} Auth Failed"}
            elif resp.status_code == 429:
                return {"ok": False, "status": "RATE_LIMITED", "status_code": 429, "latency_ms": lat, "error": "HTTP 429 Rate Limited"}
            else:
                return {"ok": False, "status": f"HTTP_{resp.status_code}", "status_code": resp.status_code, "latency_ms": lat, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        lat = round((time.time() - start) * 1000, 1)
        return {"ok": False, "status": "ERROR", "latency_ms": lat, "error": str(e)[:60]}



# -------------------------------------------------------------
# Diagnostic Health & Web Dashboard (/health)
# -------------------------------------------------------------

@app.get("/health")
async def health_check(request: Request):
    accept = request.headers.get("Accept", "").lower()
    check_keys = request.query_params.get("check_keys", "").lower() in ("true", "1") or request.query_params.get("deep", "").lower() in ("true", "1")

    # Если передан флаг ?check_keys=true или ?deep=true — делаем реальный пинг
    live_results = None
    if check_keys:
        live_results = await check_all_keys_live()

    # Сбор данных диагностики
    local_info = await local_mgr.check_health() if local_mgr else {"online": False}
    keys_info = await key_mgr.get_all_status() if key_mgr else {}
    sessions_info = await session_mgr.get_active_sessions_list() if session_mgr else []
    uptime_sec = round(time.time() - START_TIME, 1)
    cfg = config_loader.get_config()
    system_info = system_monitor.get_full_system_metrics()

    data = {
        "status": "healthy",
        "uptime_sec": uptime_sec,
        "local_backend": local_info,
        "providers_keys": keys_info,
        "active_sessions": sessions_info,
        "aliases": {a: cfg.aliases[a].description for a in cfg.aliases},
        "live_check": live_results,
        "system": system_info,
    }

    if "text/html" in accept:
        return HTMLResponse(content=render_health_dashboard_html(data, cfg))
    else:
        return JSONResponse(content=data)


def render_health_dashboard_html(data: Dict[str, Any], cfg: config_loader.AppConfig) -> str:
    master_key = config_loader.get_secrets().gateway_api_key
    loc = data.get("local_backend", {})
    uptime = data.get("uptime_sec", 0)
    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

    loc_online = loc.get("online", False)
    loc_occupied = loc.get("slot_occupied", False)
    loc_n_past = loc.get("n_past", 0)
    loc_ts = loc.get("tokens_per_sec", 0)
    loc_queue = loc.get("queue_length", 0)

    # HTML строк для таблицы ключей
    keys_html = ""
    for pid, klist in data.get("providers_keys", {}).items():
        for k in klist:
            status = k["status"]
            color = "#10b981" if status == "ACTIVE" else ("#f59e0b" if "COOLDOWN" in status else "#ef4444")
            keys_html += f"""
            <tr>
                <td style="font-weight: 600; color: #93c5fd;">{pid}</td>
                <td style="font-family: monospace;">{k['masked_key']}</td>
                <td><span style="background: {color}22; color: {color}; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">{status}</span></td>
                <td>{k['request_count']}</td>
                <td style="color: {'#ef4444' if k['fail_count'] > 0 else '#9ca3af'};">{k['fail_count']}</td>
            </tr>
            """
    if not keys_html:
        keys_html = "<tr><td colspan='5' style='text-align: center; color: #6b7280;'>No cloud provider keys configured</td></tr>"

    # HTML строк для сессий
    sess_html = ""
    for s in data.get("active_sessions", []):
        lock_badge = '<span style="color: #10b981; font-weight: bold;">● Active Slot</span>' if s["is_holding_local_slot"] else '<span style="color: #6b7280;">○ Idle</span>'
        sess_html += f"""
        <tr>
            <td style="font-family: monospace; color: #a78bfa;">{s['session_id']}</td>
            <td>{s['active_provider']} / <span style="color: #60a5fa;">{s['active_model']}</span></td>
            <td>{lock_badge}</td>
            <td>{s['idle_sec']}s ago</td>
            <td>{s['total_requests']}</td>
        </tr>
        """
    if not sess_html:
        sess_html = "<tr><td colspan='5' style='text-align: center; color: #6b7280;'>No active agent sessions</td></tr>"

    # HTML строк для алиасов
    aliases_html = ""
    for aname, adesc in data.get("aliases", {}).items():
        acfg = cfg.aliases.get(aname)
        if acfg and acfg.models:
            prio_str = "Models: " + " &rarr; ".join(acfg.models)
        elif acfg:
            prio_str = "Providers: " + " &rarr; ".join(acfg.provider_priority or acfg.priority or cfg.routing.provider_priority)
        else:
            prio_str = ""
        aliases_html += f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
            <div style="font-weight: bold; color: #38bdf8; font-size: 14px;">{aname}</div>
            <div style="font-size: 12px; color: #94a3b8; margin: 4px 0;">{adesc}</div>
            <div style="font-size: 11px; color: #64748b; font-family: monospace;">Priority: {prio_str}</div>
        </div>
        """

    # --- СИСТЕМНЫЕ МЕТРИКИ И ТЕМПЕРАТУРЫ ---
    sys_metrics = data.get("system", {})
    cpu = sys_metrics.get("cpu", {})
    mem = sys_metrics.get("memory", {})
    gpu = sys_metrics.get("gpu", {})
    disk = sys_metrics.get("disk", {})
    temps = sys_metrics.get("temperatures", [])

    # CPU
    cpu_model = cpu.get("model_name") or "Host CPU"
    cpu_pct = cpu.get("overall_percent", 0.0)
    cpu_cores_str = f"{cpu.get('physical_cores', 1)}C / {cpu.get('logical_cores', 1)}T"
    cpu_freq_str = f"{int(cpu.get('frequency_mhz'))} MHz" if cpu.get("frequency_mhz") else ""
    cpu_temp_val = cpu.get("temperature_c")
    cpu_temp_badge = f"<span class='badge-temp badge-green' style='margin-left: 6px; font-size: 11px;'>{cpu_temp_val}°C</span>" if cpu_temp_val is not None else ""
    cpu_color = "#10b981" if cpu_pct < 60 else ("#f59e0b" if cpu_pct < 85 else "#ef4444")

    # Сетка ядер CPU
    per_core = cpu.get("per_core_percent", [])
    cores_html = ""
    for idx, cp in enumerate(per_core):
        c_color = "#10b981" if cp < 60 else ("#f59e0b" if cp < 85 else "#ef4444")
        cores_html += f"""
        <div class="core-box">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 4px;">
                <span>Core {idx}</span>
                <span style="font-weight: 700; color: #f8fafc;">{cp}%</span>
            </div>
            <div class="progress-bg" style="height: 6px; margin-top: 0;">
                <div class="progress-bar" style="width: {cp}%; background: {c_color};"></div>
            </div>
        </div>
        """
    if not cores_html:
        cores_html = "<div style='color: #64748b; font-size: 13px; grid-column: 1 / -1;'>Per-core load data not exposed by OS</div>"

    # Memory
    ram_used = mem.get("ram_used_gb", 0.0)
    ram_total = mem.get("ram_total_gb", 0.0)
    ram_avail = mem.get("ram_available_gb", 0.0)
    ram_pct = mem.get("ram_percent", 0.0)
    ram_color = "#10b981" if ram_pct < 70 else ("#f59e0b" if ram_pct < 90 else "#ef4444")

    swap_used = mem.get("swap_used_gb", 0.0)
    swap_total = mem.get("swap_total_gb", 0.0)
    swap_pct = mem.get("swap_percent", 0.0)
    swap_sub = f" | Swap: {swap_used}/{swap_total} GB ({swap_pct}%)" if swap_total > 0 else ""

    # GPU
    gpu_avail = gpu.get("available", False)
    gpu_name = gpu.get("name", "No accelerator detected")
    gpu_vendor = gpu.get("vendor", "Unknown")
    gpu_pct = gpu.get("gpu_percent", 0.0)
    gpu_color = "#10b981" if gpu_pct < 70 else ("#f59e0b" if gpu_pct < 90 else "#ef4444")
    vram_used = gpu.get("vram_used_gb", 0.0)
    vram_total = gpu.get("vram_total_gb", 0.0)
    vram_pct = gpu.get("vram_percent", 0.0)
    vram_color = "#10b981" if vram_pct < 70 else ("#f59e0b" if vram_pct < 90 else "#ef4444")
    gpu_power = f"{gpu.get('power_watts')} W" if gpu.get('power_watts') is not None else "N/A"

    # Динамические блоки для карточки GPU
    gpu_boxes_html = ""
    if gpu_avail:
        gpu_boxes_html += f"""
        <div class="gpu-stat-box">
            <div class="gpu-stat-title">GPU Core Activity</div>
            <div class="gpu-stat-val" style="color: {gpu_color};">{gpu_pct}%</div>
        </div>
        """
        if vram_total > 0:
            gpu_boxes_html += f"""
            <div class="gpu-stat-box">
                <div class="gpu-stat-title">VRAM Memory</div>
                <div class="gpu-stat-val">{vram_used} / {vram_total} GB</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">{vram_pct}% allocated</div>
            </div>
            """
        if gpu_power != "N/A":
            gpu_boxes_html += f"""
            <div class="gpu-stat-box">
                <div class="gpu-stat-title">Power Draw</div>
                <div class="gpu-stat-val" style="color: #f59e0b;">{gpu_power}</div>
            </div>
            """
        gpu_sensors = gpu.get("sensors", {})
        if gpu_sensors:
            for s_name, s_val in gpu_sensors.items():
                s_color = "#38bdf8" if any(k in s_name.lower() for k in ("edge", "gpu")) else ("#fb923c" if any(k in s_name.lower() for k in ("junction", "hotspot")) else "#a78bfa")
                gpu_boxes_html += f"""
                <div class="gpu-stat-box">
                    <div class="gpu-stat-title">{s_name} Temp</div>
                    <div class="gpu-stat-val" style="color: {s_color};">{s_val}°C</div>
                </div>
                """
        elif gpu.get("temp_c") is not None:
            gpu_boxes_html += f"""
            <div class="gpu-stat-box">
                <div class="gpu-stat-title">GPU Temperature</div>
                <div class="gpu-stat-val" style="color: #38bdf8;">{gpu['temp_c']}°C</div>
            </div>
            """
    else:
        gpu_boxes_html = """
        <div style="grid-column: 1 / -1; color: #64748b; font-size: 13px; padding: 12px;">
            No discrete or integrated hardware accelerator detected on this system.
        </div>
        """

    # Динамическая сводка хоста для первой карточки
    cpu_short = cpu_model.split("@")[0].replace("Processor", "").replace("Core(TM)", "").strip()
    hw_parts = []
    if cpu_short:
        hw_parts.append(cpu_short)
    if gpu_avail and gpu_name and "no " not in gpu_name.lower():
        hw_parts.append(gpu_name)
    host_hw_str = " + ".join(hw_parts) if hw_parts else "Dynamic System Host"

    # Disk
    disk_path = disk.get("path", "/")
    disk_used = disk.get("used_gb", 0.0)
    disk_total = disk.get("total_gb", 0.0)
    disk_free = disk.get("free_gb", 0.0)
    disk_pct = disk.get("percent", 0.0)
    disk_color = "#10b981" if disk_pct < 75 else ("#f59e0b" if disk_pct < 90 else "#ef4444")

    # Температуры таблица
    temps_html = ""
    for t in temps:
        c_val = t.get("current_c", 0.0)
        h_val = f"{t.get('high_c')}°C" if t.get('high_c') else "-"
        cr_val = f"{t.get('critical_c')}°C" if t.get('critical_c') else "-"
        st = t.get("status", "normal")
        badge_cls = "badge-green" if st == "normal" else ("badge-yellow" if st == "warning" else "badge-red")
        temps_html += f"""
        <tr>
            <td style="font-weight: 600; color: #93c5fd; font-family: monospace;">{t.get('chip')}</td>
            <td style="color: #e2e8f0;">{t.get('label')}</td>
            <td><span class="badge-temp {badge_cls}">{c_val}°C</span></td>
            <td style="color: #94a3b8; font-family: monospace;">{h_val}</td>
            <td style="color: #ef4444; font-family: monospace;">{cr_val}</td>
            <td><span class="badge-temp {badge_cls}" style="font-size: 11px;">{st.upper()}</span></td>
        </tr>
        """
    if not temps_html:
        temps_html = """
        <tr>
            <td colspan="6" style="text-align: center; color: #64748b; padding: 22px;">
                No hardware thermal sensors exposed by current environment.<br>
                <span style="font-size: 12px; color: #475569;">
                    (Sensors are read dynamically from OS drivers: Linux hwmon / coretemp / thermal_zone, etc.).
                </span>
            </td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Gateway Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: #0f172a; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #1e293b; padding-bottom: 16px; }}
        .title {{ font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background: #1e293b; border: 1px solid #334155; padding: 6px 12px; border-radius: 6px; font-size: 13px; color: #94a3b8; }}

        .tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 10px; }}
        .tab-btn {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 6px; }}
        .tab-btn:hover {{ background: #334155; color: #f8fafc; }}
        .tab-btn.active {{ background: #0284c7; color: #ffffff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.25); }}
        .tab-pane {{ display: block; }}

        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; }}
        .card-title {{ font-size: 13px; text-transform: uppercase; color: #94a3b8; font-weight: 600; margin-bottom: 8px; }}
        .card-val {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
        .card-sub {{ font-size: 12px; color: #64748b; margin-top: 6px; }}

        .table-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; margin-bottom: 24px; }}
        .table-card h3 {{ font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 14px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ color: #94a3b8; padding: 10px 12px; border-bottom: 1px solid #334155; font-weight: 600; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; }}

        .actions {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
        button.action-btn {{ background: #0284c7; color: white; border: none; border-radius: 6px; padding: 9px 16px; font-size: 13px; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
        button.action-btn:hover {{ background: #0369a1; }}
        .btn-danger {{ background: #dc2626; }}
        .btn-danger:hover {{ background: #b91c1c; }}
        .btn-secondary {{ background: #334155; }}
        .btn-secondary:hover {{ background: #475569; }}

        .progress-bg {{ background: #0f172a; border-radius: 4px; height: 8px; width: 100%; margin-top: 8px; overflow: hidden; border: 1px solid #334155; }}
        .progress-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s ease; }}
        .cores-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 8px; }}
        .core-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; }}

        .badge-temp {{ padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; font-family: monospace; display: inline-block; }}
        .badge-green {{ background: #064e3b; color: #34d399; border: 1px solid #059669; }}
        .badge-yellow {{ background: #78350f; color: #fcd34d; border: 1px solid #d97706; }}
        .badge-red {{ background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }}

        .gpu-stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 10px; }}
        .gpu-stat-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; }}
        .gpu-stat-title {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; margin-bottom: 4px; }}
        .gpu-stat-val {{ font-size: 18px; font-weight: 700; color: #f8fafc; }}

        .admin-input {{ width: 100%; padding: 8px 12px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #f1f5f9; font-family: monospace; font-size: 13px; box-sizing: border-box; }}
        .admin-input:focus {{ border-color: #38bdf8; outline: none; }}
        .modal-overlay {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.75); display: flex; align-items: center; justify-content: center; z-index: 1000; }}
        .modal-content {{ background: #1e293b; border: 1px solid #475569; border-radius: 12px; width: 100%; max-width: 540px; padding: 24px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }}
        .key-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
        .badge-ping {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; font-family: monospace; display: inline-block; }}
        .badge-ping-ok {{ background: #064e3b; color: #34d399; border: 1px solid #059669; }}
        .badge-ping-err {{ background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <span>&#9889;</span> AI Gateway Dashboard
            </div>
            <div style="display: flex; gap: 8px;">
                <div class="badge">Uptime: <strong id="val-uptime">{uptime_str}</strong></div>
                <div class="badge">Port: <strong>{cfg.server.port}</strong></div>
            </div>
        </div>

        <div class="tabs">
            <button id="btn-tab-gateway" class="tab-btn active" onclick="switchTab('gateway')">
                &#9889; Шлюз и Модели
            </button>
            <button id="btn-tab-system" class="tab-btn" onclick="switchTab('system')">
                &#128421; Система и Температуры
            </button>
            <button id="btn-tab-keys" class="tab-btn" onclick="switchTab('keys')">
                &#128273; Ключи и Провайдеры
            </button>
        </div>

        <!-- TAB 1: GATEWAY & MODELS -->
        <div id="tab-gateway" class="tab-pane">
            <div class="actions">
                <button class="action-btn btn-danger" onclick="apiAction('/v1/local/cache/clear', 'Clear KV-Cache')">&#128465; Clear KV-Cache</button>
                <button class="action-btn" style="background: #059669;" onclick="apiAction('/v1/admin/check-keys', 'Ping Keys')">&#9889; Ping All Keys</button>
                <button class="action-btn" onclick="apiAction('/v1/admin/discovery', 'Refresh Models')">&#128260; Refresh Models</button>
                <button class="action-btn btn-secondary" onclick="apiAction('/v1/admin/reload-config', 'Reload Config')">&#9881; Reload config.yaml</button>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="card-title">Local Inference Node</div>
                    <div id="val-loc-status" class="card-val" style="color: {'#10b981' if loc_online else '#ef4444'};">
                        {('ONLINE' if loc_online else 'OFFLINE')}
                    </div>
                    <div id="val-loc-sub" class="card-sub">{cfg.local_backend.model_name} ({host_hw_str})</div>
                </div>
                <div class="card">
                    <div class="card-title">Slot 0 State</div>
                    <div id="val-slot-status" class="card-val" style="color: {'#f59e0b' if loc_occupied else '#10b981'};">
                        {('BUSY (Generating)' if loc_occupied else 'IDLE (Ready)')}
                    </div>
                    <div id="val-slot-queue" class="card-sub">Queue Waiters: {loc_queue}</div>
                </div>
                <div class="card">
                    <div class="card-title">KV-Cache & Speed</div>
                    <div id="val-loc-npast" class="card-val">{loc_n_past:,} <span style="font-size: 15px; font-weight: normal; color: #94a3b8;">tokens</span></div>
                    <div id="val-loc-speed" class="card-sub">Generation Speed: {loc_ts} t/s</div>
                </div>
                <div class="card">
                    <div class="card-title">Active Agent Sessions</div>
                    <div id="val-sess-count" class="card-val">{len(data.get('active_sessions', []))}</div>
                    <div class="card-sub">In-Memory Sticky Routing</div>
                </div>
            </div>

            <div class="table-card">
                <h3>&#128273; Cloud Provider Keys & Health</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Provider</th>
                            <th>Masked API Key</th>
                            <th>Status</th>
                            <th>Requests</th>
                            <th>Failures</th>
                        </tr>
                    </thead>
                    <tbody id="tbody-keys">
                        {keys_html}
                    </tbody>
                </table>
            </div>

            <div class="table-card">
                <h3>&#129302; Active Agent Tasks & Sessions</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Session ID</th>
                            <th>Affinity (Provider / Model)</th>
                            <th>Local Slot Hold</th>
                            <th>Last Active</th>
                            <th>Total Req</th>
                        </tr>
                    </thead>
                    <tbody id="tbody-sessions">
                        {sess_html}
                    </tbody>
                </table>
            </div>

            <div class="table-card">
                <h3>&#127919; Virtual Model Aliases</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 10px;">
                    {aliases_html}
                </div>
            </div>
        </div>

        <!-- TAB 2: SYSTEM & TEMPERATURES -->
        <div id="tab-system" class="tab-pane" style="display: none;">
            <div class="grid">
                <div class="card">
                    <div class="card-title" title="{cpu_model}">CPU ({cpu_short or cpu_model})</div>
                    <div id="val-cpu-pct" class="card-val" style="color: {cpu_color};">{cpu_pct}%</div>
                    <div class="progress-bg">
                        <div id="bar-cpu-pct" class="progress-bar" style="width: {cpu_pct}%; background: {cpu_color};"></div>
                    </div>
                    <div id="sub-cpu-info" class="card-sub">{cpu_cores_str} {f'| {cpu_freq_str}' if cpu_freq_str else ''} {cpu_temp_badge}</div>
                </div>

                <div class="card">
                    <div class="card-title">System RAM Memory</div>
                    <div id="val-ram-used" class="card-val">{ram_used} / {ram_total} GB</div>
                    <div class="progress-bg">
                        <div id="bar-ram-pct" class="progress-bar" style="width: {ram_pct}%; background: {ram_color};"></div>
                    </div>
                    <div id="sub-ram-info" class="card-sub">{ram_pct}% used (Free: {ram_avail} GB){swap_sub}</div>
                </div>

                <div class="card">
                    <div class="card-title">GPU: {gpu_name}</div>
                    <div id="val-gpu-pct" class="card-val" style="color: {'#10b981' if gpu_avail else '#64748b'};">
                        {f'{gpu_pct}%' if (gpu_avail and gpu_pct > 0) else ('ACTIVE' if gpu_avail else 'N/A')}
                    </div>
                    <div class="progress-bg">
                        <div id="bar-vram-pct" class="progress-bar" style="width: {vram_pct}%; background: {vram_color};"></div>
                    </div>
                    <div id="sub-gpu-info" class="card-sub">
                        {f'VRAM: {vram_used}/{vram_total} GB ({vram_pct}%)' if (gpu_avail and vram_total > 0) else gpu_name}
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">Disk Storage ({disk_path})</div>
                    <div id="val-disk-used" class="card-val">{disk_used} / {disk_total} GB</div>
                    <div class="progress-bg">
                        <div id="bar-disk-pct" class="progress-bar" style="width: {disk_pct}%; background: {disk_color};"></div>
                    </div>
                    <div id="sub-disk-info" class="card-sub">{disk_pct}% used (Free: {disk_free} GB)</div>
                </div>
            </div>

            <!-- CPU PER-CORE LOAD -->
            <div class="table-card">
                <h3>&#9881; CPU Per-Core Load ({cpu_cores_str})</h3>
                <div id="grid-cores" class="cores-grid">
                    {cores_html}
                </div>
            </div>

            <!-- GPU DETAILS -->
            <div class="table-card">
                <h3>&#127918; Hardware Accelerator Details ({gpu_name})</h3>
                <div id="grid-gpu-stats" class="gpu-stat-grid">
                    {gpu_boxes_html}
                </div>
            </div>

            <!-- TEMPERATURES TABLE -->
            <div class="table-card">
                <h3>&#127777; Hardware Thermal Sensors (Все аппаратные термодатчики)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Chip / Device</th>
                            <th>Sensor Zone / Rail</th>
                            <th>Current Temp</th>
                            <th>High Threshold</th>
                            <th>Critical Threshold</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="tbody-temps">
                        {temps_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 3: KEYS & PROVIDERS MANAGEMENT -->
        <div id="tab-keys" class="tab-pane" style="display: none;">
            <!-- State 1: Locked / Login Box -->
            <div id="keys-auth-box" style="max-width: 480px; margin: 40px auto; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 30px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.3);">
                <div style="font-size: 42px; margin-bottom: 12px;">&#128274;</div>
                <h3 style="font-size: 19px; color: #f8fafc; margin-bottom: 8px;">Панель администратора</h3>
                <p style="font-size: 13px; color: #94a3b8; line-height: 1.5; margin-bottom: 22px;">
                    Управление ключами, поставщиками и секретами защищено. Введите мастер-ключ шлюза (GATEWAY_API_KEY) или пароль админа.
                </p>
                <div style="margin-bottom: 16px; text-align: left;">
                    <label style="font-size: 12px; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 6px;">Мастер-ключ или пароль:</label>
                    <input type="password" id="auth-pwd-input" placeholder="Введите ключ или пароль..." class="admin-input" style="font-size: 14px; padding: 10px 14px;" onkeydown="if(event.key==='Enter') doLogin()">
                </div>
                <div id="auth-error-msg" style="display: none; background: #7f1d1d33; border: 1px solid #ef4444; border-radius: 6px; padding: 8px; color: #f87171; font-size: 12px; margin-bottom: 14px;"></div>
                <button class="action-btn" style="width: 100%; padding: 11px; font-size: 14px;" onclick="doLogin()">&#128273; Войти в панель управления</button>
            </div>

            <!-- State 2: Authenticated Keys & Providers Management Panel -->
            <div id="keys-admin-panel" style="display: none;">
                <!-- Admin Header Bar -->
                <div style="display: flex; justify-content: space-between; align-items: center; background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="color: #10b981; font-size: 18px;">&#9679;</span>
                        <span style="font-size: 14px; font-weight: 600; color: #f8fafc;">Сессия администратора активна</span>
                    </div>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <button class="action-btn" style="background: #059669;" onclick="openNewProviderModal()">&#10133; Добавить поставщика</button>
                        <button class="action-btn" style="background: #0284c7;" onclick="saveAllKeysAndSecrets()">&#128190; Сохранить все ключи в .env</button>
                        <button class="action-btn btn-secondary" onclick="doLogout()">&#128682; Выйти</button>
                    </div>
                </div>

                <!-- Global Gateway Secrets Card -->
                <div class="table-card">
                    <h3>&#9881; Системные секреты и шлюз (.env)</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 10px;">
                        <div>
                            <label style="font-size: 12px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 6px;">GATEWAY_API_KEY (Мастер-ключ)</label>
                            <div style="display: flex; gap: 6px;">
                                <input type="password" id="sys-gateway-key" class="admin-input" style="flex: 1;">
                                <button class="action-btn btn-secondary" onclick="toggleInputVisibility('sys-gateway-key')">&#128065;</button>
                            </div>
                        </div>
                        <div>
                            <label style="font-size: 12px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 6px;">OUTBOUND_PROXY (Исходящий прокси)</label>
                            <input type="text" id="sys-outbound-proxy" placeholder="например socks5://127.0.0.1:1080" class="admin-input">
                        </div>
                        <div>
                            <label style="font-size: 12px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 6px;">DASHBOARD_PASSWORD (Пароль дашборда)</label>
                            <div style="display: flex; gap: 6px;">
                                <input type="password" id="sys-dash-pwd" placeholder="Оставьте пустым для входа по мастер-ключу" class="admin-input" style="flex: 1;">
                                <button class="action-btn btn-secondary" onclick="toggleInputVisibility('sys-dash-pwd')">&#128065;</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Providers Cards Container -->
                <div id="providers-cards-container">
                    <div style="color: #64748b; padding: 20px; text-align: center;">Загрузка конфигурации поставщиков...</div>
                </div>
            </div>
        </div>

        <!-- MODAL: ADD NEW CUSTOM PROVIDER -->
        <div id="modal-new-provider" class="modal-overlay" style="display: none;">
            <div class="modal-content">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <h3 style="font-size: 17px; font-weight: 700; color: #38bdf8;">&#10133; Добавить нового поставщика моделей</h3>
                    <button class="action-btn btn-secondary" style="padding: 4px 8px;" onclick="closeNewProviderModal()">✕</button>
                </div>
                <div style="display: flex; flex-direction: column; gap: 14px;">
                    <div>
                        <label style="font-size: 12px; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">ID поставщика (латиница, без пробелов):</label>
                        <input type="text" id="new-p-id" placeholder="например: deepseek, together, mistral" class="admin-input" oninput="autoUpdateEnvVarName()">
                    </div>
                    <div>
                        <label style="font-size: 12px; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Отображаемое имя:</label>
                        <input type="text" id="new-p-name" placeholder="например: DeepSeek Official" class="admin-input">
                    </div>
                    <div>
                        <label style="font-size: 12px; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Base URL (OpenAI-совместимый эндпоинт):</label>
                        <input type="text" id="new-p-url" placeholder="например: https://api.deepseek.com/v1" class="admin-input">
                    </div>
                    <div>
                        <label style="font-size: 12px; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Переменная в .env для ключей:</label>
                        <input type="text" id="new-p-env" placeholder="DEEPSEEK_API_KEYS" class="admin-input">
                    </div>
                    <div>
                        <label style="font-size: 12px; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">API Ключи (через запятую, если несколько):</label>
                        <textarea id="new-p-keys" placeholder="sk-..." class="admin-input" style="height: 60px; resize: vertical;"></textarea>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <input type="checkbox" id="new-p-proxy" style="cursor: pointer; width: 16px; height: 16px;">
                        <label for="new-p-proxy" style="font-size: 13px; color: #e2e8f0; cursor: pointer;">Использовать исходящий прокси (OUTBOUND_PROXY)</label>
                    </div>
                </div>
                <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px;">
                    <button class="action-btn btn-secondary" onclick="closeNewProviderModal()">Отмена</button>
                    <button class="action-btn" style="background: #059669;" onclick="submitNewProvider()">✓ Добавить и активировать</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleInputVisibility(id) {{
            const el = document.getElementById(id);
            if (!el) return;
            el.type = el.type === 'password' ? 'text' : 'password';
        }}

        function getAuthToken() {{
            return sessionStorage.getItem('gateway_auth_token') || '';
        }}

        function setAuthToken(token) {{
            if (token) {{
                sessionStorage.setItem('gateway_auth_token', token);
            }} else {{
                sessionStorage.removeItem('gateway_auth_token');
            }}
        }}

        function switchTab(tabId) {{
            try {{
                localStorage.setItem('active_tab', tabId);
            }} catch(e) {{}}
            document.querySelectorAll('.tab-pane').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

            const pane = document.getElementById('tab-' + tabId);
            const btn = document.getElementById('btn-tab-' + tabId);
            if (pane) pane.style.display = 'block';
            if (btn) btn.classList.add('active');

            if (tabId === 'keys') {{
                checkKeysAuth();
            }}
        }}

        function checkKeysAuth() {{
            const token = getAuthToken();
            const authBox = document.getElementById('keys-auth-box');
            const adminPanel = document.getElementById('keys-admin-panel');
            if (!token) {{
                if (authBox) authBox.style.display = 'block';
                if (adminPanel) adminPanel.style.display = 'none';
            }} else {{
                if (authBox) authBox.style.display = 'none';
                if (adminPanel) adminPanel.style.display = 'block';
                loadKeysConfig();
            }}
        }}

        async function doLogin() {{
            const pwdInput = document.getElementById('auth-pwd-input');
            const errMsg = document.getElementById('auth-error-msg');
            const pwd = (pwdInput ? pwdInput.value : '').trim();
            if (!pwd) {{
                if (errMsg) {{ errMsg.innerText = 'Пожалуйста, введите пароль или мастер-ключ'; errMsg.style.display = 'block'; }}
                return;
            }}
            try {{
                const res = await fetch('/v1/admin/login', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ password: pwd }})
                }});
                const data = await res.json();
                if (res.ok && data.token) {{
                    setAuthToken(data.token);
                    if (errMsg) errMsg.style.display = 'none';
                    checkKeysAuth();
                }} else {{
                    if (errMsg) {{ errMsg.innerText = data.detail || 'Неверный пароль или мастер-ключ'; errMsg.style.display = 'block'; }}
                }}
            }} catch (e) {{
                if (errMsg) {{ errMsg.innerText = 'Ошибка подключения: ' + e; errMsg.style.display = 'block'; }}
            }}
        }}

        function doLogout() {{
            setAuthToken('');
            checkKeysAuth();
        }}

        let activeKeysConfig = null;

        async function loadKeysConfig() {{
            const token = getAuthToken();
            if (!token) return;
            try {{
                const res = await fetch('/v1/admin/keys-config', {{
                    headers: {{ 'Authorization': 'Bearer ' + token }}
                }});
                if (res.status === 401) {{
                    doLogout();
                    return;
                }}
                const data = await res.json();
                activeKeysConfig = data;
                renderKeysConfig(data);
            }} catch (e) {{
                console.error('Failed to load keys config:', e);
            }}
        }}

        function renderKeysConfig(data) {{
            const gwKeyInput = document.getElementById('sys-gateway-key');
            const proxyInput = document.getElementById('sys-outbound-proxy');
            const dashPwdInput = document.getElementById('sys-dash-pwd');

            if (gwKeyInput) gwKeyInput.value = data.gateway_api_key || '';
            if (proxyInput) proxyInput.value = data.outbound_proxy || '';
            if (dashPwdInput && data.has_dashboard_password) dashPwdInput.placeholder = 'Пароль установлен (оставьте пустым чтобы не менять)';

            const container = document.getElementById('providers-cards-container');
            if (!container) return;

            let html = '';
            const providers = data.providers || {{}};
            const pids = Object.keys(providers);

            if (pids.length === 0) {{
                container.innerHTML = '<div style="color: #64748b; padding: 20px; text-align: center;">Нет настроенных поставщиков. Нажмите «Добавить поставщика».</div>';
                return;
            }}

            for (const pid of pids) {{
                const p = providers[pid];
                const keys = p.keys || [];
                const isEnabled = p.enabled !== false;
                const useProxy = p.use_proxy === true;

                let keysRowsHtml = '';
                if (keys.length === 0) {{
                    keysRowsHtml = '<div style="font-size: 12px; color: #64748b; margin-bottom: 10px;">Нет активных ключей в .env</div>';
                }} else {{
                    for (let i = 0; i < keys.length; i++) {{
                        const k = keys[i];
                        keysRowsHtml += `
                        <div class="key-row" id="row-key-${{pid}}-${{i}}">
                            <input type="password" id="input-key-${{pid}}-${{i}}" value="${{k}}" class="admin-input" style="flex: 1; font-family: monospace;">
                            <button class="action-btn btn-secondary" style="padding: 6px 10px;" onclick="toggleInputVisibility('input-key-${{pid}}-${{i}}')">&#128065;</button>
                            <button class="action-btn" style="background: #0284c7; padding: 6px 12px;" onclick="testKeyLive('${{pid}}', ${{i}})">&#9889; Тест</button>
                            <span id="badge-key-${{pid}}-${{i}}"></span>
                            <button class="action-btn btn-danger" style="padding: 6px 10px;" onclick="deleteKeyRow('${{pid}}', ${{i}})">&#128465;</button>
                        </div>
                        `;
                    }}
                }}

                html += `
                <div class="table-card" id="card-prov-${{pid}}" style="margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 16px; font-weight: 700; color: #38bdf8;">${{p.name}}</span>
                            <span style="background: #0f172a; border: 1px solid #334155; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; color: #94a3b8;">${{pid}}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <label style="font-size: 12px; color: #e2e8f0; display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                <input type="checkbox" id="prov-enabled-${{pid}}" ${{isEnabled ? 'checked' : ''}} style="cursor: pointer;"> Включен
                            </label>
                            <label style="font-size: 12px; color: #e2e8f0; display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                <input type="checkbox" id="prov-proxy-${{pid}}" ${{useProxy ? 'checked' : ''}} style="cursor: pointer;"> Прокси
                            </label>
                            <button class="action-btn btn-danger" style="padding: 5px 10px; font-size: 12px;" onclick="deleteProvider('${{pid}}')">&#128465; Удалить</button>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 14px;">
                        <div>
                            <label style="font-size: 11px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 4px;">Base URL:</label>
                            <input type="text" id="prov-url-${{pid}}" value="${{p.base_url}}" class="admin-input">
                        </div>
                        <div>
                            <label style="font-size: 11px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 4px;">Переменная ключей в .env:</label>
                            <input type="text" id="prov-env-${{pid}}" value="${{p.api_key_env}}" class="admin-input" readonly style="opacity: 0.8;">
                        </div>
                    </div>

                    <div>
                        <label style="font-size: 12px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 8px;">Пул API ключей (Round-Robin):</label>
                        <div id="keys-list-${{pid}}">
                            ${{keysRowsHtml}}
                        </div>
                        <div style="display: flex; gap: 8px; margin-top: 10px;">
                            <input type="text" id="new-key-for-${{pid}}" placeholder="Вставьте новый API ключ..." class="admin-input" style="flex: 1;">
                            <button class="action-btn" style="background: #059669;" onclick="addNewKeyToProviderList('${{pid}}')">&#10133; Добавить ключ</button>
                        </div>
                    </div>
                </div>
                `;
            }}

            container.innerHTML = html;
        }}

        function addNewKeyToProviderList(pid) {{
            const input = document.getElementById('new-key-for-' + pid);
            if (!input || !input.value.trim()) return;
            const newKey = input.value.trim();
            if (!activeKeysConfig || !activeKeysConfig.providers[pid]) return;
            activeKeysConfig.providers[pid].keys.push(newKey);
            input.value = '';
            renderKeysConfig(activeKeysConfig);
        }}

        function deleteKeyRow(pid, idx) {{
            if (!activeKeysConfig || !activeKeysConfig.providers[pid]) return;
            activeKeysConfig.providers[pid].keys.splice(idx, 1);
            renderKeysConfig(activeKeysConfig);
        }}

        async function testKeyLive(pid, idx) {{
            const keyInput = document.getElementById('input-key-' + pid + '-' + idx);
            const badge = document.getElementById('badge-key-' + pid + '-' + idx);
            const urlInput = document.getElementById('prov-url-' + pid);
            const proxyCheck = document.getElementById('prov-proxy-' + pid);

            if (!keyInput || !keyInput.value.trim()) return;
            const key = keyInput.value.trim();
            const baseUrl = urlInput ? urlInput.value.trim() : '';
            const useProxy = proxyCheck ? proxyCheck.checked : false;

            if (badge) {{
                badge.className = 'badge-ping';
                badge.style.background = '#334155';
                badge.style.color = '#f1f5f9';
                badge.innerText = 'Проверка...';
            }}

            const token = getAuthToken();
            try {{
                const res = await fetch('/v1/admin/test-key', {{
                    method: 'POST',
                    headers: {{
                        'Authorization': 'Bearer ' + token,
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ base_url: baseUrl, key: key, use_proxy: useProxy }})
                }});
                const data = await res.json();
                if (badge) {{
                    if (data.ok) {{
                        badge.className = 'badge-ping badge-ping-ok';
                        badge.innerText = '✓ ' + data.status + ' (' + data.latency_ms + 'ms)';
                    }} else {{
                        badge.className = 'badge-ping badge-ping-err';
                        badge.innerText = '✕ ' + (data.error || 'Ошибка ' + data.status_code);
                    }}
                }}
            }} catch (e) {{
                if (badge) {{
                    badge.className = 'badge-ping badge-ping-err';
                    badge.innerText = '✕ Ошибка: ' + e;
                }}
            }}
        }}

        async function saveAllKeysAndSecrets() {{
            const token = getAuthToken();
            if (!token) return;

            const gwKey = (document.getElementById('sys-gateway-key') || {{}}).value || '';
            const proxy = (document.getElementById('sys-outbound-proxy') || {{}}).value || '';
            const dashPwd = (document.getElementById('sys-dash-pwd') || {{}}).value || '';

            const providerKeys = {{}};
            if (activeKeysConfig && activeKeysConfig.providers) {{
                for (const pid of Object.keys(activeKeysConfig.providers)) {{
                    const p = activeKeysConfig.providers[pid];
                    const envVar = p.api_key_env;
                    const keys = [];
                    const container = document.getElementById('keys-list-' + pid);
                    if (container) {{
                        const inputs = container.querySelectorAll('input[type="password"], input[type="text"]');
                        inputs.forEach(inp => {{
                            if (inp.value && inp.value.trim()) keys.push(inp.value.trim());
                        }});
                    }} else {{
                        keys.push(...(p.keys || []));
                    }}
                    providerKeys[envVar] = keys;
                }}
            }}

            const payload = {{
                gateway_api_key: gwKey.trim(),
                outbound_proxy: proxy.trim(),
                provider_keys: providerKeys
            }};
            if (dashPwd.trim()) {{
                payload.dashboard_password = dashPwd.trim();
            }}

            try {{
                const res = await fetch('/v1/admin/save-keys', {{
                    method: 'POST',
                    headers: {{
                        'Authorization': 'Bearer ' + token,
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(payload)
                }});
                const data = await res.json();
                if (res.ok) {{
                    alert('✓ Секреты и ключи успешно сохранены в .env и обновлены в шлюзе!');
                    loadKeysConfig();
                }} else {{
                    alert('Ошибка при сохранении: ' + (data.detail || JSON.stringify(data)));
                }}
            }} catch (e) {{
                alert('Ошибка сети при сохранении: ' + e);
            }}
        }}

        function openNewProviderModal() {{
            const m = document.getElementById('modal-new-provider');
            if (m) m.style.display = 'flex';
        }}

        function closeNewProviderModal() {{
            const m = document.getElementById('modal-new-provider');
            if (m) m.style.display = 'none';
        }}

        function autoUpdateEnvVarName() {{
            const idInput = document.getElementById('new-p-id');
            const envInput = document.getElementById('new-p-env');
            if (idInput && envInput) {{
                const cleaned = idInput.value.replace(/[^a-zA-Z0-9_]/g, '').toUpperCase();
                envInput.value = cleaned ? cleaned + '_API_KEYS' : '';
            }}
        }}

        async function submitNewProvider() {{
            const token = getAuthToken();
            if (!token) return;

            const pid = (document.getElementById('new-p-id') || {{}}).value || '';
            const name = (document.getElementById('new-p-name') || {{}}).value || '';
            const url = (document.getElementById('new-p-url') || {{}}).value || '';
            const envVar = (document.getElementById('new-p-env') || {{}}).value || '';
            const keysRaw = (document.getElementById('new-p-keys') || {{}}).value || '';
            const useProxy = (document.getElementById('new-p-proxy') || {{}}).checked || false;

            if (!pid.trim()) {{ alert('Укажите ID поставщика'); return; }}
            if (!url.trim()) {{ alert('Укажите Base URL поставщика'); return; }}

            const payload = {{
                action: 'save',
                provider_id: pid.trim().toLowerCase(),
                name: name.trim() || pid.trim(),
                base_url: url.trim(),
                api_key_env: envVar.trim() || (pid.trim().toUpperCase() + '_API_KEYS'),
                keys: keysRaw.split(',').map(s => s.trim()).filter(Boolean),
                use_proxy: useProxy,
                enabled: true
            }};

            try {{
                const res = await fetch('/v1/admin/save-provider', {{
                    method: 'POST',
                    headers: {{
                        'Authorization': 'Bearer ' + token,
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(payload)
                }});
                const data = await res.json();
                if (res.ok) {{
                    closeNewProviderModal();
                    alert('✓ Поставщик ' + pid + ' успешно добавлен и активирован!');
                    loadKeysConfig();
                }} else {{
                    alert('Ошибка при добавлении: ' + (data.detail || JSON.stringify(data)));
                }}
            }} catch (e) {{
                alert('Ошибка сети: ' + e);
            }}
        }}

        async function deleteProvider(pid) {{
            if (!confirm('Вы уверены, что хотите удалить поставщика \"' + pid + '\" из конфигурации?')) return;
            const token = getAuthToken();
            if (!token) return;

            try {{
                const res = await fetch('/v1/admin/save-provider', {{
                    method: 'POST',
                    headers: {{
                        'Authorization': 'Bearer ' + token,
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ action: 'delete', provider_id: pid }})
                }});
                const data = await res.json();
                if (res.ok) {{
                    alert('✓ Поставщик ' + pid + ' удален.');
                    loadKeysConfig();
                }} else {{
                    alert('Ошибка: ' + (data.detail || JSON.stringify(data)));
                }}
            }} catch (e) {{
                alert('Ошибка: ' + e);
            }}
        }}

        async function apiAction(endpoint, name) {{
            let key = getAuthToken();
            if (!key) {{
                key = prompt("Enter Gateway Master API Key for authorization:", "{master_key}");
                if (!key) return;
                setAuthToken(key);
            }}
            try {{
                const res = await fetch(endpoint, {{
                    method: 'POST',
                    headers: {{ 'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json' }}
                }});
                const json = await res.json();
                alert(name + " result: " + JSON.stringify(json, null, 2));
                location.reload();
            }} catch (e) {{
                alert("Error: " + e);
            }}
        }}

        // Восстановление активной вкладки после автообновления
        const savedTab = localStorage.getItem('active_tab') || 'gateway';
        switchTab(savedTab);

        // Живое фоновое обновление метрик без полной перезагрузки страницы (AJAX)
        async function refreshDashboardMetrics() {{
            // Если открыта вкладка ключей, не трогаем DOM, чтобы не сбивать фокус ввода
            if (localStorage.getItem('active_tab') === 'keys') return;

            try {{
                const res = await fetch('/health', {{ headers: {{ 'Accept': 'application/json' }} }});
                if (!res.ok) return;
                const d = await res.json();

                // 1. Uptime
                if (d.uptime_sec !== undefined) {{
                    const u = d.uptime_sec;
                    const uStr = Math.floor(u / 3600) + 'h ' + Math.floor((u % 3600) / 60) + 'm ' + Math.floor(u % 60) + 's';
                    const elUptime = document.getElementById('val-uptime');
                    if (elUptime) elUptime.innerText = uStr;
                }}

                // 2. Tab 1 - Gateway & Inference
                const loc = d.local_backend || {{}};
                const elLocStatus = document.getElementById('val-loc-status');
                if (elLocStatus) {{
                    elLocStatus.innerText = loc.online ? 'ONLINE' : 'OFFLINE';
                    elLocStatus.style.color = loc.online ? '#10b981' : '#ef4444';
                }}
                const elSlotStatus = document.getElementById('val-slot-status');
                if (elSlotStatus) {{
                    elSlotStatus.innerText = loc.slot_occupied ? 'BUSY (Generating)' : 'IDLE (Ready)';
                    elSlotStatus.style.color = loc.slot_occupied ? '#f59e0b' : '#10b981';
                }}
                const elSlotQueue = document.getElementById('val-slot-queue');
                if (elSlotQueue) {{
                    elSlotQueue.innerText = 'Queue Waiters: ' + (loc.queue_length || 0);
                }}
                const elLocNpast = document.getElementById('val-loc-npast');
                if (elLocNpast) {{
                    elLocNpast.innerHTML = (loc.n_past || 0).toLocaleString() + ' <span style="font-size: 15px; font-weight: normal; color: #94a3b8;">tokens</span>';
                }}
                const elLocSpeed = document.getElementById('val-loc-speed');
                if (elLocSpeed) {{
                    elLocSpeed.innerText = 'Generation Speed: ' + (loc.tokens_per_sec || 0) + ' t/s';
                }}
                const elSessCount = document.getElementById('val-sess-count');
                if (elSessCount) {{
                    elSessCount.innerText = (d.active_sessions || []).length;
                }}

                // 3. Tab 1 - Sessions Table
                const tbodySess = document.getElementById('tbody-sessions');
                if (tbodySess && d.active_sessions) {{
                    if (d.active_sessions.length === 0) {{
                        tbodySess.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #6b7280;">No active agent sessions</td></tr>';
                    }} else {{
                        let html = '';
                        for (const s of d.active_sessions) {{
                            const badge = s.is_holding_local_slot
                                ? '<span style="color: #10b981; font-weight: bold;">● Active Slot</span>'
                                : '<span style="color: #6b7280;">○ Idle</span>';
                            html += `<tr>
                                <td style="font-family: monospace; color: #a78bfa;">${{s.session_id}}</td>
                                <td>${{s.active_provider}} / <span style="color: #60a5fa;">${{s.active_model}}</span></td>
                                <td>${{badge}}</td>
                                <td>${{s.idle_sec}}s ago</td>
                                <td>${{s.total_requests}}</td>
                            </tr>`;
                        }}
                        tbodySess.innerHTML = html;
                    }}
                }}

                // 4. Tab 1 - Keys Table
                const tbodyKeys = document.getElementById('tbody-keys');
                if (tbodyKeys && d.providers_keys) {{
                    let kHtml = '';
                    for (const [pid, klist] of Object.entries(d.providers_keys)) {{
                        for (const k of klist) {{
                            const color = k.status === 'ACTIVE' ? '#10b981' : (k.status.includes('COOLDOWN') ? '#f59e0b' : '#ef4444');
                            kHtml += `<tr>
                                <td style="font-weight: 600; color: #93c5fd;">${{pid}}</td>
                                <td style="font-family: monospace;">${{k.masked_key}}</td>
                                <td><span style="background: ${{color}}22; color: ${{color}}; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">${{k.status}}</span></td>
                                <td>${{k.request_count}}</td>
                                <td style="color: ${{k.fail_count > 0 ? '#ef4444' : '#9ca3af'}};">${{k.fail_count}}</td>
                            </tr>`;
                        }}
                    }}
                    if (kHtml) tbodyKeys.innerHTML = kHtml;
                }}

                // 5. Tab 2 - System Metrics
                const sys = d.system || {{}};
                const cpu = sys.cpu || {{}};
                const mem = sys.memory || {{}};
                const gpu = sys.gpu || {{}};
                const disk = sys.disk || {{}};

                // CPU
                const cpuPct = cpu.overall_percent || 0;
                const cpuColor = cpuPct < 60 ? '#10b981' : (cpuPct < 85 ? '#f59e0b' : '#ef4444');
                const elCpuPct = document.getElementById('val-cpu-pct');
                if (elCpuPct) {{ elCpuPct.innerText = cpuPct + '%'; elCpuPct.style.color = cpuColor; }}
                const elCpuBar = document.getElementById('bar-cpu-pct');
                if (elCpuBar) {{ elCpuBar.style.width = cpuPct + '%'; elCpuBar.style.background = cpuColor; }}
                const elCpuSub = document.getElementById('sub-cpu-info');
                if (elCpuSub) {{
                    const freqStr = cpu.frequency_mhz ? ` | ${{Math.round(cpu.frequency_mhz)}} MHz` : '';
                    const tempBadge = (cpu.temperature_c !== null && cpu.temperature_c !== undefined)
                        ? `<span class="badge-temp badge-green" style="margin-left: 6px; font-size: 11px;">${{cpu.temperature_c}}°C</span>` : '';
                    elCpuSub.innerHTML = `${{cpu.physical_cores || 1}}C / ${{cpu.logical_cores || 1}}T${{freqStr}} ${{tempBadge}}`;
                }}

                // Cores Grid
                const gridCores = document.getElementById('grid-cores');
                if (gridCores && cpu.per_core_percent) {{
                    let cHtml = '';
                    cpu.per_core_percent.forEach((cp, idx) => {{
                        const cColor = cp < 60 ? '#10b981' : (cp < 85 ? '#f59e0b' : '#ef4444');
                        cHtml += `<div class="core-box">
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 4px;">
                                <span>Core ${{idx}}</span>
                                <span style="font-weight: 700; color: #f8fafc;">${{cp}}%</span>
                            </div>
                            <div class="progress-bg" style="height: 6px; margin-top: 0;">
                                <div class="progress-bar" style="width: ${{cp}}%; background: ${{cColor}};"></div>
                            </div>
                        </div>`;
                    }});
                    if (cHtml) gridCores.innerHTML = cHtml;
                }}

                // RAM
                const ramPct = mem.ram_percent || 0;
                const ramColor = ramPct < 70 ? '#10b981' : (ramPct < 90 ? '#f59e0b' : '#ef4444');
                const elRamVal = document.getElementById('val-ram-used');
                if (elRamVal) elRamVal.innerText = `${{mem.ram_used_gb || 0}} / ${{mem.ram_total_gb || 0}} GB`;
                const elRamBar = document.getElementById('bar-ram-pct');
                if (elRamBar) {{ elRamBar.style.width = ramPct + '%'; elRamBar.style.background = ramColor; }}
                const elRamSub = document.getElementById('sub-ram-info');
                if (elRamSub) {{
                    const swapStr = (mem.swap_total_gb > 0) ? ` | Swap: ${{mem.swap_used_gb}}/${{mem.swap_total_gb}} GB (${{mem.swap_percent}}%)` : '';
                    elRamSub.innerText = `${{ramPct}}% used (Free: ${{mem.ram_available_gb || 0}} GB)${{swapStr}}`;
                }}

                // GPU
                const elGpuVal = document.getElementById('val-gpu-pct');
                const vramPct = gpu.vram_percent || 0;
                const vramColor = vramPct < 70 ? '#10b981' : (vramPct < 90 ? '#f59e0b' : '#ef4444');
                if (elGpuVal) {{
                    if (gpu.available && gpu.gpu_percent > 0) {{
                        elGpuVal.innerText = gpu.gpu_percent + '%';
                        elGpuVal.style.color = '#10b981';
                    }} else if (gpu.available) {{
                        elGpuVal.innerText = 'ACTIVE';
                        elGpuVal.style.color = '#10b981';
                    }} else {{
                        elGpuVal.innerText = 'N/A';
                        elGpuVal.style.color = '#64748b';
                    }}
                }}
                const elVramBar = document.getElementById('bar-vram-pct');
                if (elVramBar) {{ elVramBar.style.width = vramPct + '%'; elVramBar.style.background = vramColor; }}
                const elGpuSub = document.getElementById('sub-gpu-info');
                if (elGpuSub) {{
                    if (gpu.available && gpu.vram_total_gb > 0) {{
                        elGpuSub.innerText = `VRAM: ${{gpu.vram_used_gb}}/${{gpu.vram_total_gb}} GB (${{vramPct}}%)`;
                    }} else {{
                        elGpuSub.innerText = gpu.name || 'No accelerator';
                    }}
                }}

                // Disk
                const diskPct = disk.percent || 0;
                const diskColor = diskPct < 75 ? '#10b981' : (diskPct < 90 ? '#f59e0b' : '#ef4444');
                const elDiskVal = document.getElementById('val-disk-used');
                if (elDiskVal) elDiskVal.innerText = `${{disk.used_gb || 0}} / ${{disk.total_gb || 0}} GB`;
                const elDiskBar = document.getElementById('bar-disk-pct');
                if (elDiskBar) {{ elDiskBar.style.width = diskPct + '%'; elDiskBar.style.background = diskColor; }}
                const elDiskSub = document.getElementById('sub-disk-info');
                if (elDiskSub) elDiskSub.innerText = `${{diskPct}}% used (Free: ${{disk.free_gb || 0}} GB)`;

                // Temperatures
                const tbodyTemps = document.getElementById('tbody-temps');
                if (tbodyTemps && sys.temperatures && sys.temperatures.length > 0) {{
                    let tHtml = '';
                    for (const t of sys.temperatures) {{
                        const st = t.status || 'normal';
                        const badgeCls = st === 'normal' ? 'badge-green' : (st === 'warning' ? 'badge-yellow' : 'badge-red');
                        const hVal = t.high_c ? t.high_c + '°C' : '-';
                        const crVal = t.critical_c ? t.critical_c + '°C' : '-';
                        tHtml += `<tr>
                            <td style="font-weight: 600; color: #93c5fd; font-family: monospace;">${{t.chip || ''}}</td>
                            <td style="color: #e2e8f0;">${{t.label || ''}}</td>
                            <td><span class="badge-temp ${{badgeCls}}">${{t.current_c}}°C</span></td>
                            <td style="color: #94a3b8; font-family: monospace;">${{hVal}}</td>
                            <td style="color: #ef4444; font-family: monospace;">${{crVal}}</td>
                            <td><span class="badge-temp ${{badgeCls}}" style="font-size: 11px;">${{st.toUpperCase()}}</span></td>
                        </tr>`;
                    }}
                    tbodyTemps.innerHTML = tHtml;
                }}

            }} catch (e) {{
                console.warn('Dashboard live refresh error:', e);
            }}
        }}

        // Живое обновление каждые 3 секунды
        setInterval(refreshDashboardMetrics, 3000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    cfg = config_loader.get_config()
    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level.lower(),
        reload=False,
    )
