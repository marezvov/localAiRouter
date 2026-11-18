import asyncio
import io
import time
from PIL import Image
from starlette.testclient import TestClient

import config_loader
from session_manager import SessionManager
from key_manager import KeyManager, KeyState
from image_pipeline import (
    has_images,
    process_image_bytes,
    process_messages_images,
    ImageLocalConfig,
    ImageCloudConfig,
)
from discovery import auto_tag_model
from main import app


def test_config_loader():
    cfg = config_loader.get_config()
    secrets = config_loader.get_secrets()

    assert cfg.server.port == 8000
    assert "auto-architect" in cfg.aliases
    assert "auto-executor" in cfg.aliases
    assert "local" in (cfg.aliases["auto-executor"].provider_priority or cfg.aliases["auto-executor"].priority)
    assert "Qwen3.6-35B-A3B" in cfg.aliases["auto-executor"].models
    assert secrets.gateway_api_key is not None


def test_session_manager_sticky_and_grace():
    sm = SessionManager(session_ttl_sec=10, grace_period_sec=2.0)

    async def _test():
        s1 = await sm.get_or_create("task_01")
        assert s1.session_id == "task_01"

        # Привязка
        await sm.bind_backend("task_01", "local", "Qwen3.6-35B-A3B")
        assert s1.active_provider == "local"

        # Проверка Grace Period для чужой сессии
        is_held = await sm.is_local_slot_reserved_for_other("task_02")
        assert is_held is True

        # Но для себя слот не заблокирован
        is_held_self = await sm.is_local_slot_reserved_for_other("task_01")
        assert is_held_self is False

        # Подождем окончания grace_period
        await asyncio.sleep(2.1)
        is_held_expired = await sm.is_local_slot_reserved_for_other("task_02")
        assert is_held_expired is False

    asyncio.run(_test())


def test_key_manager_rotation_and_cooldown():
    km = KeyManager(default_cooldown_sec=1.0)

    async def _test():
        await km.register_provider_keys("test_prov", ["key1", "key2"])

        # Round Robin
        k_a = await km.get_active_key("test_prov")
        k_b = await km.get_active_key("test_prov")
        assert {k_a, k_b} == {"key1", "key2"}

        # Отправляем key1 в COOLDOWN
        await km.mark_cooldown("test_prov", "key1", cooldown_sec=1.0)
        assert await km.get_active_key("test_prov") == "key2"
        assert await km.get_active_key("test_prov") == "key2"

        # Ждем выхода из карантина
        await asyncio.sleep(1.1)
        # Снова доступны оба
        picked = [await km.get_active_key("test_prov") for _ in range(4)]
        assert "key1" in picked

        # Помечаем key2 DEAD
        await km.mark_dead("test_prov", "key2")
        assert await km.get_active_key("test_prov") == "key1"

    asyncio.run(_test())


def test_image_pipeline():
    # Создаем 4K тестовое изображение с альфа-каналом (RGBA)
    img = Image.new("RGBA", (3840, 2160), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # Сжимаем для локального CPU (max 896px, JPEG, белый фон)
    out_bytes, mime = process_image_bytes(
        raw_bytes=raw_bytes,
        max_side_px=896,
        output_format="JPEG",
        quality=85,
        convert_transparent_to_bg="#FFFFFF",
    )

    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(out_bytes)) as result_img:
        assert result_img.mode == "RGB"
        assert max(result_img.size) == 896
        assert len(out_bytes) < len(raw_bytes)  # значительно меньше 4K оригинала


def test_auto_tag_discovery():
    assert "coding" in auto_tag_model("qwen-2.5-coder-32b", "groq")
    assert "vision" in auto_tag_model("llama-3.2-11b-vision-preview", "groq")
    assert "reasoning" in auto_tag_model("deepseek-r1-distill-llama-70b", "groq")
    assert "fast" in auto_tag_model("llama-3.1-8b", "cerebras")


def test_api_endpoints_with_test_client():
    with TestClient(app) as client:
        secrets = config_loader.get_secrets()
        auth_header = {"Authorization": f"Bearer {secrets.gateway_api_key}"}

        # 1. Health JSON
        resp = client.get("/health", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        json_data = resp.json()
        assert json_data["status"] == "healthy"
        assert "local_backend" in json_data
        assert "aliases" in json_data
        assert "system" in json_data
        assert "cpu" in json_data["system"]
        assert "memory" in json_data["system"]
        assert "gpu" in json_data["system"]
        assert "temperatures" in json_data["system"]
        assert "disk" in json_data["system"]

        # 2. Health HTML Dashboard with Tab Navigation
        resp_html = client.get("/health", headers={"Accept": "text/html"})
        assert resp_html.status_code == 200
        assert "<!DOCTYPE html>" in resp_html.text
        assert "AI Gateway Dashboard" in resp_html.text
        assert "tab-gateway" in resp_html.text
        assert "tab-system" in resp_html.text
        assert "switchTab" in resp_html.text
        assert "localStorage.getItem('active_tab')" in resp_html.text

        # 3. Models list without auth -> 401
        resp_no_auth = client.get("/v1/models")
        assert resp_no_auth.status_code == 401

        # 4. Models list with auth -> 200
        resp_models = client.get("/v1/models", headers=auth_header)
        assert resp_models.status_code == 200
        model_ids = [m["id"] for m in resp_models.json()["data"]]
        assert "auto-architect" in model_ids
        assert "auto-executor" in model_ids

        # 5. Local status
        resp_loc = client.get("/v1/local/status", headers=auth_header)
        assert resp_loc.status_code == 200

        # 6. Admin reload config
        resp_reload = client.post("/v1/admin/reload-config", headers=auth_header)
        assert resp_reload.status_code == 200
        assert resp_reload.json()["status"] == "reloaded"


def test_router_candidate_priority():
    from router import extract_session_id, Router
    from discovery import normalize_model_name, ModelDiscovery
    from session_manager import SessionState
    from starlette.requests import Request

    # 1. Проверка извлечения session_id
    scope = {
        "type": "http",
        "headers": [(b"x-session-id", b"task-agent-42")],
        "client": ("127.0.0.1", 12345),
    }
    req = Request(scope)
    sid = extract_session_id(req, {})
    assert sid == "task-agent-42"

    # 2. Проверка fallback на SHA256 если заголовок отсутствует
    scope_no_sid = {
        "type": "http",
        "headers": [],
        "client": ("192.168.1.50", 12345),
    }
    req_no_sid = Request(scope_no_sid)
    sid_fallback = extract_session_id(req_no_sid, {"messages": [{"role": "system", "content": "You are an architect"}]})
    assert sid_fallback.startswith("sess_")

    # 3. Проверка нормализации названий моделей
    assert normalize_model_name("meta-llama/llama-3.3-70b-instruct") == "llama-3.3-70b"
    assert normalize_model_name("Meta-Llama-3.3-70B-Instruct") == "llama-3.3-70b"
    assert normalize_model_name("llama-3.3-70b-versatile") == "llama-3.3-70b"
    assert normalize_model_name("Qwen/Qwen2.5-Coder-32B-Instruct") == "qwen-2.5-coder-32b"
    assert normalize_model_name("THUDM/glm-4-9b-chat") == "glm-4-9b"
    assert normalize_model_name("01-ai/yi-large-preview") == "yi-large"
    assert normalize_model_name("deepseek-ai/DeepSeek-V3") == "deepseek-v3"

    # 4. Проверка Model-First порядка и режимов выбора
    cfg = config_loader.get_config()
    km = KeyManager()
    disc = ModelDiscovery(cfg, km)
    r = Router(cfg, km, None, None, disc)

    dummy_session = SessionState(session_id="test_s")
    candidates_exec = r._build_candidate_list("auto-executor", dummy_session, req_tools=True, req_vision=False)
    assert len(candidates_exec) > 0
    # Первой должна идти первая модель из config.yaml (Qwen3.6-35B-A3B на local)
    assert candidates_exec[0] == ("local", "Qwen3.6-35B-A3B")
    # Второй должна идти qwen-2.5-coder-32b на groq
    assert candidates_exec[1] == ("groq", "qwen-2.5-coder-32b")

    # 5. Проверка кросс-провайдерного поиска неизвестной модели
    # Если клиент запросил "meta-llama/llama-3.3-70b-instruct", роутер должен сопоставить ее
    # с llama-3.3-70b на Cerebras, Groq и SambaNova в соответствии с provider_priority
    matches = disc.find_providers_for_model("meta-llama/llama-3.3-70b-instruct", cfg.routing.provider_priority)
    matched_providers = [m[0] for m in matches]
    assert "cerebras" in matched_providers or "groq" in matched_providers or "sambanova" in matched_providers


if __name__ == "__main__":
    print("Running tests...")
    test_config_loader()
    print("✓ Config loader OK")
    test_session_manager_sticky_and_grace()
    print("✓ Session manager (Sticky & Grace Period) OK")
    test_key_manager_rotation_and_cooldown()
    print("✓ Key manager (Rotation & Cooldown) OK")
    test_image_pipeline()
    print("✓ Image pipeline (4K to 896px CPU) OK")
    test_auto_tag_discovery()
    print("✓ Auto-tag discovery OK")
    test_router_candidate_priority()
    print("✓ Router Model-First, Unknown Models & Candidate Priority OK")
    test_api_endpoints_with_test_client()
    print("✓ API Endpoints & Health Dashboard OK")
    print("\nALL 7 TEST SUITES PASSED SUCCESSFULLY!")
