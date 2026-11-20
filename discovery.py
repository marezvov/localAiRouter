import json
import time
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import httpx

from config_loader import AppConfig, ProviderConfig
from key_manager import KeyManager

logger = logging.getLogger("gateway.discovery")
CACHE_FILE = Path(__file__).resolve().parent / "models_cache.json"


def normalize_model_name(name: str) -> str:
    """
    Приводит любые названия моделей (включая GLM, Qwen, Yi, DeepSeek, Llama, Mistral, Phi)
    к каноническому виду для кросс-провайдерного сопоставления.
    Примеры:
    - 'meta-llama/llama-3.3-70b-instruct' -> 'llama-3.3-70b'
    - 'Meta-Llama-3.3-70B-Instruct' -> 'llama-3.3-70b'
    - 'THUDM/glm-4-9b-chat' -> 'glm-4-9b'
    - '01-ai/yi-large-preview' -> 'yi-large'
    - 'Qwen/Qwen2.5-Coder-32B-Instruct' -> 'qwen-2.5-coder-32b'
    """
    s = name.lower().strip()
    if "/" in s:
        s = s.split("/")[-1]

    # Срезаем расширения и квантования
    s = re.sub(r"\.(gguf|bin|safetensors)$", "", s)
    s = re.sub(r"-(q[2-8]_[k_m_s0-9]+|fp16|bf16|int8|int4|bnb.*)$", "", s)

    # Заменяем подчеркивания на дефисы
    s = s.replace("_", "-")

    # Срезаем даты релиза (например -20240905, -2402, -2501)
    s = re.sub(r"-\d{4,8}$", "", s)

    # Срезаем стандартные суффиксы версий и режимов
    for suffix in ("-instruct", "-it", "-versatile", "-preview", "-chat", "-turbo", "-hf"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]

    # Удаляем префиксы организаций если они остались
    for prefix in ("meta-", "thudm-", "01-ai-", "mistralai-", "google-", "microsoft-", "deepseek-ai-", "deepseek-"):
        if s.startswith(prefix) and not s.startswith("deepseek-r1") and not s.startswith("deepseek-v"):
            s = s[len(prefix):]

    if s.startswith("meta-"):
        s = s[5:]

    # Вставка дефиса между именем семейства и версией (qwen2.5 -> qwen-2.5, llama3 -> llama-3, glm4 -> glm-4)
    s = re.sub(r"^(qwen|llama|glm|internlm|phi)(\d)", r"\1-\2", s)

    # Срезаем суффиксы подверсий типа -v0.1, -v1.5 (кроме deepseek-v2, deepseek-v3)
    if not s.startswith("deepseek-v"):
        s = re.sub(r"-v\d+(\.\d+)*$", "", s)

    return s


def parse_context_length_from_id(model_id: str, default_val: int = 32768) -> int:
    """Извлекает размер контекста из названия модели если есть метка (например 128k, 64k, 256k)."""
    mid = model_id.lower()
    match = re.search(r"(\d+)k", mid)
    if match:
        return int(match.group(1)) * 1024
    if "70b" in mid or "405b" in mid:
        return 131072
    return default_val


def auto_tag_model(model_id: str, provider_id: str) -> List[str]:
    """Автоматическое присвоение эвристических тегов модели."""
    tags = set()
    mid = model_id.lower()

    # Coding
    if any(k in mid for k in ("coder", "code", "codestral", "dev", "starcoder", "deepseek-coder")):
        tags.add("coding")
        tags.add("tools")

    # Vision
    if any(k in mid for k in ("vision", "vl", "pixtral", "omni", "4o", "visual", "llava")):
        tags.add("vision")

    # Reasoning / Smart
    if any(k in mid for k in ("r1", "qwq", "o1", "o3", "reasoner", "thinking", "deepseek-r1")):
        tags.add("reasoning")
        tags.add("smart")

    if any(k in mid for k in ("70b", "405b", "command-r-plus", "large", "deepseek-v3", "glm-4")):
        tags.add("smart")

    # Fast
    if provider_id == "cerebras" or any(k in mid for k in ("8b", "mini", "fast", "flash", "haiku", "3b", "1b", "nano")):
        tags.add("fast")

    # Tools
    if any(k in mid for k in ("llama-3", "llama-3.1", "llama-3.2", "llama-3.3", "qwen-2.5", "mistral", "gpt", "glm-4", "internlm")):
        tags.add("tools")

    return sorted(list(tags))


async def classify_models_with_llm(
    models_to_classify: List[Dict[str, Any]],
    provider_id: str,
    pcfg: ProviderConfig,
    model_name: str,
    key: str,
    proxy: Optional[str] = None,
    timeout_sec: float = 20.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Классифицирует пакет моделей через легковесную LLM (Cerebras / Groq / etc).
    Возвращает словарь: model_id -> {"canonical": str, "tags": List[str], "context_k": int, "tier": str}
    """
    if not models_to_classify:
        return {}

    model_ids = [m["id"] for m in models_to_classify]
    prompt = (
        "You are an AI model classifier. For each model ID in the input list, analyze it and return a JSON array.\n"
        "Each JSON object must have:\n"
        "- \"id\": the exact model ID from input\n"
        "- \"canonical\": canonical short base model name (e.g. \"glm-4-9b\", \"llama-3.3-70b\", \"qwen-2.5-coder-32b\")\n"
        "- \"tags\": array of tags chosen from [\"coding\", \"vision\", \"reasoning\", \"smart\", \"fast\", \"tools\"]\n"
        "- \"context_k\": integer context window size in thousands of tokens (e.g. 32, 64, 128, 256)\n"
        "- \"tier\": chosen from [\"architect\", \"executor\", \"fast\", \"utility\"]\n\n"
        f"Input models:\n{json.dumps(model_ids, ensure_ascii=False)}\n\n"
        "Respond with ONLY valid JSON array. No markdown, no prose."
    )

    url = f"{pcfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **pcfg.extra_headers,
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_sec, proxy=proxy) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return {item["id"]: item for item in parsed if isinstance(item, dict) and "id" in item}
    except Exception as e:
        logger.warning(f"LLM model classification via {provider_id}/{model_name} failed: {e}. Fallback to rule-based.")

    return {}


class ModelDiscovery:
    """
    Умный сбор моделей:
    1. Извлечение полных метаданных (контекст, возможности, модальность).
    2. Двойная гибридная классификация: эвристики + фоновый LLM-классификатор.
    3. Каноническая нормализация названий (GLM, Qwen, Yi, DeepSeek, Llama).
    4. Кэширование в models_cache.json без повторных лишних запросов.
    """

    def __init__(self, config: AppConfig, key_manager: KeyManager, outbound_proxy: Optional[str] = None):
        self.config = config
        self.key_manager = key_manager
        self.outbound_proxy = outbound_proxy
        self._cache: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.error(f"Error loading models_cache.json: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing models_cache.json: {e}")

    def _get_context_tier(self, ctx_len: int) -> str:
        tiers = self.config.discovery.context_tiers
        if ctx_len >= tiers.huge:
            return "huge"
        elif ctx_len >= tiers.large:
            return "large"
        elif ctx_len >= tiers.medium:
            return "medium"
        return "small"

    async def refresh_all(self, force: bool = False) -> Dict[str, Any]:
        """Опрашивает всех включенных провайдеров и запускает двойную классификацию."""
        now = time.time()
        last_updated = self._cache.get("updated_at", 0)
        refresh_interval = self.config.discovery.refresh_hours * 3600

        if not force and (now - last_updated) < refresh_interval and "providers" in self._cache:
            return self._cache

        logger.info("Discovering cloud models from providers...")
        providers_data: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Сбор моделей и базовых метаданных
        for pid, pcfg in self.config.providers.items():
            if not pcfg.enabled:
                continue

            models = await self.fetch_provider_models(pid, pcfg)
            providers_data[pid] = models

        # 2. Двойная классификация через LLM для новых неклассифицированных моделей
        if self.config.discovery.llm_classifier.enabled:
            await self._run_llm_classification_pass(providers_data)

        self._cache = {
            "updated_at": now,
            "providers": providers_data,
        }
        self._save_cache()
        return self._cache

    async def _run_llm_classification_pass(self, providers_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Находит неклассифицированные модели и прогоняет их через LLM."""
        llm_cfg = self.config.discovery.llm_classifier
        target_pid = llm_cfg.provider
        pcfg = self.config.providers.get(target_pid)
        if not pcfg or not pcfg.enabled:
            return

        key = await self.key_manager.get_active_key(target_pid)
        if not key:
            return

        # Находим модели без llm_classified
        unclassified: List[Dict[str, Any]] = []
        for pid, mlist in providers_data.items():
            for m in mlist:
                if not m.get("llm_classified"):
                    unclassified.append(m)

        if not unclassified:
            return

        logger.info(f"Running LLM classification for {len(unclassified)} new models via {target_pid}/{llm_cfg.model}...")

        batch_size = llm_cfg.batch_size
        for i in range(0, len(unclassified), batch_size):
            batch = unclassified[i : i + batch_size]
            results = await classify_models_with_llm(
                models_to_classify=batch,
                provider_id=target_pid,
                pcfg=pcfg,
                model_name=llm_cfg.model,
                key=key,
                proxy=self.outbound_proxy if pcfg.use_proxy else None,
                timeout_sec=llm_cfg.timeout_sec,
            )

            for m in batch:
                mid = m["id"]
                if mid in results:
                    r = results[mid]
                    llm_tags = r.get("tags", [])
                    if isinstance(llm_tags, list):
                        m["tags"] = sorted(list(set(m.get("tags", []) + llm_tags)))
                    if r.get("canonical"):
                        m["canonical"] = r["canonical"]
                    if r.get("tier"):
                        m["tier"] = r["tier"]
                    if r.get("context_k") and isinstance(r["context_k"], (int, float)):
                        m["context_length"] = int(r["context_k"] * 1024)
                        m["context_tier"] = self._get_context_tier(m["context_length"])
                    m["llm_classified"] = True

    async def fetch_provider_models(
        self,
        pid: str,
        pcfg: ProviderConfig,
        explicit_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Запрашивает список моделей у провайдера с извлечением метаданных."""
        key = explicit_key or await self.key_manager.get_active_key(pid)
        static_models = {m.id: m for m in pcfg.models}
        cached_models = {
            m["id"]: m
            for m in self._cache.get("providers", {}).get(pid, [])
            if isinstance(m, dict) and "id" in m
        }
        result_models: Dict[str, Dict[str, Any]] = {}

        # 1. Статические модели из config.yaml
        for sid, sm in static_models.items():
            cached = cached_models.get(sid, {})
            ctx_len = cached.get("context_length") or parse_context_length_from_id(sid, 32768)
            result_models[sid] = {
                "id": sid,
                "canonical": cached.get("canonical") or normalize_model_name(sid),
                "tags": sm.tags if sm.tags else (cached.get("tags") or auto_tag_model(sid, pid)),
                "tier": sm.tier or cached.get("tier"),
                "context_length": ctx_len,
                "context_tier": self._get_context_tier(ctx_len),
                "llm_classified": cached.get("llm_classified", False),
                "source": "static_config",
            }

        if not key:
            if "providers" not in self._cache:
                self._cache["providers"] = {}
            self._cache["providers"][pid] = list(result_models.values())
            return list(result_models.values())

        # 2. Динамический опрос /v1/models
        url = f"{pcfg.base_url.rstrip('/')}/models"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            **pcfg.extra_headers,
        }
        proxies = self.outbound_proxy if pcfg.use_proxy else None

        try:
            async with httpx.AsyncClient(timeout=10.0, proxy=proxies) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    model_list = data.get("data", []) if isinstance(data, dict) else []
                    for m in model_list:
                        mid = m.get("id")
                        if not mid:
                            continue

                        if pcfg.only_free_models and not mid.endswith(":free"):
                            continue

                        cached = cached_models.get(mid, {})
                        # Извлечение метаданных о контексте
                        ctx_len = m.get("context_length") or cached.get("context_length") or parse_context_length_from_id(mid, 32768)
                        ctx_tier = self._get_context_tier(ctx_len)

                        # Извлечение модальностей
                        tags = set(cached.get("tags") or auto_tag_model(mid, pid))
                        arch = m.get("architecture", {})
                        if isinstance(arch, dict) and "modality" in arch:
                            if "image" in str(arch["modality"]).lower():
                                tags.add("vision")
                        if "supported_parameters" in m and isinstance(m["supported_parameters"], list):
                            if "tools" in m["supported_parameters"]:
                                tags.add("tools")

                        canonical = cached.get("canonical") or normalize_model_name(mid)

                        if mid in result_models:
                            result_models[mid]["tags"] = sorted(list(set(result_models[mid]["tags"] + list(tags))))
                            result_models[mid]["context_length"] = ctx_len
                            result_models[mid]["context_tier"] = ctx_tier
                        else:
                            result_models[mid] = {
                                "id": mid,
                                "canonical": canonical,
                                "tags": sorted(list(tags)),
                                "tier": cached.get("tier"),
                                "context_length": ctx_len,
                                "context_tier": ctx_tier,
                                "llm_classified": cached.get("llm_classified", False),
                                "source": "dynamic_discovery",
                            }
                elif resp.status_code in (401, 403):
                    await self.key_manager.mark_dead(pid, key, f"Auth error {resp.status_code}")
                elif resp.status_code == 429:
                    await self.key_manager.mark_cooldown(pid, key, 60.0, "Rate limited on discovery")
        except Exception as e:
            logger.warning(f"Could not discover models from {pid}: {e}")

        models_list = list(result_models.values())
        if "providers" not in self._cache:
            self._cache["providers"] = {}
        self._cache["providers"][pid] = models_list
        self._cache["updated_at"] = time.time()
        self._save_cache()

        return models_list

    def find_providers_for_model(
        self,
        target_model: str,
        preferred_providers: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Ищет провайдеров, у которых доступна модель (по точному или каноническому имени).
        Сортирует по preferred_providers / provider_priority.
        """
        priority = preferred_providers or self.config.routing.provider_priority
        norm_target = normalize_model_name(target_model)
        matches: List[Tuple[int, str, str]] = []

        # 1. Локальный бэкенд
        local_name = self.config.local_backend.model_name
        if target_model.lower() == local_name.lower() or norm_target == normalize_model_name(local_name):
            rank = priority.index("local") if "local" in priority else 999
            matches.append((rank, "local", local_name))

        # 2. Облачные провайдеры
        for pid, pcfg in self.config.providers.items():
            if not pcfg.enabled:
                continue

            rank = priority.index(pid) if pid in priority else 999
            provider_models = self._cache.get("providers", {}).get(pid, [])
            if not provider_models:
                provider_models = [{"id": m.id, "tags": m.tags, "canonical": normalize_model_name(m.id)} for m in pcfg.models]

            for m in provider_models:
                mid = m["id"]
                m_canonical = m.get("canonical") or normalize_model_name(mid)

                # Точное совпадение ID
                if target_model.lower() == mid.lower():
                    matches.append((rank - 100, pid, mid))
                    break
                # Каноническое совпадение (например "glm-4-9b", "llama-3.3-70b")
                elif norm_target == m_canonical or norm_target == normalize_model_name(mid):
                    matches.append((rank, pid, mid))
                    break

        matches.sort(key=lambda x: x[0])
        return [(m[1], m[2]) for m in matches]

    def rank_models_heuristically(
        self,
        provider_id: str,
        require_tools: bool = False,
        require_vision: bool = False,
        prefer_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Ранжирует модели провайдера по эвристикам:
        1. Соответствие prefer_tags (coding, reasoning, smart, fast)
        2. Вместимость контекстного окна (context_length / context_tier)
        3. Наличие обязательных возможностей (tools, vision)
        """
        prefer_tags = prefer_tags or []
        provider_models = self._cache.get("providers", {}).get(provider_id, [])

        if not provider_models:
            pcfg = self.config.providers.get(provider_id)
            if pcfg:
                provider_models = [
                    {"id": m.id, "tags": m.tags, "context_length": parse_context_length_from_id(m.id, 32768)}
                    for m in pcfg.models
                ]

        scored: List[Tuple[float, str]] = []
        for m in provider_models:
            mid = m["id"]
            tags = m.get("tags", [])

            if require_tools and "tools" not in tags and "coding" not in tags:
                continue
            if require_vision and "vision" not in tags:
                continue

            score = 0.0
            # Очки за соответствие тегам
            score += sum(10.0 for pt in prefer_tags if pt in tags)

            # Очки за вместимость контекста (большие контексты выше в ранжировании)
            ctx_len = m.get("context_length", 32768)
            if ctx_len >= 131072:
                score += 15.0
            elif ctx_len >= 65536:
                score += 10.0
            elif ctx_len >= 32768:
                score += 5.0

            scored.append((score, mid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored]

    def get_candidate_models(
        self,
        provider_id: str,
        require_tools: bool = False,
        require_vision: bool = False,
        prefer_tags: Optional[List[str]] = None,
    ) -> List[str]:
        return self.rank_models_heuristically(
            provider_id=provider_id,
            require_tools=require_tools,
            require_vision=require_vision,
            prefer_tags=prefer_tags,
        )

    def get_all_models_list(self) -> List[Dict[str, Any]]:
        """Возвращает общий список всех моделей для OpenAI эндпоинта /v1/models."""
        models: List[Dict[str, Any]] = []

        # 1. Алиасы
        for alias_id, alias_cfg in self.config.aliases.items():
            models.append({
                "id": alias_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gateway-alias",
                "permission": [],
                "root": alias_id,
                "parent": None,
                "description": alias_cfg.description,
            })

        # 2. Локальная модель
        models.append({
            "id": self.config.local_backend.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-llama-server",
            "permission": [],
            "root": self.config.local_backend.model_name,
            "parent": None,
        })

        # 3. Облачные модели с их метаданными
        for pid, plist in self._cache.get("providers", {}).items():
            for m in plist:
                models.append({
                    "id": m["id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": pid,
                    "permission": [],
                    "root": m["id"],
                    "parent": None,
                    "canonical": m.get("canonical"),
                    "tags": m.get("tags", []),
                    "context_length": m.get("context_length"),
                    "context_tier": m.get("context_tier"),
                })

        return models
