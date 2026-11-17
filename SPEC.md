# Техническое задание (ТЗ)
## Проект: Высокопроизводительный ИИ-шлюз (AI Gateway) для локально-облачного инференса и автономных агентов

---

## 1. Общие сведения и назначение системы

### 1.1. Назначение
Высокопроизводительный асинхронный HTTP-шлюз на Python 3.12+, полностью совместимый со спецификацией **OpenAI API** (`/v1`). Шлюз объединяет локальный сервер инференса (`llama-server` с моделью `Qwen3.6-35B-A3B` на базе **AMD RX Vega 56 + Intel Core i7-4790K**) и произвольные внешние облачные API в единую отказоустойчивую точку входа.

Шлюз оптимизирован под работу **автономных программных агентов (Auto-Agents)**:
* Агент самостоятельно управляет логикой своих ролей (например, «Архитектор» на рассуждающей модели и «Исполнитель» на кодовой модели), обращаясь к шлюзу по виртуальным алиасам (`auto-architect`, `auto-executor`, `auto-fast`, `auto-vision`, `local-only`).
* Шлюз гарантирует **липкие сессии (Sticky Sessions)**, **сохранение горячего локального KV-кэша**, **мгновенный отвод параллельных задач в облако**, **бесшовный переброс (Failover в RAM)** при ошибках 429/лимитах и **защиту CPU от тяжелых картинок**.

### 1.2. Архитектурные принципы
1. **Единая точка управления (Single Point of Control):** Клиенты взаимодействуют исключительно со шлюзом `http://<IP>:8000/v1`. Все детали работы локального сервера и внешних облаков изолированы.
2. **Без избыточного хранения (Zero-DB / Pure In-Memory):** Шлюз не сохраняет сообщения на диск — клиент передает полную историю сообщений в каждом запросе. Все состояние сессий, привязок моделей и очередей живет в оперативной памяти (RAM) со скоростью наносекунд.
3. **Фиксация модели за сессией (Sticky Session Affinity):** Шлюз закрепляет выбранную модель за `session_id`, исключая хаотичные переключения между шагами задачи.
4. **Бесшовный Failover в RAM:** При ошибке `429 Too Many Requests` или сетевом сбое роутер перехватывает ошибку, берет контекст из оперативной памяти и прозрачно передает запрос в резервную модель.
5. **Защита локального KV-кэша и слот-менеджмент:** Учитывая медленный префилл на i7-4790K (~75 т/с), сброс кэша недопустим. Активный слот удерживается за сессией (Grace Period), а параллельные запросы чужих задач выталкиваются в облако.
6. **Универсальные провайдеры:** Любой облачный или локальный OpenAI-совместимый провайдер настраивается декларативно в `config.yaml`.

---

## 2. Аппаратный контекст и ограничения локального узла

* **Сервер:** Ubuntu 26 LTS, Intel Core i7-4790K (4C/8T Haswell, DDR3-2400), AMD Radeon RX Vega 56 (8 ГБ HBM2 VRAM, Mesa RADV).
* **Локальный бэкенд (`llama-server`, `http://127.0.0.1:8080`):**
  * Флаги запуска: `--slots`, `-np 1` (строго 1 параллельный слот), `-c 262144`, `--cache-type-k q4_0`, `--cache-type-v q4_0`, `-fa on`, `--fit off`, `--n-cpu-moe 40`.
  * Эндпоинты управления слотом 0: `POST /slots/0?action=erase`, `save`, `restore`.
  * Префилл MoE на CPU: ~70–80 токенов/сек (холодный старт 15k контекста занимает >3 минут).
  * Генерация на GPU: ~16–17 токенов/сек (горячий кэш стартует мгновенно).
  * Мультимодальность: проектор `mmproj-BF16.gguf` на CPU (`--no-mmproj-offload`). Требуется обязательный ресайз картинок до 896px на стороне шлюза.

---

## 3. Файловая структура проекта

```text
localAiRouter/
├── .env                  # Секреты: мастер-ключ шлюза и API-ключи провайдеров
├── .env.example          # Шаблон переменных окружения
├── config.yaml           # Конфигурация: провайдеры, алиасы, очереди, параметры сжатия
├── requirements.txt      # Зависимости Python
├── models_cache.json     # Автоматический кэш обнаруженных моделей провайдеров
├── config_loader.py      # Pydantic v2 схемы и валидация config.yaml + .env (горячая перезагрузка)
├── session_manager.py    # Реестр сессий в RAM: Sticky Affinity, удержание слота (Grace Period)
├── local_manager.py      # Управление llama-server: слот 0, очередь, KV-кэш, автоотмена
├── key_manager.py        # Пул ключей провайдеров: ротация, статусы ACTIVE/COOLDOWN/DEAD
├── discovery.py          # Автодискавери моделей провайдеров (/v1/models)
├── image_pipeline.py     # Обработка изображений (Pillow): ресайз, белый фон для PNG, сжатие
├── router.py             # Маршрутизация, failover в RAM, стриминг SSE, нормализация tool_calls
├── main.py               # FastAPI приложение, OpenAI API (/v1), admin-ручки, /health (HTML + JSON)
└── checker.py            # CLI-утилита быстрой сводки и глубокого пинга (--deep)
```

---

## 4. Конфигурация системы (`config.yaml` и `.env`)

### 4.1. Файл секретов (`.env`)
```bash
GATEWAY_API_KEY=gw-secret-key-12345

CEREBRAS_API_KEYS=csk_key1,csk_key2
GROQ_API_KEYS=gsk_key1,gsk_key2
OPENROUTER_API_KEYS=sk-or-v1-key1
SAMBANOVA_API_KEYS=samba_key1

OUTBOUND_PROXY=
```

### 4.2. Конфигурационный файл (`config.yaml`)
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

local_backend:
  url: "http://127.0.0.1:8080"
  model_name: "Qwen3.6-35B-A3B"
  max_context: 262144
  health_timeout_sec: 0.5
  queue:
    max_size: 5
    same_session_timeout_sec: 45 # Ожидание своей сессии в очереди слота
    grace_period_sec: 60         # Защита горячего слота после окончания генерации
    local_only_timeout_sec: 120  # Ожидание в режиме local-only
  cache:
    idle_clear_sec: 1800         # Автоочистка KV-кэша слота при простое (0 = выкл)
    auto_clear_on_context_shift: true

providers:
  cerebras:
    name: "Cerebras Fast Cloud"
    base_url: "https://api.cerebras.ai/v1"
    api_key_env: "CEREBRAS_API_KEYS"
    enabled: true
    use_proxy: false
    timeout_sec: 60
    models:
      - id: "llama3.1-8b"
        tags: ["fast", "tools"]
        tier: "fast"
      - id: "llama-3.3-70b"
        tags: ["coding", "tools", "smart"]
        tier: "executor"

  groq:
    name: "Groq Cloud"
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: "GROQ_API_KEYS"
    enabled: true
    use_proxy: false
    timeout_sec: 60
    models:
      - id: "qwen-2.5-coder-32b"
        tags: ["coding", "tools"]
        tier: "executor"
      - id: "llama-3.3-70b-versatile"
        tags: ["smart", "tools", "coding"]
        tier: "executor"

  openrouter:
    name: "OpenRouter"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEYS"
    enabled: true
    use_proxy: false
    timeout_sec: 90
    extra_headers:
      HTTP-Referer: "https://github.com/localAiRouter"
      X-Title: "Local AI Gateway"
    only_free_models: false

aliases:
  auto-architect:
    description: "Cloud reasoning/planner models for high-level thinking & architecture"
    models:
      - "deepseek-r1"
      - "qwq-32b"
      - "llama-3.3-70b"
    provider_priority: ["openrouter", "groq", "sambanova", "local"]
    prefer_tags: ["reasoning", "smart"]
    default_sampling:
      temperature: 0.6
      top_p: 0.95

  auto-executor:
    description: "Primary coding executor: local Qwen 35B with cloud coding fallback"
    models:
      - "Qwen3.6-35B-A3B"              # 1-й приоритет: локальный Qwen
      - "qwen-2.5-coder-32b"           # 2-й приоритет: облачный Qwen Coder
      - "llama-3.3-70b"                # 3-й приоритет: Llama 70B
    provider_priority: ["local", "groq", "cerebras", "sambanova", "openrouter"]
    require_tools: true
    prefer_tags: ["coding"]
    default_sampling:
      temperature: 0.2
      top_p: 0.95

  local-only:
    description: "Strictly local Qwen3.6 execution with KV-cache reuse"
    models:
      - "Qwen3.6-35B-A3B"
    provider_priority: ["local"]

  auto-fast:
    description: "Ultra-fast response for small tasks"
    models:
      - "llama3.1-8b"
    provider_priority: ["cerebras", "groq", "sambanova", "local"]
    prefer_tags: ["fast"]

  auto-vision:
    description: "Multimodal tasks with image preprocessing"
    provider_priority: ["openrouter", "groq", "local"]
    require_vision: true

image_processing:
  enabled: true                  # Главный выключатель обработки
  local:
    enabled: true
    max_side_px: 896             # Безопасно для 4-ядерного CPU i7-4790K
    output_format: "JPEG"
    quality: 85
    convert_transparent_to_bg: "#FFFFFF"
    grayscale: false
  cloud:
    enabled: true
    max_side_px: 2048
    quality: 90

routing:
  # Режим ранжирования: "user_first" (сначала модели из конфига, затем эвристика) или "user_only" (строго только из конфига)
  model_selection_mode: "user_first"
  provider_priority: ["local", "cerebras", "groq", "sambanova", "openrouter"]
  sticky_sessions: true          # Фиксация выбранной модели за сессией
  session_ttl_sec: 7200          # Очистка неактивных сессий из RAM через 2 часа
  key_cooldown_sec: 60           # Базовый карантин ключа при 429/5xx
  discovery_refresh_hours: 6     # Автообновление списка моделей провайдеров

discovery:
  enabled: true
  refresh_hours: 6
  llm_classifier:
    enabled: true                # Двойная гибридная классификация: правила + фоновый LLM
    provider: "cerebras"         # Быстрый облачный провайдер для фоновой классификации
    model: "llama3.1-8b"         # Легкая модель для анализа названий и метаданных
    batch_size: 25               # Размер пакета моделей за один запрос
    timeout_sec: 20.0
  context_tiers:
    huge: 131072                 # Порог огромного контекста (128k+)
    large: 65536                 # Порог большого контекста (64k+)
    medium: 32768                # Порог среднего контекста (32k+)
    small: 8192                  # Малый контекст (<32k)
```

---

## 5. Функциональные требования

### 5.1. Session Manager (Реестр сессий в RAM)
1. **Идентификатор сессии:** Передается в заголовке `X-Session-ID` или `extra_body.session_id`. При отсутствии вычисляется хэш `SHA256(Client_IP + System_Prompt[:300])`.
2. **Sticky Routing (Фиксация модели):**
   * При первом запросе сессии роутер выбирает приоритетный доступный бэкенд и фиксирует его в словаре сессий в RAM.
   * Все последующие запросы сессии идут строго на этот же бэкенд, исключая смешивание стилей и форматов.
3. **Grace Period (Удержание горячего слота):**
   * После завершения генерации локальный слот 0 удерживается за сессией на время `grace_period_sec` (60 сек).
   * Если в этот период поступает параллельный запрос от другой сессии (чужой задачи), он не сбрасывает кэш, а направляется в облачный фоллбэк.

### 5.2. Отказоустойчивый Failover в RAM
1. При ошибке `429 Too Many Requests`, `502/503/504` или таймауте текущий ключ/провайдер переводится в `COOLDOWN`.
2. Роутер перехватывает сбой, берет массив `messages` из оперативной памяти активного запроса и повторяет запрос к следующему бэкенду по приоритету.
3. Клиент получает успешный ответ 200 OK без срыва задачи.

### 5.3. Управление локальным бэкендом (`local_manager.py`)
1. **Мгновенное прерывание генерации:** Роутер отслеживает `await request.is_disconnected()` и `CancelledError`. При обрыве соединения клиентом роутер отправляет `POST /slots/0?action=erase` на `127.0.0.1:8080`, освобождая GPU за <500 мс.
2. **Автоочистка по таймауту:** Если к локальному слоту не было обращений более `idle_clear_sec`, кэш слота сбрасывается.
3. **Эндпоинты управления шлюза:**
   * `GET /v1/local/status` — состояние слота 0, активный `session_id`, `n_past` токенов кэша, скорость `t/s`, очередь.
   * `POST /v1/local/cache/clear` — сброс KV-кэша слота 0 (`action=erase`).
   * `POST /v1/local/cache/save` и `POST /v1/local/cache/restore` — сохранение/загрузка дампа кэша.
   * `POST /v1/local/abort` — принудительная остановка генерации.
   * `POST /v1/admin/reload-config` — горячая перезагрузка `config.yaml` и `.env`.

### 5.4. Предобработка изображений (`image_pipeline.py`)
1. Поиск `image_url` (data URI base64 или URL) в сообщениях.
2. Для запросов в локальный бэкенд:
   * Замена прозрачности PNG на белый фон (`#FFFFFF`).
   * Пропорциональный ресайз до `max_side_px: 896`.
   * Сжатие в JPEG (quality 85) в блоке `with Image.open(...)` с немедленным освобождением памяти.
3. Для облачных запросов: мягкое ограничение до 2048px.

### 5.5. Нормализация Tool Calls
* Если локальный Qwen вернул вызовы инструментов в формате XML (`<tool_call>...</tool_call>`), роутер парсит их и преобразует в стандартную структуру OpenAI API `tool_calls` JSON, предотвращая ошибки на стороне клиентского агента.

### 5.6. Model-First Маршрутизация, Канонизация и Режимы Выбора
1. **Приоритет моделей (Model-First):**
   * В алиасах список `models: [...]` определяет последовательность предпочтительных моделей.
   * Если одна и та же модель доступна у нескольких провайдеров (например, `llama-3.3-70b` есть на Cerebras, Groq и SambaNova), выбор провайдера определяется списком `provider_priority` (на уровне алиаса или глобально в `routing`).
2. **Каноническая нормализация названий:**
   * Автоматическое сопоставление различных синтаксисов поставщиков (например, `meta-llama/llama-3.3-70b-instruct`, `Meta-Llama-3.3-70B-Instruct`, `llama-3.3-70b-versatile` приводятся к канону `llama-3.3-70b`).
3. **Режимы выбора моделей (`model_selection_mode`):**
   * `user_first` — в первую очередь используются модели, явно заданные пользователем в `config.yaml`, после чего список дополняется обнаруженными облачными моделями, отранжированными по эвристикам (совпадение тегов, вместимость контекста, приоритет провайдера).
   * `user_only` — строгий режим, при котором используются исключительно модели, заданные пользователем в конфиге; динамические модели не подмешиваются.
4. **Неизвестные модели:**
   * При запросе клиентом прямой модели, не объявленной в алиасах, роутер ищет её по каталогам всех активных провайдеров (точное или каноническое совпадение) и направляет запросы согласно `provider_priority`.
5. **Двойная гибридная классификация и метаданные контекста:**
   * Из каталогов провайдеров извлекаются полные метаданные: реальный `context_length`, модальность (vision), поддержка `tools`.
   * Модели классифицируются по категориям контекста (`huge` >=128k, `large` >=64k, `medium` >=32k, `small` <32k), что используется в эвристическом ранжировании.
   * Новые неклассифицированные модели пакетами прогоняются через фоновую легкую LLM (`llm_classifier.provider`) для определения канонического имени, тегов и уровней (с сохранением в `models_cache.json` без повторных обращений). При сбое LLM роутер прозрачно использует надежные правила и регулярные выражения.

---

## 6. Встроенная диагностика и мониторинг (/health и checker.py)

### 6.1. Эндпоинт `GET /health`
1. **Для браузера (`Accept: text/html`):**
   * Автономный Dark-theme веб-интерфейс без внешних CDN (встроенные стили и SVG).
   * Индикатор локального бэкенда: онлайн/офлайн, занятость слота, `n_past` токенов кэша, скорость `t/s`, очередь.
   * Таблица активных сессий агента в RAM: `session_id`, текущая привязанная модель, время последней активности.
   * Таблица пула ключей: маскированные ключи, статусы (`ACTIVE`, `COOLDOWN`, `DEAD`).
   * Кнопки быстрых действий (вызов REST API шлюза):
     * «Очистить KV-кэш» $\to$ `POST /v1/local/cache/clear`
     * «Обновить модели» $\to$ `POST /v1/admin/discovery`
     * «Перезагрузить конфиг» $\to$ `POST /v1/admin/reload-config`
2. **Для скриптов (`Accept: application/json`):**
   * Структурированный JSON со всеми метриками и аптаймом.

### 6.2. CLI-утилита `checker.py`
* `python checker.py` — мгновенная сводка по локальному серверу, активным сессиям и ключам через библиотеку `rich`.
* `python checker.py --deep` — отправка тестового микро-промпта (1 токен) через каждый ключ каждого провайдера и локальный инференс с измерением задержки.

---

## 7. Критерии приёмки

1. **Sticky Session Affinity:** Сессия агента удерживает выбранную модель на всех последовательных шагах диалога.
2. **Бесшовный Failover в RAM:** Ошибка 429 переводит запрос на резервный бэкенд в оперативной памяти без срыва задачи агента.
3. **Защита горячего кэша:** Повторные запросы сессии используют горячий KV-кэш слота 0. Параллельный запрос другой сессии уходит в облако.
4. **Мгновенный Abort:** Разрыв соединения клиентом сбрасывает слот в `llama-server` менее чем за 500 мс.
5. **Предобработка картинок:** 4K картинки сжимаются до 896px JPEG перед подачей на CPU i7-4790K.
6. **Горячая перезагрузка:** Изменения в `config.yaml` вступают в силу по `POST /v1/admin/reload-config` без перезапуска сервера.
