# Local-Cloud AI Gateway (Локально-облачный ИИ-шлюз)

Высокопроизводительный асинхронный HTTP-шлюз на **Python 3.12+ (FastAPI)**, полностью совместимый со спецификацией **OpenAI API (`/v1`)**. 

Шлюз объединяет локальный сервер инференса (любой OpenAI-совместимый движок, такой как `llama-server`, `vLLM`, `Ollama` на локальных GPU/NPU/CPU) и произвольные внешние облачные API (Cerebras, Groq, SambaNova, OpenRouter, DeepSeek и др.) в единую отказоустойчивую точку входа.

Система полностью **аппаратно-независима (Hardware-Agnostic)**:
* Автоматически определяет установленные процессоры, физические ядра, логические потоки и частоты через ядро ОС.
* Динамически обнаруживает графические ускорители (NVIDIA CUDA через `nvidia-smi`, AMD Radeon через Linux `sysfs`/`hwmon`, Intel Arc, Apple Silicon, Windows Display Adapters).
* В реальном времени считывает загрузку VRAM, активность ядра GPU, энергопотребление и все доступные аппаратные термодатчики (Edge, Junction/Hotspot, VRAM Memory, Package CPU, NVMe).

Система специально оптимизирована для непрерывной работы **двухуровневых автономных программных агентов (Auto-Agents)**:
* **Tier 1: «Архитектор / Планировщик» (Облако):** Мощные рассуждающие модели (DeepSeek-R1, QwQ, Claude, o1/o3-mini) обдумывают архитектуру, анализируют требования, планируют задачи и делают ревью кода.
* **Tier 2: «Исполнитель / Кодер» (Локально):** Локальная модель `Qwen3.6-35B-A3B` (262k контекста) последовательно исполняет код, правит файлы, выполняет вызовы инструментов (`tools`) и держит рабочее дерево проекта в горячем KV-кэше с нулевой стоимостью за токен и без риска утечки кода.

---

## Оглавление

1. [Ключевые особенности и архитектура](#ключевые-особенности-и-архитектура)
2. [Аппаратный контекст и защита KV-кэша](#аппаратный-контекст-и-защита-kv-кэша)
3. [Развёртывание и установка](#развёртывание-и-установка)
   * [Развёртывание на Ubuntu (Linux)](#развёртывание-на-ubuntu-linux)
   * [Автозапуск через systemd](#настройка-автозапуска-через-systemd-на-ubuntu)
   * [Развёртывание на Windows](#развёртывание-на-windows)
4. [Конфигурация системы (`config.yaml` и `.env`)](#конфигурация-системы)
5. [Model-First маршрутизация и канонизация](#model-first-маршрутизация-и-канонизация-моделей)
6. [Двойная классификация (Эвристика + LLM)](#двойная-гибридная-классификация-эвристика--llm)
7. [Предобработка изображений (Image Pipeline)](#предобработка-изображений-image-pipeline)
8. [Диагностика, Web Dashboard и CLI](#диагностика-мониторинг-и-cli)
9. [Подключение клиентов (Cline, Roo Code, Continue, SDK)](#подключение-клиентов)
10. [Справочник API](#справочник-api)

---

## Ключевые особенности и архитектура

* **Единая точка входа (Single Point of Control):** Все клиенты и агенты обращаются по единому адресу `http://<IP>:8000/v1` с авторизацией по мастер-ключу `GATEWAY_API_KEY`.
* **Zero-DB / Pure In-Memory:** Отказ от лишних тяжелых СУБД (PostgreSQL, Redis, SQLite). Состояние сессий, очередей и привязок живет в оперативной памяти (RAM) со скоростью отклика в наносекунды. Клиенты передают полную историю сообщений в каждом запросе, исключая проблему рассинхронизации версий.
* **Липкие сессии (Sticky Session Affinity):** Модель намертво фиксируется за `session_id`, исключая хаотичные переключения между шагами задачи агента.
* **Защита горячего слота (Grace Period):** После окончания генерации локальный слот 0 удерживается за активной сессией на 60 сек (`grace_period_sec`). Параллельные запросы чужих задач не сбивают горячий кэш, а прозрачно выталкиваются в облако.
* **Бесшовный Failover в RAM:** Ошибки `429 Too Many Requests`, `502/503/504` или таймауты перехватываются роутером — массив сообщений из оперативной памяти прозрачно пересылается резервной модели без срыва задачи агента.
* **Мгновенное прерывание вычислений (Abort <500 мс):** При нажатии «Stop» в IDE роутер отслеживает разрыв клиентского сокета и мгновенно отправляет `action=erase` на локальный бэкенд, моментально освобождая GPU и слот.
* **Каноническая нормализация моделей:** Автоматическое сопоставление различных синтаксисов поставщиков (`meta-llama/llama-3.3-70b-instruct`, `Meta-Llama-3.3-70B-Instruct`, `llama-3.3-70b-versatile` приводятся к канону `llama-3.3-70b`).
* **Гибридный LLM-классификатор:** Новые неизвестные модели пакетами классифицируются через легкую нейросеть с кэшированием в `models_cache.json`.
* **Нормализация Tool Calls:** Автоматическое преобразование XML-вызовов Qwen (`<tool_call>`) в стандартный OpenAI-формат `tool_calls` JSON.
* **Встроенный автономный дашборд `/health`:** Работает без подключения к внешним CDN (чистый Dark HTML + SVG) и даёт возможность управления кэшем, перезагрузки конфига и пинга ключей прямо из браузера.

---

## Аппаратный контекст и защита KV-кэша

Система спроектирована с глубоким учетом ограничений связки **Intel Core i7-4790K (4C/8T Haswell, DDR3-2400) + AMD Radeon RX Vega 56 (8 ГБ HBM2 VRAM, Vulkan)**:

* **Скорость генерации (Decoding):** ~16.2–17.3 токенов/сек (быстро и комфортно).
* **Скорость чтения промпта (Prefill):** всего **~70–80 токенов/сек** (тяжелые слои MoE выгружены в ОЗУ через `--n-cpu-moe 40`).

### Почему сброс кэша недопустим:
* **Холодный старт:** при контексте диалога в **15 000 токенов** процессору требуется $15000 / 75 \approx 200$ секунд (>3 минут ожидания первого токена!).
* **Горячий кэш:** если префикс совпал и добавился только 1 новый шаг агента на 200 токенов, префилл занимает всего $200 / 75 \approx 2.6$ секунды, после чего генерация токенов стартует мгновенно!

Именно поэтому роутер реализует жесткую защиту горячего слота (Grace Period, FIFO-очереди своей сессии и мгновенный отвод параллельных задач в облако).

---

## Развёртывание и установка

Репозиторий проекта: **[https://github.com/marezvov/localAiRouter](https://github.com/marezvov/localAiRouter)**

### Развёртывание на Ubuntu (Linux)

#### 1. Системные требования
* Ubuntu 22.04 / 24.04 / 26.04 LTS
* Python 3.11 или 3.12+
* Git, `python3-venv`, `python3-pip`

#### 2. Клонирование и установка зависимостей
```bash
# Клонирование репозитория
cd ~/projects
git clone https://github.com/marezvov/localAiRouter.git
cd localAiRouter

# Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Настройка конфигурации и секретов
```bash
# Создание файла переменных окружения из шаблона
cp .env.example .env

# Редактирование секретов
nano .env
```
Задайте мастер-ключ шлюза и API-ключи ваших облачных провайдеров:
```ini
GATEWAY_API_KEY=gw-secret-key-12345

CEREBRAS_API_KEYS=csk_your_key_1,csk_your_key_2
GROQ_API_KEYS=gsk_your_key_1
OPENROUTER_API_KEYS=sk-or-v1-your-key
SAMBANOVA_API_KEYS=samba_your_key

OUTBOUND_PROXY=
```

#### 4. Запуск и проверка
```bash
# Тестовый запуск
python main.py
```
В соседнем терминале запустите CLI-диагностику:
```bash
python checker.py
```

---

### Настройка автозапуска через `systemd` на Ubuntu

Чтобы шлюз автоматически запускался при старте сервера и перезапускался при сбоях:

#### 1. Создание unit-файла сервиса
```bash
sudo nano /etc/systemd/system/local-ai-router.service
```

Вставьте следующее содержимое (замените `maxim` и пути на ваши):
```ini
[Unit]
Description=Local-Cloud AI Gateway (FastAPI)
After=network-online.target llama-server.service
Wants=network-online.target

[Service]
Type=simple
User=maxim
Group=maxim
WorkingDirectory=/home/maxim/projects/localAiRouter
ExecStart=/home/maxim/projects/localAiRouter/.venv/bin/python main.py
Restart=always
RestartSec=3
EnvironmentFile=/home/maxim/projects/localAiRouter/.env

[Install]
WantedBy=multi-user.target
```

#### 2. Активация и старт сервиса
```bash
sudo systemctl daemon-reload
sudo systemctl enable local-ai-router.service
sudo systemctl start local-ai-router.service

# Проверка статуса
sudo systemctl status local-ai-router.service

# Просмотр логов в реальном времени
journalctl -u local-ai-router.service -f
```

---

### Развёртывание на Windows

#### 1. Установка через PowerShell:
```powershell
cd C:\projects
git clone https://github.com/marezvov/localAiRouter.git
cd localAiRouter

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

#### 2. Запуск:
```powershell
python main.py
```

---

## Конфигурация системы

Все поведение роутера декларативно описывается в `config.yaml`. Изменения применяются **на лету без перезапуска сервера** через эндпоинт `POST /v1/admin/reload-config` (или кнопку в веб-дашборде).

### Полная структура `config.yaml` с комментариями:

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
    same_session_timeout_sec: 45 # Сколько ждать освобождения слота своей задаче
    grace_period_sec: 60         # Защита горячего слота за сессией после завершения генерации
    local_only_timeout_sec: 120  # Сколько ждать в режиме local-only
  cache:
    idle_clear_sec: 1800         # Автоочистка KV-кэша при простое 30 мин (0 = выкл)
    auto_clear_on_context_shift: true

# Универсальные OpenAI-совместимые облачные провайдеры
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

  sambanova:
    name: "SambaNova Cloud"
    base_url: "https://api.sambanova.ai/v1"
    api_key_env: "SAMBANOVA_API_KEYS"
    enabled: true
    use_proxy: false
    timeout_sec: 60
    models:
      - id: "Meta-Llama-3.3-70B-Instruct"
        tags: ["coding", "tools", "smart"]
        tier: "executor"

  openrouter:
    name: "OpenRouter"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEYS"
    enabled: true
    use_proxy: false
    timeout_sec: 90
    extra_headers:
      HTTP-Referer: "https://github.com/marezvov/localAiRouter"
      X-Title: "Local AI Gateway"
    only_free_models: false

# Виртуальные алиасы (Model-First архитектура)
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
      - "Qwen3.6-35B-A3B"              # 1-й выбор: локальный Qwen
      - "qwen-2.5-coder-32b"           # 2-й выбор: облачный Qwen Coder
      - "llama-3.3-70b"                # 3-й выбор: Llama 70B
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

# Обработка изображений (защита CPU)
image_processing:
  enabled: true                  # Главный переключатель (false = полное отключение)
  local:
    enabled: true                # Ресайз до 896px JPEG для CPU i7-4790K
    max_side_px: 896
    output_format: "JPEG"
    quality: 85
    convert_transparent_to_bg: "#FFFFFF"
    grayscale: false
  cloud:
    enabled: true                # Мягкий лимит до 2048px для облака
    max_side_px: 2048
    quality: 90

# Маршрутизация и режимы выбора
routing:
  # "user_first" - сначала модели из config.yaml, затем эвристика Discovery
  # "user_only"  - строго ТОЛЬКО то, что перечислено в config.yaml
  model_selection_mode: "user_first"
  provider_priority: ["local", "cerebras", "groq", "sambanova", "openrouter"]
  sticky_sessions: true          # Фиксация выбранной модели за сессией
  session_ttl_sec: 7200          # Очистка неактивных сессий из RAM через 2 часа
  key_cooldown_sec: 60           # Базовый карантин ключа при 429/5xx
  discovery_refresh_hours: 6

# Автоматическое обнаружение и двойная классификация
discovery:
  enabled: true
  refresh_hours: 6
  llm_classifier:
    enabled: true                # Фоновая разметка новых моделей через нейросеть
    provider: "cerebras"         # Быстрый провайдер классификатора
    model: "llama3.1-8b"         # Легкая модель
    batch_size: 25
    timeout_sec: 20.0
  context_tiers:
    huge: 131072                 # 128k+
    large: 65536                 # 64k+
    medium: 32768                # 32k+
    small: 8192                  # <32k
```

---

## Model-First маршрутизация и канонизация моделей

В отличие от традиционных шлюзов, ориентированных на провайдеров, Local AI Router использует архитектуру **Model-First**:

1. **Определение модели:** В алиасе список `models: [...]` задаёт идеальную цепочку предпочтительных моделей.
2. **Выбор поставщика для одинаковых моделей:** Если модель (например, `llama-3.3-70b`) доступна сразу у нескольких поставщиков, роутер выберет поставщика согласно списку `provider_priority` (на уровне алиаса или глобально).
3. **Каноническая нормализация названий:** Роутер автоматически очищает технические префиксы и суффиксы провайдеров:
   * `meta-llama/llama-3.3-70b-instruct` $\to$ `llama-3.3-70b`
   * `llama-3.3-70b-versatile` $\to$ `llama-3.3-70b`
   * `Meta-Llama-3.3-70B-Instruct` $\to$ `llama-3.3-70b`
   * `THUDM/glm-4-9b-chat` $\to$ `glm-4-9b`
   * `01-ai/yi-large-preview` $\to$ `yi-large`
   * `Qwen/Qwen2.5-Coder-32B-Instruct` $\to$ `qwen-2.5-coder-32b`
   * `deepseek-ai/DeepSeek-V3` $\to$ `deepseek-v3`
4. **Неизвестные модели:** Если клиент запрашивает модель напрямую (например, `deepseek-chat` или `gpt-4o`), роутер находит её по каталогам всех включенных провайдеров и направляет запросы по приоритету.

---

## Двойная гибридная классификация (Эвристика + LLM)

При первом опросе `/v1/models` роутер выполняет двухэтапную классификацию:
1. **Эвристический анализ:** моментальное назначение тегов (`coding`, `vision`, `tools`, `reasoning`, `fast`) по шаблонам и метаданным API.
2. **Фоновый LLM-классификатор:** новые неклассифицированные модели пакетами отправляются в быструю модель (например, `llama3.1-8b` на Cerebras) для определения канонического имени, уровня задачи (`tier`) и оценки контекста.
3. **Кэширование в `models_cache.json`:** результаты сохраняются на диск, поэтому при последующих стартах шлюза запросы к классификатору не повторяются.

---

## Предобработка изображений (Image Pipeline)

Проектор `mmproj-BF16.gguf` для `Qwen3.6-35B-A3B` рассчитывается на 4-ядерном процессоре Intel Core i7-4790K. Подача несжатых 4K-скриншотов вызывает критическую задержку кодирования.

Пайплайн на базе библиотеки `Pillow`:
* Автоматически находит картинки в формате `image_url` (data URI base64 или по HTTP-ссылкам).
* Для локальной модели: заменяет прозрачность PNG на сплошной белый фон, пропорционально уменьшает изображение до `896px` по длинной стороне и сжимает в прогрессивный JPEG (качество 85).
* Для облачных моделей: мягкий лимит до `2048px`.
* **Управление:** обработку можно полностью отключить (`image_processing.enabled: false`) или настроить раздельно для локального узла и облака.

---

## Диагностика, мониторинг и CLI

### 1. Вкладочный веб-дашборд (`http://localhost:8000/health`)
Адаптивный Dark HTML-интерфейс без сторонних CDN (чистые инлайн стили и SVG) с сохранением выбранной вкладки в `localStorage` при 10-секундном автообновлении:

#### Вкладка 1: «⚡ Шлюз и Модели»
* **Индикатор локального инференса:** онлайн/офлайн, статус слота 0, размер горячего KV-кэша (`n_past`), скорость декодинга (t/s), длина очереди ожидания.
* **Таблица активных сессий агентов:** `session_id`, активная модель, статус удержания локального слота (`● Active Slot` / `○ Idle`), время последней активности.
* **Таблица API-ключей:** маскированные ключи, статусы (`ACTIVE`, `COOLDOWN`, `DEAD`), счетчики запросов и ошибок.
* **Сетка виртуальных алиасов:** отображение цепочек приоритетов моделей и поставщиков.
* **Кнопки управления:** «Clear KV-Cache», «Ping All Keys», «Refresh Models», «Reload Config».

#### Вкладка 2: «🖥️ Система и Температуры»
* **Общие метрики:**
  * **CPU:** суммарная загрузка %, текущая частота MHz, физические ядра и логические потоки, температура Package.
  * **RAM & Swap:** использовано / всего ГБ, процентный прогресс-бар, объем свободной памяти, размер файла подкачки.
  * **GPU & VRAM:** динамически обнаруженная модель видеокарты (NVIDIA, AMD, Intel, Apple Silicon), загрузка ядра %, занято / всего VRAM, энергопотребление в ваттах.
  * **Диск:** занято / всего ГБ на системном накопителе с прогресс-баром.
* **Поядерная загрузка процессора:** карточки Core 0 ... Core N-1 с визуальными мини-барами нагрузки каждого ядра.
* **Детали аппаратных ускорителей:** динамическое считывание метрик из драйверов (активность GPU, VRAM, энергопотребление, температуры чипа, Hotspot/Junction и видеопамяти без захардкоженных моделей).
* **Сводная таблица термодатчиков:** все обнаруженные датчики (`coretemp`, `amdgpu`, `nvme`, `acpitz`, `thermal_zone`) с цветовыми бейджами температур (зеленый <60°C, желтый 60-80°C, красный >80°C) и статусами NORMAL/WARNING/CRITICAL.

### 2. Консольная утилита `checker.py`
Мгновенная сводка в терминале через библиотеку `rich` (статус локального бэкенда, ключи провайдеров, алиасы, сводка системных ресурсов и таблица температур):
```bash
python checker.py
```
Глубокий пинг всех ключей реальными микро-запросами с измерением латентности:
```bash
python checker.py --deep
```

### 3. Набор автоматических тестов:
```bash
python test_gateway.py
```
Запуск 7 автоматизированных тестовых наборов (конфиг, сессии, очереди, ключи, сжатие картинок 4K, канонизация моделей, API эндпоинты, JSON-метрики системы и HTML-вкладки дашборда).

---

## Подключение клиентов

Local AI Router полностью совместим с OpenAI API (`v1`).

### 1. IDE-агенты (Cline / Roo Code / Continue)
* **API Provider:** `OpenAI Compatible`
* **Base URL:** `http://<IP-сервера>:8000/v1`
* **API Key:** `gw-secret-key-12345` (ваш `GATEWAY_API_KEY`)
* **Model ID:**
  * `auto-executor` — основной кодер (локальный Qwen + облачный фоллбэк при занятости).
  * `auto-architect` — облачный планировщик и архитектор (DeepSeek-R1, QwQ).
  * `local-only` — строго локальная генерация.
  * `auto-fast` — быстрые утилиты.
  * `auto-vision` — задачи со скриншотами и UI.

### 2. Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="gw-secret-key-12345",
    default_headers={"X-Session-ID": "agent-task-001"}  # Для удержания горячего кэша
)

response = client.chat.completions.create(
    model="auto-executor",
    messages=[
        {"role": "system", "content": "You are an expert coding assistant."},
        {"role": "user", "content": "Write a FastAPI middleware for request timing."}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

---

## Справочник API

| Метод | Эндпоинт | Описание |
| :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | Стандартный OpenAI чат (обычный и SSE стриминг) |
| `GET` | `/v1/models` | Список всех моделей (алиасы, локальная модель, облачные) |
| `GET` | `/v1/models/{model_id}` | Детали конкретной модели |
| `GET` | `/health` | Диагностика: HTML (в браузере) или JSON (по API/curl) |
| `GET` | `/health?check_keys=true` | Диагностика с реальным пингом всех ключей |
| `GET` | `/v1/local/status` | Детальный статус слота 0, KV-кэша и скорости локального инференса |
| `POST` | `/v1/local/cache/clear` | Принудительная очистка KV-кэша слота 0 в `llama-server` |
| `POST` | `/v1/local/cache/save` | Сохранение дампа KV-кэша слота на диск |
| `POST` | `/v1/local/cache/restore` | Загрузка дампа KV-кэша слота с диска |
| `POST` | `/v1/local/abort` | Экстренная остановка генерации на GPU |
| `POST` | `/v1/admin/reload-config` | Горячая перезагрузка `config.yaml` и `.env` без рестарта |
| `POST` | `/v1/admin/discovery` | Принудительное обновление каталога облачных моделей |
| `POST` | `/v1/admin/check-keys` | Активная проверка доступности всех ключей с замером пинга |

---

## Лицензия

Проект распространяется под лицензией MIT. Исходный код доступен на GitHub: **[https://github.com/marezvov/localAiRouter](https://github.com/marezvov/localAiRouter)**.
