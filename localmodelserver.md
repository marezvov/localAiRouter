# Развёртывание локального LLM-сервера (`llama.cpp` + Vulkan) на AMD Radeon RX Vega 56 и Intel Core i7-4790K

Полное руководство по сборке `llama.cpp` с поддержкой Vulkan, загрузке модели **Qwen3.6-35B-A3B** через Python `venv` + `huggingface-hub`, настройке гибридного инференса (GPU + CPU) на максимальный контекст **262 144 токенов** и настройке автозапуска через `systemd`.

---

## 1. Конфигурация оборудования и распределение нагрузки

* **Процессор:** Intel Core i7-4790K (4 физических ядра / 8 потоков, архитектура Haswell, без поддержки ReBAR).
* **Оперативная память:** 32 ГБ DDR3 (2400 МГц).
* **Видеокарта:** AMD Radeon RX Vega 56 (8 ГБ HBM2 VRAM, драйвер Mesa RADV `VEGA10` / `GFX900`).
* **ОС:** Ubuntu 26 LTS.
* **Модель:** `Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K_M.gguf` (~21–22 ГБ, архитектура Mixture-of-Experts: 40 слоёв, 35B общих параметров, 3B активных параметров на токен, гибридное внимание Gated DeltaNet + Gated Attention).

### Как распределяется память:
* **VRAM (8 ГБ HBM2):** Неэкспертные слои (Attention, роутеры, эмбеддинги) + весь квантованный KV-кэш на 262k токенов + вычислительные буферы Vulkan (~6.5–7.2 ГБ).
* **RAM (32 ГБ DDR3):** Все 40 слоёв тяжелых MoE-экспертов (~19–20 ГБ) через Page Cache ядра Linux (`mmap`).
* **Итоговая производительность:** ~70–85 токенов/с на обработке промпта и стабильные **~16.2–17.3 токенов/с** на генерации текста.

---

## 2. Подготовка системы и прав доступа к GPU

### 2.1. Установка зависимостей для сборки и Vulkan
```bash
sudo apt update && sudo apt install -y \
  build-essential cmake git \
  libvulkan-dev mesa-vulkan-drivers vulkan-tools glslang-tools glslc \
  python3 python3-venv python3-pip
```

### 2.2. Настройка прав доступа к видеокарте (критично для работы по SSH и в фоне)
По умолчанию в Ubuntu пользователь в SSH-сессии не имеет доступа к `/dev/dri/renderD128`, из-за чего драйвер RADV падает с ошибкой `Permission denied`, откатывается на софтверный рендерер `llvmpipe`, и `llama-server` не видит видеокарту (`Available devices: (none)`).

Добавьте текущего пользователя в группы `render` и `video`:
```bash
sudo usermod -aG render,video $USER
```
Чтобы изменения применились без перезагрузки ПК, обновите группу в текущей сессии (или переподключитесь по SSH):
```bash
newgrp render
```

### 2.3. Проверка работы Vulkan
```bash
vulkaninfo --summary
```
В разделе `Devices` должно отображаться устройство **`AMD Radeon RX Vega (RADV VEGA10)`**.

---

## 3. Сборка `llama.cpp` с поддержкой Vulkan

Создайте рабочую директорию проекта, клонируйте репозиторий и скомпилируйте проект с флагом `-DGGML_VULKAN=ON`:

```bash
mkdir -p ~/projects/llminference
cd ~/projects/llminference

git clone https://github.com
cd llama.cpp

cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)

cd ~/projects/llminference
```

### Проверка, что `llama-server` видит видеокарту:
```bash
./llama.cpp/build/bin/llama-server --list-devices
```
Ожидаемый вывод:
```text
Available devices:
  Vulkan0: AMD Radeon RX Vega (RADV VEGA10) (8192 MiB, 8012 MiB free)
```

---

## 4. Скачивание модели через `venv` и `huggingface-hub`

Создайте изолированное Python-окружение, установите `huggingface-hub` (с ускорителем загрузки `hf_transfer`) и скачайте нужный квант `Q4_K_M`:

```bash
cd ~/projects/llminference

# Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка клиента Hugging Face
pip install -U "huggingface_hub[hf_transfer]"

# Создание папки под модель и загрузка файла Q4_K_M
mkdir -p models/huihui-qwen3.6-35b-a3b
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF \
  --include "*Q4_K_M*.gguf" \
  --local-dir models/huihui-qwen3.6-35b-a3b

# Выход из venv (для работы llama-server оно больше не требуется)
deactivate
```
*(Примечание: если команда `hf` в вашей версии пакета называется по-старому, используйте `huggingface-cli download` с теми же аргументами).*

---

## 5. Команда запуска и подробный разбор всех ключей

### Рабочая команда запуска на максимальный контекст (262 144 токенов):
```bash
cd ~/projects/llminference

./llama.cpp/build/bin/llama-server \
  -m models/huihui-qwen3.6-35b-a3b/Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe 40 \
  --fit off \
  -np 1 \
  -c 262144 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -b 1024 \
  -ub 256 \
  -t 4 \
  -tb 8 \
  --host 0.0.0.0 \
  --port 8080
```

### Подробное объяснение каждого флага

| Ключ | Значение | Что делает и на что влияет |
| :--- | :--- | :--- |
| **`-m`** | `models/.../*.gguf` | Путь к файлу модели в формате GGUF. |
| **`-ngl`** | `99` | (`--n-gpu-layers`) Количество слоёв, выгружаемых на видеокарту. Значение `99` отправляет на GPU все слои модели (в Qwen3.6 их 40). |
| **`--n-cpu-moe`** | `40` | Перехватывает тяжелые слои MoE-экспертов (FFN) из первых 40 слоёв и оставляет их в системной ОЗУ для расчёта на процессоре. На GPU остаются только слои внимания (Attention), роутеры и KV-кэш. Без этого флага 21 ГБ модель не поместится в 8 ГБ VRAM. |
| **`--fit`** | `off` | **Критически важный флаг для RX Vega 56.** Отключает автоматический подбор параметров памяти при старте. Без него `llama-server` делает пробные аллокации огромных буферов, из-за чего драйвер Mesa RADV на архитектуре Vega падает с ошибкой `Not enough memory for command submission` (`ErrorDeviceLost`). |
| **`-np`** | `1` | (`--parallel`) Количество параллельных слотов запросов. По умолчанию сервер резервирует память под 4 слота. Установка `1` отдаёт весь объём KV-кэша одному пользователю, что позволяет вместить полные 262k контекста в 8 ГБ VRAM. |
| **`-c`** | `262144` | (`--ctx-size`) Размер контекстного окна в токенах (256k — нативный максимум Qwen3.6-35B-A3B). |
| **`-fa`** | `on` | (`--flash-attn`) Включает Flash Attention. Обязателен для работы квантования V-кэша (`--cache-type-v q4_0`), без него сервер откажется запускаться с ошибкой `quantized V cache requires flash_attn to be enabled`. |
| **`--cache-type-k`** | `q4_0` | Сжимает ключи (Keys) KV-кэша из 16 бит (`f16`) в 4 бита (`q4_0`). Экономит видеопамять в ~4 раза почти без потери качества. |
| **`--cache-type-v`** | `q4_0` | Сжимает значения (Values) KV-кэша в 4 бита (`q4_0`). Позволяет уместить 262k контекста в 8 ГБ HBM2. |
| **`-b`** | `1024` | (`--batch-size`) Логический размер батча при обработке входного промпта. Увеличение до 1024 ускоряет чтение длинных текстов. |
| **`-ub`** | `256` | (`--ubatch-size`) Физический микро-батч, отправляемый на GPU за один проход. Значение `256` безопасно для 8 ГБ VRAM; повышение до `512` на 262k контекста может вызвать переполнение буферов RADV. |
| **`-t`** | `4` | (`--threads`) Количество потоков CPU для генерации токенов. Равно числу **физических ядер** i7-4790K (4 ядра) — использование виртуальных потоков Hyper-Threading при генерации замедляет работу из-за конкуренции за кэш и пропускную способность DDR3. |
| **`-tb`** | `8` | (`--threads-batch`) Количество потоков CPU на этапе обработки промпта (Prompt Processing). Задействует все 8 логических потоков i7-4790K для ускорения чтения больших входных текстов. |
| **`--host`** | `0.0.0.0` | Делает веб-интерфейс и API доступными по всей локальной сети (а не только с `127.0.0.1`). |
| **`--port`** | `8080` | Порт HTTP-сервера и OpenAI-совместимого API (`http://<IP-сервера>:8080/v1`). |

---

## 6. Важные особенности и подводные камни этой сборки

1. **Управление памятью (`-lm` / `--load-mode` вместо устаревшего `--no-mmap`):**
   * В новых версиях `llama.cpp` флаги `--no-mmap` и `--mlock` удалены и заменены на `-lm <none|mmap|mlock>`.
   * На связке **32 ГБ ОЗУ + RX Vega 56** флаг `-lm none` использовать **не рекомендуется**: при `-lm none` модель забирает ~22 ГБ обычной ОЗУ, после чего драйверу `amdgpu` не хватает свободной системной памяти под GTT-пул (~15.1 ГБ, 50% от ОЗУ по правилу ядра Linux), и запуск падает с `ErrorDeviceLost`.
   * Оставляйте режим по умолчанию (`mmap`): при первом запуске ядро Linux загружает веса с диска в файловый кэш ОЗУ (Page Cache), после чего все последующие запуски занимают ~1.5 секунды, а чтение при генерации идёт напрямую из DDR3 на полной скорости.
2. **Неподдерживаемый `--context-shift`:**
   * В новых версиях `llama.cpp` старый флаг `--ctx-shift` переименован в `--context-shift`, однако для гибридной архитектуры Qwen3.6 (с рекуррентными слоями Gated DeltaNet) автоматический сдвиг контекста не поддерживается и принудительно отключается самим сервером.
3. **Сохранение блоков рассуждений (`<think>`):**
   * По умолчанию шаблон чата Qwen3.6 сохраняет цепочки размышлений в истории диалога (`preserving reasoning`). Это повышает точность в сложных задачах и кодинге, но быстрее заполняет контекст. При необходимости экономить токены можно добавить флаг `--no-reasoning-preserve`.
4. **Рекомендуемые параметры сэмплирования для Qwen3.6:**
   * Для стабильных рассуждений и генерации кода в клиенте/роутере рекомендуется передавать: `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`, `min_p: 0.0`.

---

## 7. Настройка автозапуска через `systemd`

### 7.1. Создание unit-файла
```bash
sudo nano /etc/systemd/system/llama-server.service
```

Вставьте следующую конфигурацию:
```ini
[Unit]
Description=llama.cpp HTTP Server (Qwen3.6-35B-A3B Vulkan)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=maxim
Group=maxim
SupplementaryGroups=render video
Используйте код с осторожностью.
WorkingDirectory=/home/maxim/projects/llminference
ExecStart=/home/maxim/projects/llminference/llama.cpp/build/bin/llama-server 
-m /home/maxim/projects/llminference/models/huihui-qwen3.6-35b-a3b/Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K_M.gguf 
-ngl 99 
--n-cpu-moe 40 
--fit off 
-np 1 
-c 262144 
-fa on 
--cache-type-k q4_0 
--cache-type-v q4_0 
-b 1024 
-ub 256 
-t 4 
-tb 8 
--host 0.0.0.0 
--port 8080
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
```

7.2. Активация и запуск службы

bash sudo systemctl daemon-reload sudo systemctl enable llama-server.service sudo systemctl start llama-server.service 

7.3. Управление и мониторинг

• Статус службы: sudo systemctl status llama-server.service
• Логи в реальном времени: journalctl -u llama-server.service -f
• Перезапуск после изменения флагов: sudo systemctl daemon-reload && sudo systemctl restart llama-server.service
• Остановка службы: sudo systemctl stop llama-server.service
• Открытие порта в файрволе (если включён ufw): sudo ufw allow 8080/tcp