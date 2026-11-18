import os
import re
import glob
import shutil
import platform
import subprocess
import logging
from typing import Dict, Any, List, Optional
import psutil

logger = logging.getLogger("gateway.monitor")

# Инициализируем системный счетчик CPU при загрузке модуля
try:
    psutil.cpu_percent(interval=None)
except Exception:
    pass


def get_cpu_model_name() -> str:
    """
    Динамически определяет точную модель центрального процессора в текущей ОС.
    Не использует захардкоженные имена.
    """
    # 1. Linux: /proc/cpuinfo
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("model name") or line.startswith("Processor"):
                        name = line.split(":", 1)[1].strip()
                        if name:
                            return name
        except Exception:
            pass

    # 2. Windows: Registry HKEY_LOCAL_MACHINE
    if os.name == "nt":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if name and str(name).strip():
                return str(name).strip()
        except Exception:
            pass

    # 3. macOS (Darwin): sysctl
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=1.0).decode("utf-8").strip()
            if out:
                return out
        except Exception:
            pass

    # 4. Fallback через platform
    val = platform.processor() or platform.machine()
    return val if val else "Host CPU"


def get_cpu_metrics() -> Dict[str, Any]:
    """Сбор загрузки процессора, частоты, модели и ядер."""
    try:
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        overall = psutil.cpu_percent(interval=None)
        if overall == 0.0 and per_core:
            overall = round(sum(per_core) / len(per_core), 1)

        freq_current = None
        try:
            freq = psutil.cpu_freq()
            freq_current = round(freq.current, 0) if freq else None
        except Exception:
            pass

        # Динамический поиск температуры CPU
        cpu_temp = None
        temps = getattr(psutil, "sensors_temperatures", lambda: {})()
        for sensor_name in ("coretemp", "k10temp", "cpu_thermal", "acpitz", "soc_thermal"):
            if sensor_name in temps and temps[sensor_name]:
                entries = temps[sensor_name]
                for entry in entries:
                    lbl = entry.label.lower()
                    if any(k in lbl for k in ("package", "tctl", "die", "cpu")):
                        cpu_temp = entry.current
                        break
                if cpu_temp is None and entries:
                    cpu_temp = entries[0].current
                if cpu_temp is not None:
                    break

        phys_cores = psutil.cpu_count(logical=False) or 1
        log_cores = psutil.cpu_count(logical=True) or phys_cores

        return {
            "model_name": get_cpu_model_name(),
            "overall_percent": overall,
            "per_core_percent": per_core,
            "physical_cores": phys_cores,
            "logical_cores": log_cores,
            "frequency_mhz": freq_current,
            "temperature_c": round(cpu_temp, 1) if cpu_temp is not None else None,
        }
    except Exception as e:
        logger.debug(f"Error getting CPU metrics: {e}")
        return {
            "model_name": get_cpu_model_name(),
            "overall_percent": 0.0,
            "per_core_percent": [],
            "physical_cores": 1,
            "logical_cores": 1,
            "frequency_mhz": None,
            "temperature_c": None,
        }


def get_memory_metrics() -> Dict[str, Any]:
    """Сбор использования оперативной памяти (RAM) и файла подкачки (Swap)."""
    ram_total_gb = 0.0
    ram_used_gb = 0.0
    ram_available_gb = 0.0
    ram_percent = 0.0

    try:
        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / (1024**3), 2)
        ram_used_gb = round(vm.used / (1024**3), 2)
        ram_available_gb = round(vm.available / (1024**3), 2)
        ram_percent = vm.percent
    except Exception as e:
        logger.debug(f"Error getting virtual_memory: {e}")

    swap_total_gb = 0.0
    swap_used_gb = 0.0
    swap_percent = 0.0
    try:
        sw = psutil.swap_memory()
        swap_total_gb = round(sw.total / (1024**3), 2)
        swap_used_gb = round(sw.used / (1024**3), 2)
        swap_percent = sw.percent
    except Exception as e:
        logger.debug(f"Error getting swap_memory: {e}")

    return {
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_available_gb": ram_available_gb,
        "ram_percent": ram_percent,
        "swap_total_gb": swap_total_gb,
        "swap_used_gb": swap_used_gb,
        "swap_percent": swap_percent,
    }


def _read_sysfs_int(path: str) -> Optional[int]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return int(f.read().strip())
    except Exception:
        pass
    return None


def _read_sysfs_str(path: str) -> Optional[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _detect_linux_pci_gpus() -> List[str]:
    """Динамический поиск названий GPU через lspci на Linux."""
    names = []
    if shutil.which("lspci"):
        try:
            out = subprocess.check_output(["lspci", "-nn"], timeout=2.0).decode("utf-8", errors="ignore")
            for line in out.splitlines():
                lower = line.lower()
                if any(k in lower for k in ("vga compatible", "3d controller", "display controller")):
                    # Ищем коммерческое название в квадратных скобках [GeForce RTX ...] / [Radeon RX ...]
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        card_desc = parts[2].strip()
                        brackets = re.findall(r"\[(.*?)\]", card_desc)
                        # Обычно brackets содержит [Vendor] [Chip] [Friendly Name] [PCI-ID]
                        friendly = None
                        for b in brackets:
                            if any(w in b.lower() for w in ("geforce", "radeon", "arc", "rtx", "gtx", "quadro", "tesla", "instinct", "vega", "navi")):
                                friendly = b
                                break
                        if friendly:
                            names.append(friendly)
                        elif brackets:
                            names.append(brackets[-1] if not re.match(r"^[0-9a-f]{4}:[0-9a-f]{4}$", brackets[-1].lower()) else brackets[0])
                        else:
                            names.append(card_desc)
        except Exception:
            pass
    return names


def _detect_windows_display_adapters() -> List[Dict[str, Any]]:
    """Динамическое определение видеокарт через реестр Windows."""
    adapters = []
    if os.name == "nt":
        try:
            import winreg
            for i in range(16):
                try:
                    key = winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        rf"SYSTEM\CurrentControlSet\Control\Class\{{4d36e968-e325-11ce-bfc1-08002be10318}}\{i:04d}"
                    )
                    desc, _ = winreg.QueryValueEx(key, "DriverDesc")
                    vram_bytes = 0
                    for vname in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            v, _ = winreg.QueryValueEx(key, vname)
                            if v and int(v) > 0:
                                vram_bytes = int(v)
                                break
                        except Exception:
                            pass
                    winreg.CloseKey(key)
                    if desc and str(desc).strip():
                        adapters.append({
                            "name": str(desc).strip(),
                            "vram_bytes": vram_bytes,
                            "vram_total_gb": round(vram_bytes / (1024**3), 2) if vram_bytes > 0 else 0.0,
                        })
                except Exception:
                    pass
        except Exception:
            pass
    return adapters


def get_gpu_metrics() -> Dict[str, Any]:
    """
    Полностью динамическое обнаружение ускорителей (NVIDIA, AMD, Intel, Apple Silicon, etc.)
    без жестко зашитых конфигураций оборудования.
    """
    # 1. Проверяем NVIDIA через nvidia-smi (если доступен в системе - Linux или Windows)
    if shutil.which("nvidia-smi"):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            out = subprocess.check_output(cmd, timeout=2.0).decode("utf-8", errors="ignore").strip()
            if out:
                first_line = out.splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 5:
                    gpu_name = parts[0]
                    gpu_util = float(parts[1]) if parts[1] else 0.0
                    mem_used_mb = float(parts[2]) if parts[2] else 0.0
                    mem_total_mb = float(parts[3]) if parts[3] else 0.0
                    temp_gpu = float(parts[4]) if parts[4] else None
                    power_w = float(parts[5]) if len(parts) > 5 and parts[5] else None

                    vram_total_gb = round(mem_total_mb / 1024.0, 2)
                    vram_used_gb = round(mem_used_mb / 1024.0, 2)
                    vram_pct = round((mem_used_mb / mem_total_mb) * 100.0, 1) if mem_total_mb > 0 else 0.0

                    sensors_dict = {}
                    if temp_gpu is not None:
                        sensors_dict["GPU"] = temp_gpu

                    return {
                        "available": True,
                        "name": gpu_name,
                        "vendor": "NVIDIA",
                        "gpu_percent": gpu_util,
                        "vram_total_gb": vram_total_gb,
                        "vram_used_gb": vram_used_gb,
                        "vram_percent": vram_pct,
                        "temp_c": temp_gpu,
                        "temp_edge_c": temp_gpu,
                        "temp_junction_c": None,
                        "temp_memory_c": None,
                        "power_watts": power_w,
                        "sensors": sensors_dict,
                    }
        except Exception:
            pass

    # 2. Проверяем Linux DRM sysfs (/sys/class/drm/card*/device) для AMD / Intel / Nouveau
    card_devices = glob.glob("/sys/class/drm/card*/device")
    if card_devices:
        pci_names = _detect_linux_pci_gpus()

        for dev_path in card_devices:
            vram_total_path = os.path.join(dev_path, "mem_info_vram_total")
            vram_used_path = os.path.join(dev_path, "mem_info_vram_used")
            gpu_busy_path = os.path.join(dev_path, "gpu_busy_percent")

            # Проверяем, есть ли поддержка VRAM / busy в драйвере (amdgpu / i915 / xe)
            if os.path.exists(vram_total_path) or os.path.exists(gpu_busy_path):
                vram_total = _read_sysfs_int(vram_total_path) or 0
                vram_used = _read_sysfs_int(vram_used_path) or 0
                gpu_busy = _read_sysfs_int(gpu_busy_path) or 0

                # Определяем производителя по драйверу или vendor
                uevent_content = _read_sysfs_str(os.path.join(dev_path, "uevent")) or ""
                vendor_id = _read_sysfs_str(os.path.join(dev_path, "vendor")) or ""

                if "amdgpu" in uevent_content or "0x1002" in vendor_id:
                    vendor = "AMD"
                elif "i915" in uevent_content or "xe" in uevent_content or "0x8086" in vendor_id:
                    vendor = "Intel"
                elif "nvidia" in uevent_content or "0x10de" in vendor_id:
                    vendor = "NVIDIA"
                else:
                    vendor = "GPU Accelerator"

                # Определяем имя модели GPU
                gpu_name = None
                # Пробуем взять из lspci
                for pci_n in pci_names:
                    if vendor.lower() in pci_n.lower() or "radeon" in pci_n.lower() or "geforce" in pci_n.lower():
                        gpu_name = pci_n
                        break
                if not gpu_name and pci_names:
                    gpu_name = pci_names[0]
                if not gpu_name:
                    product_n = _read_sysfs_str(os.path.join(dev_path, "product_name"))
                    gpu_name = product_n if product_n else f"{vendor} Graphics Accelerator"

                # Динамический сбор термодатчиков из hwmon
                edge_temp = None
                junction_temp = None
                mem_temp = None
                power_w = None
                sensors_dict = {}

                hwmon_dirs = glob.glob(os.path.join(dev_path, "hwmon", "hwmon*"))
                for hdir in hwmon_dirs:
                    power_raw = _read_sysfs_int(os.path.join(hdir, "power1_average")) or _read_sysfs_int(os.path.join(hdir, "power1_input"))
                    if power_raw:
                        power_w = round(power_raw / 1_000_000.0, 1)

                    # Динамически сканируем все temp*_input и temp*_label
                    for temp_file in glob.glob(os.path.join(hdir, "temp*_input")):
                        t_val = _read_sysfs_int(temp_file)
                        if t_val is not None:
                            val_c = round(t_val / 1000.0, 1)
                            # Ищем соответствующий label-файл (например temp1_label)
                            base_prefix = temp_file[:-6]  # отсекаем "_input"
                            label_file = base_prefix + "_label"
                            label_name = _read_sysfs_str(label_file)
                            if not label_name:
                                label_name = os.path.basename(base_prefix)

                            label_lower = label_name.lower()
                            if "edge" in label_lower or "gpu" in label_lower:
                                edge_temp = val_c
                            elif "junction" in label_lower or "hotspot" in label_lower:
                                junction_temp = val_c
                            elif "mem" in label_lower or "hbm" in label_lower or "vram" in label_lower:
                                mem_temp = val_c

                            sensors_dict[label_name.capitalize()] = val_c

                # Дополнительная проверка через psutil.sensors_temperatures()
                if edge_temp is None:
                    temps = getattr(psutil, "sensors_temperatures", lambda: {})()
                    for chip_key in ("amdgpu", "nouveau", "intel_rapl", "soc_dts"):
                        if chip_key in temps:
                            for entry in temps[chip_key]:
                                lbl = (entry.label or chip_key).lower()
                                if "edge" in lbl or not entry.label:
                                    edge_temp = entry.current
                                elif "junction" in lbl or "hotspot" in lbl:
                                    junction_temp = entry.current
                                elif "mem" in lbl or "hbm" in lbl:
                                    mem_temp = entry.current
                                sensors_dict[lbl.capitalize()] = entry.current

                vram_total_gb = round(vram_total / (1024**3), 2)
                vram_used_gb = round(vram_used / (1024**3), 2)
                vram_percent = round((vram_used / vram_total) * 100.0, 1) if vram_total > 0 else 0.0

                primary_temp = edge_temp or junction_temp or (next(iter(sensors_dict.values())) if sensors_dict else None)

                return {
                    "available": True,
                    "name": gpu_name,
                    "vendor": vendor,
                    "gpu_percent": gpu_busy,
                    "vram_total_gb": vram_total_gb,
                    "vram_used_gb": vram_used_gb,
                    "vram_percent": vram_percent,
                    "temp_c": primary_temp,
                    "temp_edge_c": edge_temp,
                    "temp_junction_c": junction_temp,
                    "temp_memory_c": mem_temp,
                    "power_watts": power_w,
                    "sensors": sensors_dict,
                }

    # 3. Проверяем Windows Display Adapters через реестр (если Windows и nvidia-smi не запущен)
    if os.name == "nt":
        win_gpus = _detect_windows_display_adapters()
        if win_gpus:
            # Выбираем адаптер с наибольшей VRAM (обычно дискретная карта)
            best_gpu = max(win_gpus, key=lambda g: g.get("vram_bytes", 0))
            gpu_name = best_gpu["name"]
            vendor = "NVIDIA" if "nvidia" in gpu_name.lower() else ("AMD" if "amd" in gpu_name.lower() or "radeon" in gpu_name.lower() else ("Intel" if "intel" in gpu_name.lower() else "Unknown"))

            return {
                "available": True,
                "name": gpu_name,
                "vendor": vendor,
                "gpu_percent": 0.0,
                "vram_total_gb": best_gpu["vram_total_gb"],
                "vram_used_gb": 0.0,
                "vram_percent": 0.0,
                "temp_c": None,
                "temp_edge_c": None,
                "temp_junction_c": None,
                "temp_memory_c": None,
                "power_watts": None,
                "sensors": {},
            }

    # 4. Проверяем Apple Silicon на macOS (Darwin)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        cpu_name = get_cpu_model_name()
        vm = psutil.virtual_memory()
        total_unified_gb = round(vm.total / (1024**3), 2)
        return {
            "available": True,
            "name": f"{cpu_name} (Apple Silicon GPU)",
            "vendor": "Apple",
            "gpu_percent": 0.0,
            "vram_total_gb": total_unified_gb,
            "vram_used_gb": round(vm.used / (1024**3), 2),
            "vram_percent": vm.percent,
            "temp_c": None,
            "temp_edge_c": None,
            "temp_junction_c": None,
            "temp_memory_c": None,
            "power_watts": None,
            "sensors": {},
        }

    # 5. Ускоритель не найден
    return {
        "available": False,
        "name": "No Accelerator Detected",
        "vendor": "None",
        "gpu_percent": 0.0,
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_percent": 0.0,
        "temp_c": None,
        "temp_edge_c": None,
        "temp_junction_c": None,
        "temp_memory_c": None,
        "power_watts": None,
        "sensors": {},
    }


def get_all_temperatures() -> List[Dict[str, Any]]:
    """
    Собирает все доступные аппаратные термодатчики (CPU, GPU, чипсет, NVMe, диски).
    Возвращает список структурированных записей для таблицы температур.
    """
    sensors_list: List[Dict[str, Any]] = []

    # 1. Из psutil.sensors_temperatures()
    temps_dict = getattr(psutil, "sensors_temperatures", lambda: {})()
    for chip_name, entries in temps_dict.items():
        for entry in entries:
            label = entry.label if entry.label else chip_name
            current = entry.current
            high = entry.high or 80.0
            critical = entry.critical or 95.0

            if current >= critical or current >= 90.0:
                status = "critical"
            elif current >= high or current >= 75.0:
                status = "warning"
            else:
                status = "normal"

            sensors_list.append({
                "chip": chip_name,
                "label": label,
                "current_c": round(current, 1),
                "high_c": round(high, 1) if entry.high else None,
                "critical_c": round(critical, 1) if entry.critical else None,
                "status": status,
            })

    # 2. Если psutil пуст, читаем Linux sysfs /sys/class/thermal
    if not sensors_list and os.path.exists("/sys/class/thermal"):
        thermal_zones = glob.glob("/sys/class/thermal/thermal_zone*")
        for zone in thermal_zones:
            temp_path = os.path.join(zone, "temp")
            type_path = os.path.join(zone, "type")
            t_val = _read_sysfs_int(temp_path)
            t_name = "Thermal Zone"
            if os.path.exists(type_path):
                t_str = _read_sysfs_str(type_path)
                if t_str:
                    t_name = t_str

            if t_val is not None:
                current = round(t_val / 1000.0, 1)
                status = "critical" if current >= 90 else ("warning" if current >= 75 else "normal")
                sensors_list.append({
                    "chip": os.path.basename(zone),
                    "label": t_name,
                    "current_c": current,
                    "high_c": 80.0,
                    "critical_c": 95.0,
                    "status": status,
                })

    sensors_list.sort(key=lambda s: s["current_c"], reverse=True)
    return sensors_list


def get_disk_metrics() -> Dict[str, Any]:
    """Сбор информации о системном накопителе."""
    try:
        root_path = "/" if os.name != "nt" else os.path.abspath(os.sep)
        du = psutil.disk_usage(root_path)
        return {
            "path": root_path,
            "total_gb": round(du.total / (1024**3), 1),
            "used_gb": round(du.used / (1024**3), 1),
            "free_gb": round(du.free / (1024**3), 1),
            "percent": du.percent,
        }
    except Exception as e:
        logger.debug(f"Error getting disk metrics: {e}")
        return {"path": "/", "total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}


def get_full_system_metrics() -> Dict[str, Any]:
    """Полная сводка системных метрик и температур для дашборда и JSON API."""
    return {
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "gpu": get_gpu_metrics(),
        "temperatures": get_all_temperatures(),
        "disk": get_disk_metrics(),
    }
