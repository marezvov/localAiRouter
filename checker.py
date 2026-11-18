import sys
import time
import asyncio
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config_loader
from key_manager import get_key_manager
from local_manager import get_local_manager
from discovery import ModelDiscovery

console = Console()


async def check_local_deep(url: str, timeout_sec: float) -> tuple[bool, float, str]:
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(
                f"{url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }
            )
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                return True, elapsed, "OK"
            return False, elapsed, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, (time.time() - start) * 1000, str(e)[:40]


async def check_cloud_key_deep(base_url: str, key: str, timeout_sec: float, extra_headers: dict) -> tuple[bool, float, str]:
    start = time.time()
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **extra_headers,
        }
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            # Сначала проверяем /models как быстрый пинг
            resp = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                return True, elapsed, "OK"
            return False, elapsed, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, (time.time() - start) * 1000, str(e)[:40]


async def run_diagnostics(deep: bool = False):
    cfg = config_loader.get_config()
    secrets = config_loader.get_secrets()

    console.print(Panel.fit(
        f"[bold cyan]AI Gateway Diagnostics[/bold cyan] [dim](Deep: {deep})[/dim]\n"
        f"Gateway Key: [bold green]{secrets.gateway_api_key[:4]}***{secrets.gateway_api_key[-3:]}[/bold green] | "
        f"Server: [bold white]{cfg.server.host}:{cfg.server.port}[/bold white]",
        border_style="cyan"
    ))

    # 1. Local Backend Status
    local_mgr = get_local_manager(cfg.local_backend)
    loc_health = await local_mgr.check_health()
    loc_online = loc_health.get("online", False)

    loc_table = Table(title="Local Backend (llama-server)", border_style="blue")
    loc_table.add_column("Property", style="bold")
    loc_table.add_column("Value")

    loc_table.add_row("Status", "[bold green]ONLINE[/bold green]" if loc_online else "[bold red]OFFLINE[/bold red]")
    loc_table.add_row("URL", cfg.local_backend.url)
    loc_table.add_row("Model", cfg.local_backend.model_name)
    loc_table.add_row("Slot 0 Occupied", "[yellow]YES[/yellow]" if loc_health.get("slot_occupied") else "[green]NO[/green]")
    loc_table.add_row("KV-Cache Tokens (n_past)", f"{loc_health.get('n_past', 0):,}")
    loc_table.add_row("Speed (t/s)", str(loc_health.get("tokens_per_sec", 0)))
    loc_table.add_row("Queue Waiters", str(loc_health.get("queue_length", 0)))

    if deep and loc_online:
        with console.status("[bold green]Pinging local model..."):
            ok, ping_ms, ping_msg = await check_local_deep(cfg.local_backend.url, 15.0)
            loc_table.add_row("Deep Ping (1 token)", f"{'[green]' if ok else '[red]'}{ping_msg} ({ping_ms:.1f}ms)[/]")

    console.print(loc_table)
    console.print()

    # 2. Providers & Keys
    key_mgr = get_key_manager()
    for pid, pcfg in cfg.providers.items():
        keys = secrets.provider_keys.get(pcfg.api_key_env, [])
        await key_mgr.register_provider_keys(pid, keys)

    prov_table = Table(title="Cloud Providers & API Keys", border_style="magenta")
    prov_table.add_column("Provider", style="bold cyan")
    prov_table.add_column("Masked Key", style="cyan")
    prov_table.add_column("Status")
    prov_table.add_column("Base URL", style="dim")
    if deep:
        prov_table.add_column("Deep Latency")

    discovery = ModelDiscovery(cfg, key_mgr, secrets.outbound_proxy)
    if deep:
        with console.status("[bold magenta]Refreshing cloud models discovery..."):
            await discovery.refresh_all(force=True)

    for pid, pcfg in cfg.providers.items():
        keys = secrets.provider_keys.get(pcfg.api_key_env, [])
        if not keys:
            prov_table.add_row(
                pid,
                "[dim](none)[/dim]",
                "[yellow]NO KEYS IN .ENV[/yellow]",
                pcfg.base_url,
                *(["-"] if deep else []),
            )
            continue

        for k in keys:
            masked = f"{k[:4]}...{k[-4:]}" if len(k) > 8 else "***"
            status = "[green]ACTIVE[/green]" if pcfg.enabled else "[dim]DISABLED[/dim]"

            row = [pid, masked, status, pcfg.base_url]

            if deep:
                if pcfg.enabled:
                    ok, lat, msg = await check_cloud_key_deep(pcfg.base_url, k, 10.0, pcfg.extra_headers)
                    color = "green" if ok else "red"
                    row.append(f"[{color}]{msg} ({lat:.1f}ms)[/{color}]")
                else:
                    row.append("[dim]skipped[/dim]")

            prov_table.add_row(*row)

    console.print(prov_table)
    console.print()

    # 3. Virtual Aliases
    alias_table = Table(title="Virtual Model Aliases", border_style="green")
    alias_table.add_column("Alias", style="bold green")
    alias_table.add_column("Priority Chain")
    alias_table.add_column("Requirements")
    alias_table.add_column("Description", style="dim")

    for aname, acfg in cfg.aliases.items():
        reqs = []
        if acfg.require_tools:
            reqs.append("tools")
        if acfg.require_vision:
            reqs.append("vision")
        req_str = ", ".join(reqs) if reqs else "-"
        if acfg.models:
            prio_str = " -> ".join(acfg.models)
        else:
            prio_str = " -> ".join(acfg.provider_priority or acfg.priority or cfg.routing.provider_priority)
        alias_table.add_row(aname, prio_str, req_str, acfg.description)

    console.print(alias_table)

    # 4. System & Hardware Metrics
    import system_monitor
    sys_metrics = system_monitor.get_full_system_metrics()
    cpu = sys_metrics.get("cpu", {})
    mem = sys_metrics.get("memory", {})
    gpu = sys_metrics.get("gpu", {})
    disk = sys_metrics.get("disk", {})
    temps = sys_metrics.get("temperatures", [])

    sys_table = Table(title="Hardware & System Resource Status", border_style="yellow")
    sys_table.add_column("Subsystem", style="bold yellow")
    sys_table.add_column("Details")
    sys_table.add_column("Utilization / Status")

    # CPU
    cpu_model = cpu.get("model_name", "Host CPU")
    cpu_freq_str = f" @ {int(cpu.get('frequency_mhz'))} MHz" if cpu.get("frequency_mhz") else ""
    cpu_temp_str = f" | Temp: {cpu.get('temperature_c')}°C" if cpu.get("temperature_c") else ""
    sys_table.add_row(
        f"CPU ({cpu_model})",
        f"{cpu.get('physical_cores', 1)} Cores / {cpu.get('logical_cores', 1)} Threads{cpu_freq_str}{cpu_temp_str}",
        f"{cpu.get('overall_percent', 0.0)}%"
    )

    # RAM
    ram_str = f"{mem.get('ram_used_gb', 0)} / {mem.get('ram_total_gb', 0)} GB (Free: {mem.get('ram_available_gb', 0)} GB)"
    sys_table.add_row("RAM Memory", ram_str, f"{mem.get('ram_percent', 0)}%")

    # Swap (if present)
    if mem.get("swap_total_gb", 0) > 0:
        sys_table.add_row("Swap", f"{mem.get('swap_used_gb', 0)} / {mem.get('swap_total_gb', 0)} GB", f"{mem.get('swap_percent', 0)}%")

    # GPU
    if gpu.get("available"):
        gpu_name = gpu.get("name", "GPU")
        vram_str = f"VRAM: {gpu.get('vram_used_gb', 0)} / {gpu.get('vram_total_gb', 0)} GB ({gpu.get('vram_percent', 0)}%)" if gpu.get("vram_total_gb", 0) > 0 else "Active"
        power_str = f" | Power: {gpu.get('power_watts')} W" if gpu.get('power_watts') else ""
        temps_gpu = []
        gpu_sensors = gpu.get("sensors", {})
        if gpu_sensors:
            for s_name, s_val in gpu_sensors.items():
                temps_gpu.append(f"{s_name}: {s_val}°C")
        elif gpu.get("temp_c") is not None:
            temps_gpu.append(f"{gpu['temp_c']}°C")
        elif gpu.get("temp_edge_c") is not None:
            temps_gpu.append(f"Edge: {gpu.get('temp_edge_c')}°C")

        gpu_details = f"{vram_str}{power_str}" + (f" | {' / '.join(temps_gpu)}" if temps_gpu else "")
        status_str = f"{gpu.get('gpu_percent', 0)}%" if gpu.get("gpu_percent", 0) > 0 else "[green]DETECTED[/green]"
        sys_table.add_row(f"GPU ({gpu_name})", gpu_details, status_str)
    else:
        sys_table.add_row("GPU", gpu.get("name", "No accelerator detected"), "[dim]N/A[/dim]")

    # Disk
    disk_str = f"{disk.get('used_gb', 0)} / {disk.get('total_gb', 0)} GB on {disk.get('path', '/')}"
    sys_table.add_row("Disk Storage", disk_str, f"{disk.get('percent', 0)}%")

    console.print()
    console.print(sys_table)

    # Temperatures table (if any detected)
    if temps:
        temp_table = Table(title="Hardware Thermal Sensors (°C)", border_style="red")
        temp_table.add_column("Chip / Device", style="bold cyan")
        temp_table.add_column("Sensor Zone")
        temp_table.add_column("Current")
        temp_table.add_column("High")
        temp_table.add_column("Critical")
        temp_table.add_column("Status")

        for t in temps:
            st = t.get("status", "normal")
            color = "green" if st == "normal" else ("yellow" if st == "warning" else "bold red")
            temp_table.add_row(
                t.get("chip", "-"),
                t.get("label", "-"),
                f"[{color}]{t.get('current_c')}°C[/{color}]",
                f"{t.get('high_c')}°C" if t.get("high_c") else "-",
                f"{t.get('critical_c')}°C" if t.get("critical_c") else "-",
                f"[{color}]{st.upper()}[/{color}]"
            )
        console.print()
        console.print(temp_table)


if __name__ == "__main__":
    deep_mode = "--deep" in sys.argv
    asyncio.run(run_diagnostics(deep=deep_mode))
