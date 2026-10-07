"""Set up a Windows Scheduled Task to run scripts/run_prediction.py every
morning.

Uso:
  python scripts/schedule_daily.py                    # crea la tarea, 8:00 AM
  python scripts/schedule_daily.py --hour 7 --minute 30
  python scripts/schedule_daily.py --remove            # borra la tarea
  python scripts/schedule_daily.py --odds-snapshots    # tarea de snapshots de cuotas, cada 6 h
  python scripts/schedule_daily.py --odds-snapshots --remove

run_prediction.py ejecuta daily_workflow (paso 4/4): guarda picks que superan
el filtro de actualidad y edge >= 3% con 1/4 Kelly, y envia el correo diario.
El runner pasa --no-push: la tarea programada nunca crea commits ni hace push.
Ver docs/metrics/2026-09-18-scanner-automation.md.

--odds-snapshots registra otra tarea (ia-tenis-odds-snapshots) que corre
scripts/snapshot_odds.py cada --every-hours horas desde --hour:--minute,
para construir el historico de cuotas de apertura/cierre.

Nota sobre zona horaria: schtasks no tiene concepto de IANA timezone -- usa
siempre la hora local de la maquina. --hour/--minute se interpretan como
hora local (America/Lima, si la maquina esta configurada asi).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TASK_NAME = "ia-tenis-daily-prediction"
LAUNCHER_NAME = "run_daily_task.cmd"
ODDS_SNAPSHOT_TASK_NAME = "ia-tenis-odds-snapshots"
ODDS_SNAPSHOT_LAUNCHER_NAME = "run_odds_snapshot_task.cmd"
MAX_TR_LENGTH = 261


def _validate_time(hour: int, minute: int) -> None:
    if not (0 <= hour <= 23):
        raise ValueError(f"Hora invalida: {hour} (debe estar entre 0 y 23)")
    if not (0 <= minute <= 59):
        raise ValueError(f"Minuto invalido: {minute} (debe estar entre 0 y 59)")


def task_exists(task_name: str = DEFAULT_TASK_NAME) -> bool:
    result = subprocess.run(
        ["schtasks", "/query", "/tn", task_name],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def remove_task(task_name: str = DEFAULT_TASK_NAME) -> None:
    if task_exists(task_name):
        subprocess.run(
            ["schtasks", "/delete", "/tn", task_name, "/f"],
            capture_output=True, text=True,
        )


def _write_launcher(launcher: Path, command: str) -> str:
    """schtasks rejects a /tr value over 261 characters, which the full
    interpreter + script paths easily exceed in a deep checkout. Point the
    task at a short launcher that holds the real command instead; returns
    the /tr value."""
    launcher.write_text(
        f'@echo off\ncd /d "{_ROOT}"\n{command}\n', encoding="utf-8", newline="\r\n",
    )
    task_command = f'"{launcher}"'
    if len(task_command) > MAX_TR_LENGTH:
        raise ValueError(
            f"Ruta del lanzador demasiado larga para schtasks ({len(task_command)} > "
            f"{MAX_TR_LENGTH} caracteres): {launcher}"
        )
    return task_command


def _allow_on_battery(task_name: str):
    """schtasks /create can't set power conditions, and its defaults keep a
    task from starting on battery (it just sits "Queued") and kill it when
    the laptop unplugs -- found 2026-10-07 on the production laptop. Also
    turn on StartWhenAvailable so a run missed while asleep happens later."""
    script = (
        "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -StartWhenAvailable; "
        f"Set-ScheduledTask -TaskName '{task_name}' -Settings $s | Out-Null"
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
    )


def _create_and_configure(schtasks_cmd: list[str], task_name: str):
    result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return result
    return _allow_on_battery(task_name)


def create_task(
    hour: int = 8, minute: int = 0, task_name: str = DEFAULT_TASK_NAME,
    tour: str = "both", days_ahead: int = 1,
):
    """Create (or replace, idempotently) a daily Windows Scheduled Task
    that runs scripts/run_prediction.py --log at hour:minute
    local time, via a launcher at data/logs/run_daily_task.cmd. Removes any existing task of the same name first -- never
    duplicates."""
    _validate_time(hour, minute)
    remove_task(task_name)

    log_dir = _ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_script = _ROOT / "scripts" / "run_prediction.py"
    time_str = f"{hour:02d}:{minute:02d}"
    command = (
        f'"{sys.executable}" "{run_script}" '
        f"--tour {tour} --days-ahead {days_ahead} --log"
    )

    task_command = _write_launcher(log_dir / LAUNCHER_NAME, command)

    return _create_and_configure(
        [
            "schtasks", "/create", "/tn", task_name, "/tr", task_command,
            "/sc", "daily", "/st", time_str, "/f",
        ],
        task_name,
    )


def create_odds_snapshot_task(
    hour: int = 1, minute: int = 0, every_hours: int = 6, task_name: str = ODDS_SNAPSHOT_TASK_NAME,
):
    """Create (or replace) a Scheduled Task running scripts/snapshot_odds.py
    every `every_hours` hours starting at hour:minute local time."""
    _validate_time(hour, minute)
    if not (1 <= every_hours <= 23):
        raise ValueError(f"Intervalo invalido: {every_hours} (debe estar entre 1 y 23 horas)")
    remove_task(task_name)

    log_dir = _ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = _ROOT / "scripts" / "snapshot_odds.py"
    log_file = log_dir / "odds_snapshots.log"
    command = f'"{sys.executable}" "{script}" --commit >> "{log_file}" 2>&1'
    task_command = _write_launcher(log_dir / ODDS_SNAPSHOT_LAUNCHER_NAME, command)

    return _create_and_configure(
        [
            "schtasks", "/create", "/tn", task_name, "/tr", task_command,
            "/sc", "hourly", "/mo", str(every_hours), "/st", f"{hour:02d}:{minute:02d}", "/f",
        ],
        task_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configura la tarea programada diaria de predicciones (Windows Task Scheduler)"
    )
    parser.add_argument("--hour", type=int, default=None,
                        help="Hora local, 0-23 (default: 8; con --odds-snapshots, 1)")
    parser.add_argument("--minute", type=int, default=0, help="Minuto, 0-59 (default: 0)")
    parser.add_argument("--tour", choices=["atp", "wta", "davis", "both"], default="both")
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--remove", action="store_true", help="Borra la tarea en vez de crearla")
    parser.add_argument("--odds-snapshots", action="store_true",
                        help="Configura la tarea de snapshots de cuotas en vez de la diaria")
    parser.add_argument("--every-hours", type=int, default=6, help="Intervalo de snapshots (default: 6)")
    args = parser.parse_args()

    if args.odds_snapshots:
        task_name = args.task_name if args.task_name != DEFAULT_TASK_NAME else ODDS_SNAPSHOT_TASK_NAME
        if args.remove:
            remove_task(task_name)
            print(f"Tarea '{task_name}' eliminada (si existia).")
            return
        hour = 1 if args.hour is None else args.hour
        result = create_odds_snapshot_task(hour, args.minute, args.every_hours, task_name)
        if result.returncode != 0:
            print(f"ERROR creando la tarea: {result.stderr}")
            sys.exit(result.returncode)
        print(f"Tarea '{task_name}' creada: snapshot de cuotas cada {args.every_hours} h desde "
              f"las {hour:02d}:{args.minute:02d}. Log: data/logs/odds_snapshots.log")
        return

    if args.hour is None:
        args.hour = 8
    if args.remove:
        remove_task(args.task_name)
        print(f"Tarea '{args.task_name}' eliminada (si existia).")
        return

    result = create_task(
        hour=args.hour, minute=args.minute, task_name=args.task_name,
        tour=args.tour, days_ahead=args.days_ahead,
    )
    if result.returncode != 0:
        print(f"ERROR creando la tarea: {result.stderr}")
        sys.exit(result.returncode)
    print(f"Tarea '{args.task_name}' creada: corre todos los dias a las {args.hour:02d}:{args.minute:02d}.")
    print("El flujo diario registra picks elegibles con 1/4 Kelly y envia correo; no hace commit ni push.")


if __name__ == "__main__":
    main()
