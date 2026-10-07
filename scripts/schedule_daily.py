"""Set up a Windows Scheduled Task to run scripts/run_prediction.py every
morning.

Uso:
  python scripts/schedule_daily.py                    # crea la tarea, 8:00 AM
  python scripts/schedule_daily.py --hour 7 --minute 30
  python scripts/schedule_daily.py --remove            # borra la tarea

run_prediction.py ejecuta daily_workflow (paso 4/4): guarda picks que superan
el filtro de actualidad y edge >= 3% con 1/4 Kelly, y envia el correo diario.
El runner pasa --no-push: la tarea programada nunca crea commits ni hace push.
Ver docs/metrics/2026-09-18-scanner-automation.md.

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

    # schtasks rejects a /tr value over 261 characters, which the full
    # interpreter + script paths easily exceed in a deep checkout. Point the
    # task at a short launcher that holds the real command instead.
    launcher = log_dir / LAUNCHER_NAME
    launcher.write_text(
        f'@echo off\ncd /d "{_ROOT}"\n{command}\n', encoding="utf-8", newline="\r\n",
    )
    task_command = f'"{launcher}"'
    if len(task_command) > MAX_TR_LENGTH:
        raise ValueError(
            f"Ruta del lanzador demasiado larga para schtasks ({len(task_command)} > "
            f"{MAX_TR_LENGTH} caracteres): {launcher}"
        )

    return subprocess.run(
        [
            "schtasks", "/create", "/tn", task_name, "/tr", task_command,
            "/sc", "daily", "/st", time_str, "/f",
        ],
        capture_output=True, text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configura la tarea programada diaria de predicciones (Windows Task Scheduler)"
    )
    parser.add_argument("--hour", type=int, default=8, help="Hora local, 0-23 (default: 8)")
    parser.add_argument("--minute", type=int, default=0, help="Minuto, 0-59 (default: 0)")
    parser.add_argument("--tour", choices=["atp", "wta", "davis", "both"], default="both")
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--remove", action="store_true", help="Borra la tarea en vez de crearla")
    args = parser.parse_args()

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
