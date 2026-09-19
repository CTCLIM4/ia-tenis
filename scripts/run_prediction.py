"""Runner unico para producir predicciones en vivo.

Uso:
  python scripts/run_prediction.py              # ATP + WTA + Davis
  python scripts/run_prediction.py --tour wta    # solo WTA
  python scripts/run_prediction.py --days-ahead 2  # incluye proximos 2 dias
  python scripts/run_prediction.py --log         # tambien escribe data/logs/daily_{fecha}.log

Flujo:
  1. Descarga data fresca (ATP + WTA + Davis Cup)
  2. Corre pipeline de features para cada tour
  3. Reentrena modelos LR
  4. Ejecuta scanner diario (--auto-save, no interactivo)
  5. Imprime resumen con picks generados
"""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log_path_for_today(root: Path = ROOT) -> Path:
    """data/logs/daily_{YYYY-MM-DD}.log under root, creating the directory
    if needed."""
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"daily_{date.today().isoformat()}.log"


def run(cmd: list[str], check: bool = True, stdin_input: str = "\n", log_file=None) -> int:
    """Run a subprocess command and return exit code.

    log_file: an open, writable file object -- when given, the command's
    header line and its stdout/stderr both go there (in addition to the
    console), so an unattended scheduled run leaves a real record instead
    of output nobody was watching."""
    header = f"\n>>> {' '.join(cmd)}"
    print(header)
    if log_file is not None:
        log_file.write(header + "\n")
        log_file.flush()
        result = subprocess.run(
            cmd, cwd=str(ROOT), input=stdin_input.encode(),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
    else:
        result = subprocess.run(cmd, cwd=str(ROOT), input=stdin_input.encode(), capture_output=False)

    if check and result.returncode != 0:
        msg = f"ERROR: command failed with exit code {result.returncode}"
        print(msg)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()
        sys.exit(result.returncode)
    return result.returncode

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner unico: descarga data -> pipeline -> retrain -> scanner"
    )
    parser.add_argument("--tour", choices=["atp", "wta", "davis", "both"],
                        default="both", help="Tour para generar picks")
    parser.add_argument("--days-ahead", type=int, default=1,
                        help="Ventana de partidos a incluir (default: 1)")
    parser.add_argument("--no-download", action="store_true",
                        help="Saltar descarga de data (usar cache)")
    parser.add_argument("--log", action="store_true",
                        help="Tambien escribe la salida a data/logs/daily_{fecha}.log")
    args = parser.parse_args()

    tours = ("atp", "wta", "davis") if args.tour == "both" else (args.tour,)

    log_file = None
    if args.log:
        log_path = log_path_for_today()
        log_file = open(log_path, "a", encoding="utf-8")
        print(f"Log: {log_path}")

    try:
        # Step 1: Download fresh data
        if not args.no_download:
            print("=" * 60)
            print("  PASO 1/4: Descargando data fresca")
            print("=" * 60)
            run([sys.executable, "scripts/download_data.py"], log_file=log_file)

        # Step 2: Run pipeline for each tour
        print("\n" + "=" * 60)
        print("  PASO 2/4: Corriendo pipeline de features")
        print("=" * 60)
        for tour in tours:
            run([sys.executable, "-m", "src.pipeline", tour], log_file=log_file)

        # Step 3: Retrain models — llamar funciones internas directamente para evitar input()
        print("\n" + "=" * 60)
        print("  PASO 3/4: Reentrenando modelos LR")
        print("=" * 60)
        for tour in tours:
            code = (
                f"from src.value_analysis import load_model; "
                f"load_model('{tour}', retrain=True); print('Modelo {tour.upper()} listo.')"
            )
            run([sys.executable, "-c", code], log_file=log_file)

        # Step 4: Run scanner
        print("\n" + "=" * 60)
        print("  PASO 4/4: Ejecutando scanner diario")
        print("=" * 60)
        cmd = [sys.executable, "-m", "src.daily_scanner",
               "--tour", args.tour,
               "--days-ahead", str(args.days_ahead),
               "--auto-save"]
        run(cmd, log_file=log_file)

        print("\n" + "=" * 60)
        print("  Pipeline de predicciones completo.")
        print("=" * 60)
    finally:
        if log_file is not None:
            log_file.close()

if __name__ == "__main__":
    main()
