"""Runner unico para producir predicciones en vivo.

Uso:
  python scripts/run_prediction.py              # ATP + WTA + Davis
  python scripts/run_prediction.py --tour wta    # solo WTA
  python scripts/run_prediction.py --days-ahead 2  # incluye proximos 2 dias

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run(cmd: list[str], check: bool = True, stdin_input: str = "\n") -> int:
    """Run a subprocess command and return exit code."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), input=stdin_input.encode(), capture_output=False)
    if check and result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
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
    args = parser.parse_args()

    tours = ("atp", "wta", "davis") if args.tour == "both" else (args.tour,)

    # Step 1: Download fresh data
    if not args.no_download:
        print("=" * 60)
        print("  PASO 1/4: Descargando data fresca")
        print("=" * 60)
        run([sys.executable, "scripts/download_data.py"])

    # Step 2: Run pipeline for each tour
    print("\n" + "=" * 60)
    print("  PASO 2/4: Corriendo pipeline de features")
    print("=" * 60)
    for tour in tours:
        run([sys.executable, "-m", "src.pipeline", tour])

    # Step 3: Retrain models — llamar funciones internas directamente para evitar input()
    print("\n" + "=" * 60)
    print("  PASO 3/4: Reentrenando modelos LR")
    print("=" * 60)
    for tour in tours:
        if tour == "davis":
            code = (
                "from src.value_analysis import load_model; "
                "load_model('davis', retrain=True); print('Modelo DAVIS listo.')"
            )
        elif tour == "wta":
            code = (
                "from src.value_analysis import load_model; "
                "load_model('wta', retrain=True); print('Modelo WTA listo.')"
            )
        else:
            code = (
                "from src.value_analysis import load_model; "
                "load_model('atp', retrain=True); print('Modelo ATP listo.')"
            )
        run([sys.executable, "-c", code])

    # Step 4: Run scanner
    print("\n" + "=" * 60)
    print("  PASO 4/4: Ejecutando scanner diario")
    print("=" * 60)
    cmd = [sys.executable, "-m", "src.daily_scanner",
           "--tour", args.tour,
           "--days-ahead", str(args.days_ahead),
           "--auto-save"]
    run(cmd)

    print("\n" + "=" * 60)
    print("  Pipeline de predicciones completo.")
    print("=" * 60)

if __name__ == "__main__":
    main()
