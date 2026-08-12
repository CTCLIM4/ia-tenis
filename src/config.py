"""Carga variables de entorno desde .env antes de que el resto del proyecto
las lea vía os.environ (ODDS_API_KEY, etc).

Se resuelve la ruta del .env de forma explícita (raíz del repo) en vez de
depender del cwd desde el que se invoque el script, para que funcione igual
corriendo `python -m src.daily_scanner` desde cualquier directorio.

Importar este módulo por su efecto secundario, antes de leer os.environ:
    import src.config  # noqa: F401
"""
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = _ROOT / ".env"

load_dotenv(DOTENV_PATH)

# Below this sample size (elo.match_counts for a player), a prediction is
# considered too data-poor to trust regardless of the edge it produces —
# see src/calibration_audit.py's blocked_low_sample decision.
MIN_MATCHES_THRESHOLD = 25

# Edge above this (with a sufficient sample) is treated as too good to be
# true — likely stale odds or a matching error rather than real value — see
# src/calibration_audit.py's blocked_suspicious_edge decision.
MAX_SUSPICIOUS_EDGE = 0.10
