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
