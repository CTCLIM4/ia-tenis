"""
Create an immutable dataset snapshot under data/snapshots/{id}/.

Copies each tour's current data/raw/{...} files and
data/processed/{tour}_features.csv into data/snapshots/{id}/, then writes
metadata.json (last_match_date, row counts, per-file sha256 hashes) — see
docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md.

Uso:
  python scripts/create_snapshot.py                   # id = hoy (YYYY-MM-DD)
  python scripts/create_snapshot.py --id 2026-07-24   # id explicito
  python scripts/create_snapshot.py --tours atp        # solo un tour

Requiere que data/processed/{tour}_features.csv ya exista para cada tour
solicitado (correr primero: python -m src.pipeline {tour}).
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.snapshots import create_snapshot, load_snapshot_metadata


def run(snapshot_id: str | None = None, tours: tuple[str, ...] = ("atp", "wta")) -> Path:
    snapshot_dir = create_snapshot(snapshot_id=snapshot_id, tours=tours)
    meta = load_snapshot_metadata(snapshot_dir.name)
    print(f"Snapshot creado: {snapshot_dir}")
    for tour, tour_meta in meta["tours"].items():
        print(f"  {tour.upper()}: {tour_meta['n_rows']:,} filas, "
              f"ultimo partido {tour_meta['last_match_date']}")
    return snapshot_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crear un snapshot inmutable de los datasets actuales (raw + procesado)"
    )
    parser.add_argument("--id", dest="snapshot_id", default=None,
                        help="Id del snapshot (default: fecha de hoy, YYYY-MM-DD)")
    parser.add_argument("--tours", nargs="+", default=["atp", "wta"],
                        choices=["atp", "wta"], help="Tours a incluir (default: ambos)")
    args = parser.parse_args()
    run(snapshot_id=args.snapshot_id, tours=tuple(args.tours))


if __name__ == "__main__":
    main()
