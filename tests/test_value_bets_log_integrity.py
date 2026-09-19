"""Data invariant on the real data/value_bets_log.csv (not a synthetic
fixture): no two rows for a still-pending match may share the same
identity (tour + tournament + player_a + player_b + match_date).

Prompted by 2026-09-19 production verification: two pre-existing rows for
WTA Guadalajara Open Stearns P. vs Jovic I. (odds 3.80 and 3.90) predated
the check_existing_log_entry() dedup guard added in 158a0e2 and were never
retroactively cleaned up. A resolved match legitimately keeps its one row
forever, but a pending duplicate means one run's stake/P&L would be
double-counted once the match settles."""
import csv
from collections import defaultdict

from src.value_analysis import LOG_PATH


def test_no_duplicate_pending_matches_in_real_log():
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending_keys = defaultdict(int)
    for row in rows:
        if row.get("result") in (None, "", "pending"):
            key = (
                row.get("tour"), row.get("tournament"),
                row.get("player_a"), row.get("player_b"), row.get("match_date"),
            )
            pending_keys[key] += 1

    dupes = {k: n for k, n in pending_keys.items() if n > 1}
    assert not dupes, f"duplicate pending match rows in {LOG_PATH}: {dupes}"
