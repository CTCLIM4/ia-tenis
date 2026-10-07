# WTA source lag investigation — 2026-10-05

## Finding

The WTA downloader is using the current 2026 link published on
https://www.tennis-data.co.uk/data.php:

`https://www.tennis-data.co.uk/hrjk-85HytOjkhth76j_ygh4jf7/2026w/2026.xlsx`

On 2026-10-05 the remote file returned HTTP 200, 343,699 bytes, and
`Last-Modified: Sun, 27 Sep 2026 13:25:33 GMT`. Its SHA-256 matched the
downloaded `data/raw/tennis_wta_tduk/2026w.xlsx` exactly:

`9bb1d5b2ecc57fc1c0866e50f722b2a4ebc7616090e8848b41b08f438ce7c753`

The latest match date in that file and the rebuilt WTA model cache is
2026-09-27. WTA's own reporting confirms that Beijing matches were completed
after that date, including on 2026-10-05:
https://www.wtatennis.com/news/4587000/gauff-pulls-away-late-to-defeat-rising-sun-in-beijing

Tennis-Data says its current-season files are updated weekly at the end of
each tournament week. The source file itself is behind; the downloader did
not select an obsolete link or skip an updated file. There is no local code
fix that makes this source publish results earlier.

## Alternative assessed

https://www.valuebetennis.com/donnees.htm publishes a 2026 CSV under CC BY
4.0 and reported generation on 2026-10-05. A read-only inspection of its
2026 file found 108 WTA main-tour/Masters/Grand Slam rows dated 2026-09-28
through 2026-10-04, with the latest at 17:00 UTC on October 4.

It cannot be dropped into the existing WTA loader. The CSV is semicolon
delimited and uses full player names, French surface values, winner IDs, and
different tournament categories. The current tennis-data.co.uk files use
names such as `Birrell K.` and include per-match `WRank`/`LRank`; the
alternative has no per-match rankings. Before using it, an adapter would
need explicit player identity matching, tour/category filtering, surface
conversion, deduplication against the existing season, and a documented
policy for missing historical rank features. Validate overlaps and
walk-forward metrics before rebuilding the production WTA model.

No alternative rows were merged into the current data or model in this
investigation. The existing WTA staleness warning remains meaningful.

## Follow-up integration, 2026-10-05

The later integration added `src/data/wta_supplement.py` and a current-year
download to `scripts/download_data.py`. It keeps the Tennis-Data file intact
and stores the CC BY 4.0 CSV separately under the WTA raw directory. Only
settled WTA main-draw rows strictly after the primary file's latest match
date are eligible. The loader resolves full names to historical WTA keys,
carries forward a rank only when observed in the previous 35 days, and uses
an all-or-nothing 90% identity-coverage gate. Unknown or ambiguous names
are excluded; an invalid download leaves the prior supplement untouched.

On this date, 67 of 72 eligible Beijing main-draw matches were incorporated
(93% coverage); five matches involving four previously unresolvable names
were omitted. The latest WTA model match date moved to 2026-10-04. The
original 47,599 feature rows remained identical, including their order;
the new processed set has 47,666 rows. Under the stable same-day ordering
used for both comparison runs, 2026 walk-forward log-loss was 0.6013 on
the original 2,191 matches and 0.6029 after adding the 67 new matches.
These are different evaluation samples, so the small difference is not
evidence of a change in accuracy on the original matches.

## Refresh, 2026-10-06

Valuebetennis published a new current-year CSV generated on 2026-10-06.
The downloader refreshed the local copy. Against the still-current
tennis-data.co.uk primary file (latest match 2026-09-27), 81 settled WTA
main-draw matches were eligible and 75 were safely incorporated (92.6%
coverage). Six matches were omitted because their players do not have an
unambiguous historical WTA identity: Xinran Sun (three matches), Yushan
Shao, Yihan Qu, and Yu Jun Lin. This remains above the 90% coverage gate.

The WTA pipeline and model cache were rebuilt from 47,674 matches. The
latest match date in the cache is 2026-10-06. The 2026 walk-forward sample
has 2,266 matches, with 0.6589 accuracy and 0.6025 log-loss. This sample
contains eight matches more than the preceding run; do not compare its
aggregate metrics as if evaluated on identical rows.
