"""
Render the current state of the models into static HTML pages for
GitHub Pages:
  - docs/index.html: the live CURRENT_SEASON power rankings (if the season
    has produced any completed games yet).
  - docs/season-2025.html: the final, full-season 2025 power rankings
    (a completed season, so this page is static once generated).
  - docs/metrics.html: what each metric means, why it matters, how it's
    computed, and the trained model weights.

Self-contained (no external CSS/JS) so each page works as a plain static
file with no build step. Run after power_rankings.py; see
.github/workflows/update-rankings.yml for the scheduled pipeline that keeps
docs/index.html current during the season.
"""
import glob
import html
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

sys.path.insert(0, os.path.dirname(__file__))
from power_rankings import CURRENT_SEASON  # noqa: E402

FINAL_SEASON = 2025  # last fully completed season, its own static recap page

RANK_DISPLAY_COLS = [
    ("rank", "Rank", "{:d}"),
    ("team", "Team", "{}"),
    ("games_played", "GP", "{:d}"),
    ("win_pct", "Win %", "{:.1%}"),
    ("power_score", "Power Score", "{:.2f}"),
    ("turnover_margin_pg", "TO Margin/G", "{:+.2f}"),
    ("passer_rating_diff", "Passer Rtg Diff", "{:+.1f}"),
    ("ypp_diff", "YPP Diff", "{:+.2f}"),
    ("success_rate_diff", "Success Rate Diff", "{:+.1%}"),
    ("redzone_td_pct_diff", "Red Zone TD% Diff", "{:+.1%}"),
    ("adj_epa_rating", "Adj. EPA Rating", "{:+.3f}"),
]

COLUMN_TOOLTIPS = {
    "rank": "Position in the Power Score ranking, 1 = best.",
    "team": "NFL team abbreviation.",
    "games_played": "Games played this season (regular season only).",
    "win_pct": "Percentage of games won this season.",
    "power_score": "Composite ranking score: the six metrics to the right, "
                   "z-scored and combined using weights trained on 2023-2025 "
                   "team-season point margin.",
    "turnover_margin_pg": "Takeaways minus giveaways, averaged per game.",
    "passer_rating_diff": "The team's own passer rating minus the passer "
                          "rating it allows on defense.",
    "ypp_diff": "Yards gained per offensive play minus yards allowed per "
                "defensive play.",
    "success_rate_diff": "Share of offensive plays gaining a meaningful "
                         "fraction of yards-to-go, minus the same rate "
                         "allowed on defense.",
    "redzone_td_pct_diff": "Touchdown % on offensive red zone trips minus "
                           "touchdown % allowed on defense.",
    "adj_epa_rating": "Opponent-adjusted EPA/play rating — a DVOA-style "
                      "proxy, not licensed DVOA data.",
}

PAGES = [
    ("index.html", "Live Rankings"),
    ("season-2025.html", "2025 Final Rankings"),
    ("metrics.html", "Metrics Explained"),
]

METRICS = [
    {
        "name": "Turnover Margin",
        "column": "turnover_margin_pg",
        "stat": "The difference between takeaways (interceptions and fumble "
                "recoveries) and giveaways (interceptions thrown and fumbles lost) "
                "in a game.",
        "why": "Winning the turnover battle drastically shifts win probability. "
               "Teams that finish a game with a positive turnover margin win a vast "
               "majority of their games at every level of football &mdash; in our own "
               "2023-2025 data, teams that won the turnover battle won 76% of their "
               "games, versus 24% for teams that lost it.",
        "how": "Giveaways = interceptions thrown + fumbles lost while on offense. "
               "Takeaways = the opponent's giveaways in that same game. We sum both "
               "across the season and report the per-game average margin "
               "(takeaways &minus; giveaways) &divide; games played.",
    },
    {
        "name": "Passer Rating Differential",
        "column": "passer_rating_diff",
        "stat": "The gap between a team's own passer rating (thrown by its "
                "offense) and the passer rating it allows opponents to throw "
                "against its defense.",
        "why": "Dominating this metric is one of the most reliable separators "
               "between winning and losing teams, because it captures both elite "
               "quarterback play and a strong pass defense in a single number.",
        "how": "Standard NFL passer rating (completion %, yards/attempt, TD%, and "
               "INT% each clipped to a 0&ndash;2.375 scale, combined and multiplied "
               "up to a 0&ndash;158.3 scale), computed once from season-cumulative "
               "attempts/completions/yards/TDs/INTs &mdash; both for the team's own "
               "offense and for what it allowed on defense &mdash; then subtracted.",
    },
    {
        "name": "Yards Per Play &amp; Success Rate",
        "column": "ypp_diff / success_rate_diff",
        "stat": "Efficiency measured per snap &mdash; yards gained per play, and the "
                "share of plays that count as \"successful\" (gaining a meaningful "
                "fraction of the yards needed for a first down) &mdash; rather than "
                "cumulative totals.",
        "why": "Raw yardage totals can be misleading: garbage-time drives in a "
               "blowout inflate them without reflecting real game control. "
               "Efficiency per snap, and converting a high share of downs, is what "
               "actually dictates who controls a game.",
        "how": "For every rush/pass snap (excluding kneels, spikes, and special "
               "teams), we sum yards gained and nflfastR's play-level success flag "
               "across the season for both sides of the ball, divide by play count, "
               "and subtract what the defense allowed from what the offense gained.",
    },
    {
        "name": "Red Zone Efficiency",
        "column": "redzone_td_pct_diff",
        "stat": "Touchdown conversion percentage on drives that reach inside the "
                "opponent's 20-yard line, for both offense and defense.",
        "why": "Moving the ball down the field is worth little if it results in "
               "field goals instead of touchdowns. Winning teams punch the ball in "
               "once they get close, and keep opponents out of the end zone when "
               "the defense is backed up.",
        "how": "Every offensive drive that ever crosses the 20 counts as a red zone "
               "trip; it counts as converted only if that drive's outcome is a "
               "touchdown. We use season totals (total TDs &divide; total trips), "
               "never an average of per-game percentages, so one small-sample game "
               "(e.g. 1-for-1) can't swing the number.",
    },
    {
        "name": "Adjusted EPA Rating",
        "column": "adj_epa_rating",
        "stat": "An opponent-adjusted efficiency rating &mdash; this project's "
                "stand-in for DVOA (Defense-adjusted Value Over Average).",
        "why": "A metric that reliably correlates with sustained, multi-year "
               "winning records has to filter out schedule strength: a team that "
               "padded its stats against bad defenses shouldn't rank the same as "
               "one that earned them against good ones.",
        "how": "<strong>Caveat:</strong> real DVOA (Football Outsiders/FTN) is a "
               "proprietary, licensed metric that weights every play by down, "
               "distance, score, and time remaining, using a non-public "
               "methodology &mdash; it isn't available in any free dataset, so this "
               "project can't reproduce it exactly. Instead we compute each team's "
               "raw EPA (Expected Points Added) per play on offense and defense, "
               "then run each team's per-game net EPA margin through the "
               "<strong>Simple Rating System</strong> (SRS) &mdash; the same "
               "least-squares opponent-adjustment idea DVOA is built on, just "
               "without the situational weighting. Read <code>adj_epa_rating</code> "
               "as directionally DVOA-like, not the real thing.",
    },
]


def find_rankings(season):
    """Latest (by week number) rankings CSV on disk for a given season."""
    pattern = os.path.join(PROCESSED_DIR, f"power_rankings_{season}_wk*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None, None
    def week_of(path):
        return int(path.rsplit("_wk", 1)[1].replace(".csv", ""))
    latest = max(matches, key=week_of)
    return pd.read_csv(latest), week_of(latest)


# Rank and Team are frozen in place (position: sticky) as the table scrolls
# laterally -- these CSS classes give them a fixed left offset (one after
# the other) so they stack correctly. Widths must stay in sync with the
# matching `width` rules in STYLE below.
STICKY_COL_CLASS = {"rank": "sticky-col sticky-col-1", "team": "sticky-col sticky-col-2"}


# Columns sorted as text (alphabetical); everything else in RANK_DISPLAY_COLS
# sorts numerically on its underlying (unformatted) value.
TEXT_SORT_COLS = {"team"}

# These six columns are signed differentials (team's own number minus what
# it allows/faces) -- colored green/positive or red/negative in the table,
# reinforcing the +/- sign already in the formatted text rather than being
# the only signal.
SIGNED_DIFF_COLS = {
    "turnover_margin_pg", "passer_rating_diff", "ypp_diff",
    "success_rate_diff", "redzone_td_pct_diff", "adj_epa_rating",
}


def diff_class(val):
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "diff-zero"
    if num > 0:
        return "diff-pos"
    if num < 0:
        return "diff-neg"
    return "diff-zero"


def rankings_table_html(df):
    head_cells = []
    for col, label, _ in RANK_DISPLAY_COLS:
        classes = [STICKY_COL_CLASS[col]] if col in STICKY_COL_CLASS else []
        classes.append("sortable")
        sort_type = "str" if col in TEXT_SORT_COLS else "num"
        tip = COLUMN_TOOLTIPS.get(col)
        cls_attr = f' class="{" ".join(classes)}"'
        data_attrs = f' data-col="{col}" data-type="{sort_type}"'
        label_html = f'<span class="has-tip">{label}</span>' if tip else label
        title_attr = f' title="{html.escape(tip)}"' if tip else ""
        head_cells.append(
            f'<th{cls_attr}{data_attrs}{title_attr}>'
            f'<span class="th-inner">{label_html}<span class="sort-arrow"></span></span></th>'
        )
    head = "".join(head_cells)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col, _, fmt in RANK_DISPLAY_COLS:
            val = row[col]
            try:
                text = fmt.format(val)
            except (ValueError, TypeError):
                text = html.escape(str(val))
            classes = [STICKY_COL_CLASS[col]] if col in STICKY_COL_CLASS else []
            if col == "team":
                classes.append("team-cell")
            if col in SIGNED_DIFF_COLS:
                classes.append(diff_class(val))
            cls_attr = f' class="{" ".join(classes)}"' if classes else ""
            sort_val = html.escape(str(val))
            cells.append(f'<td{cls_attr} data-sort="{sort_val}">{text}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<table class="sortable-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


TABLE_HINT_HTML = (
    '<p class="muted table-hint">Hover over a column header for a quick explanation, '
    'or see <a href="metrics.html">Metrics Explained</a> for full detail.</p>'
)


def load_weights():
    path = os.path.join(PROCESSED_DIR, "power_ranking_weights.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def live_rankings_section():
    df, week = find_rankings(CURRENT_SEASON)
    if df is None:
        return f"""
        <section class="card">
          <h2>{CURRENT_SEASON} Power Rankings</h2>
          <p class="muted">
            The {CURRENT_SEASON} season hasn't produced any completed games yet.
            Rankings will appear here automatically once games are played.
          </p>
        </section>
        """
    df = df.sort_values("rank")
    return f"""
    <section class="card">
      <h2>{CURRENT_SEASON} Power Rankings <span class="muted">&mdash; through week {week}</span></h2>
      <div class="table-wrap">{rankings_table_html(df)}</div>
      {TABLE_HINT_HTML}
    </section>
    """


def final_season_section(season):
    df, week = find_rankings(season)
    if df is None:
        return f"""
        <section class="card">
          <h2>{season} Final Power Rankings</h2>
          <p class="muted">No rankings file found for {season} yet.</p>
        </section>
        """
    df = df.sort_values("rank")
    games = int(df["games_played"].max())
    return f"""
    <section class="card">
      <h2>{season} Season &mdash; Final Power Rankings</h2>
      <p class="muted">
        Full {games}-game regular season (through week {week}). Same model as the live
        {CURRENT_SEASON} rankings, weights trained on 2023-2025 team-seasons.
      </p>
      <div class="table-wrap">{rankings_table_html(df)}</div>
      {TABLE_HINT_HTML}
    </section>
    """


def weights_section():
    w = load_weights()
    if w is None:
        return ""
    rows = sorted(w["weights"].items(), key=lambda kv: -abs(kv[1]))
    trs = "".join(
        f'<tr><td>{feat}</td><td class="{diff_class(val)}">{val:+.3f}</td>'
        f"<td>{w['univariate_r_with_point_margin'][feat]:.2f}</td></tr>"
        for feat, val in rows
    )
    seasons = ", ".join(str(s) for s in w["train_seasons"])
    return f"""
    <section class="card">
      <h2>Model Weights</h2>
      <p class="muted">
        Trained on {w['n_team_seasons']} team-seasons ({seasons}), R&sup2; = {w['r_squared']:.3f}
        predicting average point margin per game.
      </p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Metric</th><th>Joint weight (standardized)</th><th>Standalone correlation</th></tr></thead>
          <tbody>{trs}</tbody>
        </table>
      </div>
    </section>
    """


def nav_html(active_file):
    links = "".join(
        f'<a class="nav-link{" active" if fname == active_file else ""}" href="{fname}">{label}</a>'
        for fname, label in PAGES
    )
    return f'<nav class="top-nav">{links}</nav>'


STYLE = """
  :root {
    color-scheme: light dark;
    --bg: #0d0d0d;
    --card-bg: #1a1a19;
    --text: #ffffff;
    --muted: #c3c2b7;
    --accent: #3987e5;
    --border: rgba(255, 255, 255, 0.10);
    --hover-bg: rgba(57, 135, 229, 0.08);
    --positive: #0ca30c;
    --negative: #e66767;
    --stripe-bg: rgba(255, 255, 255, 0.03);
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .35);
    --metric-1: #3987e5; --metric-2: #d95926; --metric-3: #199e70;
    --metric-4: #c98500; --metric-5: #d55181;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f9f9f7;
      --card-bg: #fcfcfb;
      --text: #0b0b0b;
      --muted: #52514e;
      --accent: #2a78d6;
      --border: rgba(11, 11, 11, 0.10);
      --hover-bg: rgba(42, 120, 214, 0.06);
      --positive: #006300;
      --negative: #c22a2a;
      --stripe-bg: rgba(11, 11, 11, 0.02);
      --shadow: 0 1px 2px rgba(11, 11, 11, .06), 0 8px 20px rgba(11, 11, 11, .06);
      --metric-1: #2a78d6; --metric-2: #eb6834; --metric-3: #1baf7a;
      --metric-4: #eda100; --metric-5: #e87ba4;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header { position: relative; padding: 2rem 1.25rem 1.25rem; max-width: 1000px; margin: 0 auto; }
  header::after {
    content: ""; position: absolute; left: 1.25rem; right: 1.25rem; bottom: 0; height: 3px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--metric-1), var(--metric-3), var(--metric-5));
  }
  header h1 { margin: 0 0 0.25rem; font-size: 1.6rem; }
  header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  .top-nav { display: flex; gap: 0.4rem; margin-top: 1.1rem; }
  .nav-link {
    display: inline-block; color: var(--muted); text-decoration: none; font-size: 0.85rem;
    font-weight: 600; padding: 0.4rem 0.9rem; border-radius: 999px;
    transition: background 0.15s, color 0.15s;
  }
  .nav-link.active { color: #fff; background: var(--accent); }
  .nav-link:not(.active):hover { color: var(--text); background: var(--hover-bg); }
  main { max-width: 1000px; margin: 0 auto; padding: 0 1.25rem 3rem; }
  .card {
    background: var(--card-bg); border: 1px solid var(--border); box-shadow: var(--shadow);
    border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
  }
  .card h2 { margin-top: 0; font-size: 1.15rem; }
  .muted { color: var(--muted); font-size: 0.85rem; }
  /* overflow: auto (not overflow-x only) makes this a real 2-axis scroll
     container with a bounded height, so the sticky header (top) and
     sticky Rank/Team columns (left) below both stick relative to THIS
     box's own scrollport, consistently, instead of the page's. */
  .table-wrap { overflow: auto; max-height: 70vh; }
  /* border-collapse: separate (not collapse) -- collapsed borders and
     position: sticky don't composite reliably together in every browser
     (most notably Safari): the sticky cell's background can fail to
     paint opaque, letting scrolled-under text show through/overlap. We
     only ever set border-bottom (no vertical borders), so separate +
     zero spacing renders identically while avoiding that whole bug class. */
  table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 0.85rem; white-space: nowrap; }
  th, td { padding: 0.5rem 0.7rem; text-align: right; border-bottom: 1px solid var(--border); }
  th:first-child, td:first-child { text-align: left; }
  td.team-cell { text-align: left; font-weight: 600; }
  .diff-pos { color: var(--positive); }
  .diff-neg { color: var(--negative); }
  .diff-zero { color: var(--muted); }
  thead th {
    position: sticky; top: 0; z-index: 2;
    background: var(--card-bg); background-clip: padding-box; color: var(--muted);
    font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
  }
  thead th .has-tip { cursor: help; border-bottom: 1px dotted var(--muted); padding-bottom: 1px; }
  thead th .has-tip:hover { color: var(--accent); border-bottom-color: var(--accent); }
  thead th.sortable { cursor: pointer; user-select: none; }
  thead th.sortable:hover { color: var(--accent); }
  th .th-inner { display: inline-flex; align-items: center; gap: 0.3rem; }
  th .sort-arrow { font-size: 0.6rem; color: var(--muted); width: 0.7em; display: inline-block; }
  th .sort-arrow::before { content: ""; }
  th.sort-asc .sort-arrow::before { content: "\\25B2"; color: var(--accent); }
  th.sort-desc .sort-arrow::before { content: "\\25BC"; color: var(--accent); }
  /* Zebra striping. Sticky cells carry their own opaque background (needed
     to mask content scrolling underneath), so an even row's stripe must be
     re-applied to its sticky cells explicitly or a seam appears once the
     table is scrolled horizontally. Hover is declared after, so it always
     wins over the stripe on the same row. */
  tbody tr:nth-child(even) { background: var(--stripe-bg); }
  tbody tr:nth-child(even) .sticky-col { background: var(--stripe-bg); }
  tbody tr:hover { background: var(--hover-bg); }
  tbody tr:hover .sticky-col { background: var(--hover-bg); }
  .table-hint { margin: 0.6rem 0 0; }

  /* Rank and Team stay put as the table scrolls sideways. Widths are fixed
     so the second column's left offset is predictable. will-change
     promotes each sticky cell to its own GPU layer -- confirmed (user
     report) needed specifically for Safari on macOS: Safari has a
     long-documented repaint bug where a horizontally-sticky table cell's
     compositor tile doesn't get correctly invalidated as the table
     scrolls sideways, leaving stale pre-scroll content visible under the
     new content painting in from underneath. This is the standard fix
     for that exact bug. Scoped to .sticky-col only (covers both body
     cells and the two sticky header corner cells via the shared class) --
     not applied to thead th generally, since that bug is specific to
     horizontal stickiness, not the header's separate vertical stickiness. */
  .sticky-col { position: sticky; z-index: 1; will-change: transform; background: var(--card-bg); background-clip: padding-box; }
  .sticky-col-1 { left: 0; width: 3.25rem; min-width: 3.25rem; }
  .sticky-col-2 { left: 3.25rem; width: 4.5rem; min-width: 4.5rem; }
  .sticky-col-2 { box-shadow: 2px 0 4px -2px rgba(0, 0, 0, 0.15); }
  thead th.sticky-col { z-index: 3; }
  .metric-label {
    color: var(--accent); font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; margin: 1rem 0 0.15rem;
  }
  .metric-label:first-of-type { margin-top: 0.25rem; }
  .card p:not(.metric-label):not(.muted) { margin: 0 0 0.5rem; line-height: 1.5; }
  /* The border stripe is decorative (fine at lower contrast); the label
     TEXT stays in the uniform --accent color rather than --metric-N --
     three of the five metric hues (yellow/aqua/magenta) measure well
     under WCAG's 4.5:1 text-contrast minimum against the light card
     surface (as low as 2.1:1), so recoloring the text itself would make
     those cards' labels hard to read in light mode. */
  .metric-card-1 { border-left: 4px solid var(--metric-1); }
  .metric-card-2 { border-left: 4px solid var(--metric-2); }
  .metric-card-3 { border-left: 4px solid var(--metric-3); }
  .metric-card-4 { border-left: 4px solid var(--metric-4); }
  .metric-card-5 { border-left: 4px solid var(--metric-5); }
  footer { max-width: 1000px; margin: 0 auto; padding: 0 1.25rem 2rem; color: var(--muted); font-size: 0.8rem; }
  a { color: var(--accent); }
"""


# Click a column header to sort the rankings table by that metric; click
# again to flip direction. Numeric columns default to descending (best
# value first, matching "rank 1 = best"); the Team column defaults to
# ascending (A-Z). Sorting reorders <tr> elements in place -- the sticky
# columns/header and hover styling are plain CSS on those same elements,
# so they keep working after a re-sort with no extra code.
SORT_SCRIPT = """
document.querySelectorAll('table.sortable-table').forEach(function (table) {
  var thead = table.tHead;
  var tbody = table.tBodies[0];
  if (!thead || !tbody) return;

  thead.querySelectorAll('th.sortable').forEach(function (th) {
    th.addEventListener('click', function () {
      var idx = th.cellIndex;
      var type = th.dataset.type;
      var wasAsc = th.classList.contains('sort-asc');
      var wasDesc = th.classList.contains('sort-desc');
      // First click: text and "rank" (where lower is better) start ascending;
      // every other numeric metric (where higher is better) starts descending.
      var ascFirst = type === 'str' || th.dataset.col === 'rank';
      var dir = wasAsc ? 'desc' : wasDesc ? 'asc' : (ascFirst ? 'asc' : 'desc');

      thead.querySelectorAll('th').forEach(function (h) {
        h.classList.remove('sort-asc', 'sort-desc');
      });
      th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');

      var rows = Array.prototype.slice.call(tbody.rows);
      rows.sort(function (a, b) {
        var av = a.cells[idx].dataset.sort;
        var bv = b.cells[idx].dataset.sort;
        var cmp;
        if (type === 'str') {
          cmp = av.localeCompare(bv);
        } else {
          cmp = parseFloat(av) - parseFloat(bv);
        }
        return dir === 'asc' ? cmp : -cmp;
      });
      rows.forEach(function (row) { tbody.appendChild(row); });
    });
  });
});
"""


def page_shell(active_file, title, subtitle, body_html):
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{STYLE}</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  {nav_html(active_file)}
</header>
<main>
{body_html}
</main>
<footer>
  Generated {generated_at} &middot; data from
  <a href="https://github.com/nflverse/nflverse-data">nflverse-data</a> &middot;
  <a href="https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'elanmuniz/NFL-Research')}">source</a>
</footer>
<script>{SORT_SCRIPT}</script>
</body>
</html>
"""


def metrics_section():
    cards = []
    for i, m in enumerate(METRICS, start=1):
        cards.append(f"""
        <section class="card metric-card-{i}">
          <h2>{m['name']}</h2>
          <p class="metric-label">The stat</p>
          <p>{m['stat']}</p>
          <p class="metric-label">Why it matters</p>
          <p>{m['why']}</p>
          <p class="metric-label">How we compute it</p>
          <p>{m['how']}</p>
        </section>
        """)
    return "".join(cards)


def build_metrics_page():
    body = metrics_section() + weights_section()
    subtitle = ("What each metric in the Power Score model means, why it's a leading "
                "indicator of winning football, and exactly how it's computed from "
                "play-by-play data.")
    return page_shell("metrics.html", "Metrics Explained", subtitle, body)


def build_index_page():
    body = live_rankings_section()
    subtitle = (
        "Ranked on turnover margin, passer rating differential, yards/play &amp; success rate, "
        "red zone efficiency, and an opponent-adjusted (DVOA-style) EPA rating."
    )
    return page_shell("index.html", f"NFL Power Rankings &mdash; {CURRENT_SEASON}", subtitle, body)


def build_final_season_page(season):
    body = final_season_section(season)
    subtitle = f"Final regular-season Power Score standings for all 32 teams, {season}."
    return page_shell(f"season-{season}.html", f"{season} Season &mdash; Final Rankings", subtitle, body)


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)

    with open(os.path.join(DOCS_DIR, "index.html"), "w") as f:
        f.write(build_index_page())
    print(f"Wrote {DOCS_DIR}/index.html")

    with open(os.path.join(DOCS_DIR, f"season-{FINAL_SEASON}.html"), "w") as f:
        f.write(build_final_season_page(FINAL_SEASON))
    print(f"Wrote {DOCS_DIR}/season-{FINAL_SEASON}.html")

    with open(os.path.join(DOCS_DIR, "metrics.html"), "w") as f:
        f.write(build_metrics_page())
    print(f"Wrote {DOCS_DIR}/metrics.html")

    # Tell GitHub Pages not to run this through Jekyll.
    open(os.path.join(DOCS_DIR, ".nojekyll"), "w").close()
