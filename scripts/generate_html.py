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


# Rank and Team are frozen in place as the table scrolls laterally. They're
# rendered as their own separate <table> (see rankings_table_html) rather
# than via position: sticky -- see the comment on .frozen-pane in STYLE for
# why a second physical table replaced five rounds of sticky-positioning
# fixes that never actually stopped Safari from painting other columns'
# text over them.
FROZEN_COLS = {"rank", "team"}


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


def _table_head_html(cols):
    cells = []
    for col, label, _ in cols:
        sort_type = "str" if col in TEXT_SORT_COLS else "num"
        tip = COLUMN_TOOLTIPS.get(col)
        data_attrs = f' data-col="{col}" data-type="{sort_type}"'
        label_html = f'<span class="has-tip">{label}</span>' if tip else label
        title_attr = f' title="{html.escape(tip)}"' if tip else ""
        inner = f'<span class="th-inner">{label_html}<span class="sort-arrow"></span></span>'
        cells.append(f'<th class="sortable"{data_attrs}{title_attr}>{inner}</th>')
    return f'<thead><tr>{"".join(cells)}</tr></thead>'


def _table_body_html(df, cols):
    rows = []
    for _, row in df.iterrows():
        # data-row-id links this row to its counterpart in the other pane's
        # table, so a sort triggered from either pane can reorder both --
        # see PANE_SYNC_SCRIPT/SORT_SCRIPT.
        row_id = html.escape(str(row["team"]))
        cells = []
        for col, _, fmt in cols:
            val = row[col]
            try:
                text = fmt.format(val)
            except (ValueError, TypeError):
                text = html.escape(str(val))
            classes = []
            if col == "team":
                classes.append("team-cell")
            if col in SIGNED_DIFF_COLS:
                classes.append(diff_class(val))
            cls_attr = f' class="{" ".join(classes)}"' if classes else ""
            sort_val = html.escape(str(val))
            cells.append(f'<td{cls_attr} data-sort="{sort_val}">{text}</td>')
        rows.append(f'<tr data-row-id="{row_id}">{"".join(cells)}</tr>')
    return f'<tbody>{"".join(rows)}</tbody>'


def rankings_table_html(df):
    frozen_cols = [c for c in RANK_DISPLAY_COLS if c[0] in FROZEN_COLS]
    scroll_cols = [c for c in RANK_DISPLAY_COLS if c[0] not in FROZEN_COLS]
    frozen_table = (
        f'<table class="sortable-table frozen-table">'
        f'{_table_head_html(frozen_cols)}{_table_body_html(df, frozen_cols)}</table>'
    )
    scroll_table = (
        f'<table class="sortable-table scroll-table">'
        f'{_table_head_html(scroll_cols)}{_table_body_html(df, scroll_cols)}</table>'
    )
    # Two physically separate tables, not one table with sticky-positioned
    # columns -- see the comment on .frozen-pane in STYLE for why. The
    # frozen pane (Rank/Team) never scrolls horizontally at all and has no
    # overlapping content to paint over it; only its vertical scroll
    # position is mirrored from the scroll pane, via PANE_SYNC_SCRIPT.
    return (
        f'<div class="rankings-wrap">'
        f'<div class="frozen-pane">{frozen_table}</div>'
        f'<div class="scroll-pane">{scroll_table}</div>'
        f'</div>'
    )


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
      {rankings_table_html(df)}
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
      {rankings_table_html(df)}
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
  /* Used only by the plain (non-rankings) weights table. */
  .table-wrap { overflow: auto; max-height: 70vh; }
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
  /* Hover declared after nth-child, so it wins over the stripe on the
     same row. Both rules match the frozen table's rows too. */
  tbody tr:nth-child(even) { background: var(--stripe-bg); }
  tbody tr:hover { background: var(--hover-bg); }
  .table-hint { margin: 0.6rem 0 0; }

  /* Rank/Team ("frozen") vs. the rest of the columns ("scroll"): rendered
     as two entirely separate <table> elements (see rankings_table_html),
     not one table with position: sticky columns. History: five rounds of
     fixes that kept the frozen columns overlapping the scrolling ones in
     the SAME table -- border-collapse, background-clip, will-change, a
     JS-driven transform replacing native sticky, then moving the paint
     surface onto a plain absolutely-positioned div -- each closed off one
     specific compositing/stacking theory and each still left other
     columns' text visibly painting over Rank/Team during horizontal
     scroll on macOS Safari, confirmed by repeated user testing. That
     pattern (every fix verifiably correct by geometry/opacity/z-index,
     still broken on the one browser that could never be tested directly
     in this environment) means the bug lived in overlapping content
     sharing one scroll/stacking context, not in any single property.

     Fix: don't overlap at all. Two independent tables side by side, only
     the frozen one's vertical scroll position mirrored from the other via
     JS (PANE_SYNC_SCRIPT) -- there is no shared stacking context left for
     a browser's table-rendering quirks to get wrong. */
  .rankings-wrap { display: flex; max-height: 70vh; }
  .frozen-pane {
    flex: 0 0 auto; overflow: hidden; max-height: 70vh;
    box-shadow: 2px 0 4px -2px rgba(0, 0, 0, 0.15);
  }
  .scroll-pane { flex: 1 1 auto; min-width: 0; overflow: auto; max-height: 70vh; }
  .frozen-table { width: auto; }
  .scroll-table { width: auto; min-width: 100%; }
  .frozen-table th:first-child, .frozen-table td:first-child {
    width: 3.25rem; min-width: 3.25rem; text-align: left;
  }
  .frozen-table th:last-child, .frozen-table td:last-child {
    width: 4.5rem; min-width: 4.5rem;
  }
  /* The generic th:first-child/td:first-child left-align rule (for the
     weights table's "Metric" column) would otherwise also catch the
     scroll table's first column (GP, numeric) since it's a first-child
     too -- override it back to the default right alignment. */
  .scroll-table th:first-child, .scroll-table td:first-child { text-align: right; }
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
# ascending (A-Z). The rankings pages now split Rank/Team into a separate
# "frozen" <table> from the rest of the columns' "scroll" <table> (see
# rankings_table_html/.rankings-wrap) -- so a sort click in either table
# has to reorder its own <tbody> AND the other table's <tbody> to match,
# using each <tr>'s data-row-id (team abbreviation) to find its
# counterpart row. Grouping by the shared .rankings-wrap/.table-wrap
# ancestor is what finds that counterpart; a wrap with only one table
# (e.g. the weights table) just does nothing extra.
SORT_SCRIPT = """
document.querySelectorAll('.rankings-wrap, .table-wrap').forEach(function (wrap) {
  var tables = Array.prototype.slice.call(wrap.querySelectorAll('table.sortable-table'));
  if (!tables.length) return;

  function reorder(table, rowIds) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var byId = {};
    Array.prototype.slice.call(tbody.rows).forEach(function (row) {
      byId[row.dataset.rowId] = row;
    });
    rowIds.forEach(function (id) {
      var row = byId[id];
      if (row) tbody.appendChild(row);
    });
  }

  tables.forEach(function (table) {
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

        tables.forEach(function (t) {
          t.tHead.querySelectorAll('th').forEach(function (h) {
            h.classList.remove('sort-asc', 'sort-desc');
          });
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

        var rowIds = rows.map(function (row) { return row.dataset.rowId; });
        tables.forEach(function (t) {
          if (t !== table) reorder(t, rowIds);
        });
      });
    });
  });
});
"""


# Mirrors the frozen pane's vertical scroll position to match the scroll
# pane's, since only the scroll pane is natively scrolled (by the user or
# a trackpad/wheel event) -- the frozen pane's own scrollTop is set here in
# response, not scrolled directly. See the comment on .rankings-wrap in
# STYLE for why Rank/Team live in a wholly separate table rather than as
# sticky-positioned columns of one table.
PANE_SYNC_SCRIPT = """
document.querySelectorAll('.rankings-wrap').forEach(function (wrap) {
  var frozen = wrap.querySelector('.frozen-pane');
  var scroll = wrap.querySelector('.scroll-pane');
  if (!frozen || !scroll) return;
  scroll.addEventListener('scroll', function () {
    frozen.scrollTop = scroll.scrollTop;
  }, { passive: true });
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
<script>{PANE_SYNC_SCRIPT}</script>
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
