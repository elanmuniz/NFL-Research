"""
Render the current state of the models into a static HTML page for
GitHub Pages: data/processed/power_rankings_<CURRENT_SEASON>_wk*.csv (if it
exists yet) plus the evergreen turnover-margin/third-down results.

Self-contained (no external CSS/JS) so it works as a plain static file with
no build step. Run after power_rankings.py; see .github/workflows/update-rankings.yml
for the scheduled pipeline that keeps this current during the season.
"""
import glob
import html
import json
import os
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from power_rankings import CURRENT_SEASON  # noqa: E402

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


def find_latest_rankings():
    pattern = os.path.join(PROCESSED_DIR, f"power_rankings_{CURRENT_SEASON}_wk*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None, None
    def week_of(path):
        return int(path.rsplit("_wk", 1)[1].replace(".csv", ""))
    latest = max(matches, key=week_of)
    return pd.read_csv(latest), week_of(latest)


def rankings_table_html(df):
    head = "".join(f"<th>{label}</th>" for _, label, _ in RANK_DISPLAY_COLS)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col, _, fmt in RANK_DISPLAY_COLS:
            val = row[col]
            try:
                text = fmt.format(val)
            except (ValueError, TypeError):
                text = html.escape(str(val))
            css = ' class="team-cell"' if col == "team" else ""
            cells.append(f"<td{css}>{text}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def load_summary():
    path = os.path.join(PROCESSED_DIR, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_weights():
    path = os.path.join(PROCESSED_DIR, "power_ranking_weights.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def rankings_section():
    df, week = find_latest_rankings()
    if df is None:
        return f"""
        <section class="card">
          <h2>{CURRENT_SEASON} Power Rankings</h2>
          <p class="muted">
            The {CURRENT_SEASON} season hasn't produced any completed games yet
            (or this page was built before <code>scripts/fetch_data.py</code> picked up
            the first week's data). Rankings will appear here automatically once games
            are played &mdash; see the weekly update workflow in the repo.
          </p>
        </section>
        """
    df = df.sort_values("rank")
    return f"""
    <section class="card">
      <h2>{CURRENT_SEASON} Power Rankings <span class="muted">&mdash; through week {week}</span></h2>
      <div class="table-wrap">{rankings_table_html(df)}</div>
    </section>
    """


def weights_section():
    w = load_weights()
    if w is None:
        return ""
    rows = sorted(w["weights"].items(), key=lambda kv: -abs(kv[1]))
    trs = "".join(
        f"<tr><td>{feat}</td><td>{val:+.3f}</td>"
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


def turnover_section():
    s = load_summary()
    if s is None:
        return ""
    overall = {row["bucket"]: row for row in s["overall"]}
    won = overall.get("won_margin", {})
    lost = overall.get("lost_margin", {})
    return f"""
    <section class="card">
      <h2>Background: Turnover Margin vs. Win %</h2>
      <p class="muted">2023-2025, all games.</p>
      <div class="stat-row">
        <div class="stat"><div class="stat-value">{won.get('win_pct', 0):.1f}%</div>
          <div class="stat-label">Win % when winning the turnover battle</div></div>
        <div class="stat"><div class="stat-value">{lost.get('win_pct', 0):.1f}%</div>
          <div class="stat-label">Win % when losing the turnover battle</div></div>
      </div>
    </section>
    """


def build_page():
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Power Rankings {CURRENT_SEASON}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #0b0d12;
    --card-bg: #151822;
    --text: #e8eaf0;
    --muted: #9aa1b2;
    --accent: #4f8cff;
    --border: #262b38;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #f4f5f8;
      --card-bg: #ffffff;
      --text: #16181d;
      --muted: #5b6472;
      --accent: #2b62d9;
      --border: #e3e6ec;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  header {{ padding: 2rem 1.25rem 1rem; max-width: 1000px; margin: 0 auto; }}
  header h1 {{ margin: 0 0 0.25rem; font-size: 1.6rem; }}
  header p {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
  main {{ max-width: 1000px; margin: 0 auto; padding: 0 1.25rem 3rem; }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
  }}
  .card h2 {{ margin-top: 0; font-size: 1.15rem; }}
  .muted {{ color: var(--muted); font-size: 0.85rem; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; white-space: nowrap; }}
  th, td {{ padding: 0.5rem 0.7rem; text-align: right; border-bottom: 1px solid var(--border); }}
  th:first-child, td:first-child {{ text-align: left; }}
  td.team-cell {{ text-align: left; font-weight: 600; }}
  thead th {{ color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
  tbody tr:hover {{ background: rgba(79, 140, 255, 0.08); }}
  .stat-row {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .stat-label {{ color: var(--muted); font-size: 0.85rem; max-width: 16rem; }}
  footer {{ max-width: 1000px; margin: 0 auto; padding: 0 1.25rem 2rem; color: var(--muted); font-size: 0.8rem; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>NFL Power Rankings &mdash; {CURRENT_SEASON}</h1>
  <p>Ranked on turnover margin, passer rating differential, yards/play &amp; success rate,
     red zone efficiency, and an opponent-adjusted (DVOA-style) EPA rating.</p>
</header>
<main>
{rankings_section()}
{weights_section()}
{turnover_section()}
</main>
<footer>
  Generated {generated_at} &middot; data from
  <a href="https://github.com/nflverse/nflverse-data">nflverse-data</a> &middot;
  <a href="https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'elanmuniz/NFL-Research')}">source</a>
</footer>
</body>
</html>
"""


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    out_path = os.path.join(DOCS_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(build_page())
    # Tell GitHub Pages not to run this through Jekyll.
    open(os.path.join(DOCS_DIR, ".nojekyll"), "w").close()
    print(f"Wrote {out_path}")
