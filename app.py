import os
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
from pathlib import Path

# =========================
# Moneycontrol CricPulse
# =========================
DEFAULT_ZIP_PATH = "T20I match data.zip"

EXCLUDE_WIDES_FROM_BALLS_FACED = True
COUNT_NOBALL_AS_BALL_FACED = True

# Minimum criteria
MIN_BALLS_BATTER_LEADERBOARD = 120   # batters: keep 120
MIN_BALLS_BOWLER_LISTS = 6           # bowlers/lists: reduce to 6

# Exclude these wicket kinds from being credited to the bowler
NON_BOWLER_WICKET_KINDS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}

TEAM_COLORS = {
    "India": "#1E5AA8",
    "Australia": "#F2C400",
    "England": "#1B3F8B",
    "New Zealand": "#111111",
    "South Africa": "#0B7A3B",
    "Pakistan": "#0A7A3E",
    "Sri Lanka": "#1A2B7B",
    "Bangladesh": "#0B7A3B",
    "Afghanistan": "#1460AA",
    "West Indies": "#7A0026",
    "Ireland": "#0B7A3B",
    "Scotland": "#1B3F8B",
    "Zimbabwe": "#0B7A3B",
    "Netherlands": "#F47C20",
    "Nepal": "#1E5AA8",
    "UAE": "#C8102E",
    "Namibia": "#0033A0",
    "Oman": "#C8102E",
    "USA": "#B22234",
    "Canada": "#C8102E",
}


# -------------------------
# Utils
# -------------------------
def _hash_color(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    return f"#{h}"

def team_color(team: str | None) -> str:
    if not team or not isinstance(team, str):
        return "#2C2C2C"
    return TEAM_COLORS.get(team, _hash_color(team))

def safe_text(x) -> str:
    return "" if x is None else str(x)

def pick_default(options: list[str], preferred: str) -> str:
    if preferred in options:
        return preferred
    return options[0] if options else ""

def get_primary_batting_team(df: pd.DataFrame, batter: str) -> str | None:
    s = df.loc[df["batter"] == batter, "batting_team"].dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()

def get_primary_bowling_team(df: pd.DataFrame, bowler: str) -> str | None:
    s = df.loc[df["bowler"] == bowler, "bowling_team"].dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()


# -------------------------
# Parse ZIP → deliveries
# -------------------------
@st.cache_data(show_spinner=False)
def parse_zip_to_deliveries(zip_path: str) -> pd.DataFrame:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        match_files = [
            n for n in zf.namelist()
            if n.startswith("T20I match data/") and n.endswith(".json") and "__MACOSX" not in n
        ]

        rows = []
        for fn in match_files:
            match = json.loads(zf.read(fn))
            match_id = os.path.splitext(os.path.basename(fn))[0]

            dates = match.get("info", {}).get("dates", [])
            match_date = str(dates[0]) if dates else None
            teams = match.get("info", {}).get("teams", [])

            for innings_idx, inn in enumerate(match.get("innings", []), start=1):
                batting_team = inn.get("team")
                bowling_team = None
                if teams and batting_team:
                    other = [t for t in teams if t != batting_team]
                    bowling_team = other[0] if other else None

                seq = 0
                for over in inn.get("overs", []):
                    over_no = over.get("over")  # 0-indexed
                    legal_in_over = 0

                    for d in over.get("deliveries", []):
                        seq += 1

                        batter = d.get("batter")
                        bowler = d.get("bowler")

                        runs = d.get("runs", {})
                        runs_batter = int(runs.get("batter", 0))
                        runs_total = int(runs.get("total", 0))

                        extras = d.get("extras", {})
                        is_wide = "wides" in extras
                        is_noball = "noballs" in extras

                        # legal ball index within over (0..5); wides do not advance
                        if not is_wide:
                            legal_in_over += 1
                            ball_in_over_legal_idx = legal_in_over - 1
                        else:
                            ball_in_over_legal_idx = -1

                        # balls faced logic
                        ball_faced_legal = True
                        if EXCLUDE_WIDES_FROM_BALLS_FACED and is_wide:
                            ball_faced_legal = False
                        if (not COUNT_NOBALL_AS_BALL_FACED) and is_noball:
                            ball_faced_legal = False

                        wicket_player_out = []
                        wicket_kind = []
                        if "wickets" in d and d["wickets"]:
                            for w in d["wickets"]:
                                wicket_player_out.append(w.get("player_out", ""))
                                k = w.get("kind", "")
                                wicket_kind.append(str(k).strip().lower() if k is not None else "")

                        rows.append({
                            "match_id": match_id,
                            "date": match_date,
                            "innings": innings_idx,
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "over": over_no,
                            "seq": seq,
                            "batter": batter,
                            "bowler": bowler,
                            "runs_batter": runs_batter,
                            "runs_total": runs_total,
                            "is_wide": bool(is_wide),
                            "is_noball": bool(is_noball),
                            "ball_faced": 1 if ball_faced_legal else 0,
                            "legal_ball_bowled": 0 if is_wide else 1,
                            "ball_in_over_legal_idx": ball_in_over_legal_idx,
                            "wicket_player_out": "|".join(wicket_player_out) if wicket_player_out else "",
                            "wicket_kind": "|".join(wicket_kind) if wicket_kind else "",
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No match JSON files found/parsed in the ZIP.")

    df = df.sort_values(["match_id", "innings", "seq"]).reset_index(drop=True)

    # Batter out?
    def batter_out(row) -> int:
        if not row["wicket_player_out"]:
            return 0
        return 1 if row["batter"] in row["wicket_player_out"].split("|") else 0

    df["batter_out"] = df.apply(batter_out, axis=1)

    # Bowler wicket? (exclude run-outs etc.)
    def is_bowler_wicket(wicket_kind_str: str) -> int:
        if not isinstance(wicket_kind_str, str) or not wicket_kind_str.strip():
            return 0
        kinds = [k.strip().lower() for k in wicket_kind_str.split("|") if k.strip()]
        for k in kinds:
            if k not in NON_BOWLER_WICKET_KINDS:
                return 1
        return 0

    df["bowler_wicket"] = df["wicket_kind"].apply(is_bowler_wicket)

    return df


# --------------------------------------------
# Batter starts windows (3 overs)
# --------------------------------------------
@st.cache_data(show_spinner=False)
def build_batter_windows_3overs(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    for (mid, inn), inn_df in df.groupby(["match_id", "innings"], sort=False):
        inn_df = inn_df.sort_values("seq").reset_index(drop=True)
        if inn_df.empty:
            continue

        over_last_idx = inn_df.groupby("over").apply(lambda x: x.index.max()).to_dict()
        max_over = int(inn_df["over"].max())
        last_idx_innings = int(inn_df.index.max())

        overs = inn_df["over"].to_numpy(dtype=int)
        end_idx_for_row = np.empty(len(inn_df), dtype=int)
        for i in range(len(inn_df)):
            end_over = overs[i] + 2
            if end_over <= max_over and end_over in over_last_idx:
                end_idx_for_row[i] = over_last_idx[end_over]
            else:
                end_idx_for_row[i] = last_idx_innings

        for batter, bdf in inn_df.groupby("batter", sort=False):
            bdf = bdf.sort_values("seq")
            bdf_start = bdf[(bdf["ball_faced"] == 1) & (bdf["ball_in_over_legal_idx"] >= 0)]
            if bdf_start.empty:
                continue

            idx_all = bdf.index.to_numpy(dtype=int)
            runs_vals = bdf["runs_batter"].to_numpy(dtype=float)
            balls_vals = bdf["ball_faced"].to_numpy(dtype=float)
            outs_vals = bdf["batter_out"].to_numpy(dtype=float)

            pref_runs = np.concatenate([[0.0], np.cumsum(runs_vals)])
            pref_balls = np.concatenate([[0.0], np.cumsum(balls_vals)])
            pref_outs = np.concatenate([[0.0], np.cumsum(outs_vals)])

            start_idx = bdf_start.index.to_numpy(dtype=int)
            start_pos = np.searchsorted(idx_all, start_idx, side="left")
            cutoff_idx = end_idx_for_row[start_idx]
            cutoff_pos = np.searchsorted(idx_all, cutoff_idx, side="right")

            win_runs = pref_runs[cutoff_pos] - pref_runs[start_pos]
            win_balls = pref_balls[cutoff_pos] - pref_balls[start_pos]
            win_outs = pref_outs[cutoff_pos] - pref_outs[start_pos]

            meta = inn_df.loc[start_idx, ["bowling_team", "over", "ball_in_over_legal_idx"]].copy()
            meta["match_id"] = mid
            meta["innings"] = inn
            meta["batter"] = batter
            meta["runs"] = win_runs
            meta["balls"] = win_balls
            meta["outs"] = win_outs
            out_rows.append(meta)

    moment = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if moment.empty:
        raise ValueError("Batter windows are empty (unexpected).")

    moment.rename(columns={"over": "start_over", "ball_in_over_legal_idx": "start_ball_in_over"}, inplace=True)

    agg = (
        moment.groupby(["batter", "bowling_team", "start_over", "start_ball_in_over"], as_index=False)
              .agg(starts=("match_id", "count"), runs=("runs", "sum"), balls=("balls", "sum"), outs=("outs", "sum"))
    )
    agg["sr"] = np.where(agg["balls"] > 0, agg["runs"] / agg["balls"] * 100, np.nan)
    agg["dismissal_pct_per_start"] = np.where(agg["starts"] > 0, agg["outs"] / agg["starts"] * 100, np.nan)
    return agg


# --------------------------------------------
# Bowler starts windows (3 overs)
# --------------------------------------------
@st.cache_data(show_spinner=False)
def build_bowler_windows_3overs(df: pd.DataFrame) -> pd.DataFrame:
    if "bowler" not in df.columns or df["bowler"].dropna().empty:
        return pd.DataFrame()

    out_rows = []

    for (mid, inn), inn_df in df.groupby(["match_id", "innings"], sort=False):
        inn_df = inn_df.sort_values("seq").reset_index(drop=True)
        if inn_df.empty:
            continue

        over_last_idx = inn_df.groupby("over").apply(lambda x: x.index.max()).to_dict()
        max_over = int(inn_df["over"].max())
        last_idx_innings = int(inn_df.index.max())

        overs = inn_df["over"].to_numpy(dtype=int)
        end_idx_for_row = np.empty(len(inn_df), dtype=int)
        for i in range(len(inn_df)):
            end_over = overs[i] + 2
            if end_over <= max_over and end_over in over_last_idx:
                end_idx_for_row[i] = over_last_idx[end_over]
            else:
                end_idx_for_row[i] = last_idx_innings

        for bowler, bdf in inn_df.groupby("bowler", sort=False):
            if not isinstance(bowler, str) or not bowler:
                continue

            bdf = bdf.sort_values("seq")
            bdf_start = bdf[(bdf["legal_ball_bowled"] == 1) & (bdf["ball_in_over_legal_idx"] >= 0)]
            if bdf_start.empty:
                continue

            idx_all = bdf.index.to_numpy(dtype=int)
            runs_vals = bdf["runs_total"].to_numpy(dtype=float)
            balls_vals = bdf["legal_ball_bowled"].to_numpy(dtype=float)
            wkts_vals = bdf["bowler_wicket"].to_numpy(dtype=float)

            pref_runs = np.concatenate([[0.0], np.cumsum(runs_vals)])
            pref_balls = np.concatenate([[0.0], np.cumsum(balls_vals)])
            pref_wkts = np.concatenate([[0.0], np.cumsum(wkts_vals)])

            start_idx = bdf_start.index.to_numpy(dtype=int)
            start_pos = np.searchsorted(idx_all, start_idx, side="left")
            cutoff_idx = end_idx_for_row[start_idx]
            cutoff_pos = np.searchsorted(idx_all, cutoff_idx, side="right")

            win_runs = pref_runs[cutoff_pos] - pref_runs[start_pos]
            win_balls = pref_balls[cutoff_pos] - pref_balls[start_pos]
            win_wkts = pref_wkts[cutoff_pos] - pref_wkts[start_pos]

            meta = inn_df.loc[start_idx, ["batting_team", "over", "ball_in_over_legal_idx"]].copy()
            meta["match_id"] = mid
            meta["innings"] = inn
            meta["bowler"] = bowler
            meta["runs_conceded"] = win_runs
            meta["legal_balls"] = win_balls
            meta["wkts"] = win_wkts
            out_rows.append(meta)

    moment = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if moment.empty:
        return pd.DataFrame()

    moment.rename(columns={"over": "start_over", "ball_in_over_legal_idx": "start_ball_in_over"}, inplace=True)

    agg = (
        moment.groupby(["bowler", "batting_team", "start_over", "start_ball_in_over"], as_index=False)
              .agg(starts=("match_id", "count"), runs_conceded=("runs_conceded", "sum"),
                   legal_balls=("legal_balls", "sum"), wkts=("wkts", "sum"))
    )

    agg["overs_bowled"] = agg["legal_balls"] / 6.0
    agg["econ"] = np.where(agg["overs_bowled"] > 0, agg["runs_conceded"] / agg["overs_bowled"], np.nan)
    agg["wicket_pct_per_start"] = np.where(agg["starts"] > 0, agg["wkts"] / agg["starts"] * 100, np.nan)

    totals = df.groupby("bowler", as_index=False).agg(total_wkts=("bowler_wicket", "sum"))
    agg = agg.merge(totals, on="bowler", how="left")
    agg["wickets_share_pct"] = np.where(agg["total_wkts"] > 0, agg["wkts"] / agg["total_wkts"] * 100, np.nan)
    return agg


# --------------------------------------------
# Slot helpers
# --------------------------------------------
def best_bowlers_vs_batter_in_slot(df: pd.DataFrame, batter: str, start_over0: int, start_ball0: int, min_balls: int) -> pd.DataFrame:
    if "bowler" not in df.columns or df["bowler"].dropna().empty:
        return pd.DataFrame()

    starts = df[
        (df["batter"] == batter) &
        (df["ball_faced"] == 1) &
        (df["over"] == start_over0) &
        (df["ball_in_over_legal_idx"] == start_ball0)
    ].copy()
    if starts.empty:
        return pd.DataFrame()

    rows = []
    end_over = start_over0 + 2

    for (mid, inn), gstarts in starts.groupby(["match_id", "innings"], sort=False):
        inn_df = df[(df["match_id"] == mid) & (df["innings"] == inn)].sort_values("seq")
        base_mask = (inn_df["over"] >= start_over0) & (inn_df["over"] <= end_over)

        for _, srow in gstarts.iterrows():
            start_seq = int(srow["seq"])
            win_df = inn_df[base_mask & (inn_df["seq"] >= start_seq)]
            sub = win_df[win_df["batter"] == batter]
            if sub.empty:
                continue
            grp = sub.groupby("bowler", as_index=False).agg(
                runs=("runs_total", "sum"),
                balls=("legal_ball_bowled", "sum"),
                wkts=("bowler_wicket", "sum"),
            )
            grp["starts"] = 1
            rows.append(grp)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True).groupby("bowler", as_index=False).agg(
        starts=("starts", "sum"),
        runs=("runs", "sum"),
        balls=("balls", "sum"),
        wkts=("wkts", "sum"),
    )
    out = out[out["bowler"].notna() & (out["bowler"] != "")]
    out["overs"] = out["balls"] / 6.0
    out["econ_vs_batter"] = np.where(out["overs"] > 0, out["runs"] / out["overs"], np.nan)
    out["wicket_pct_per_start"] = np.where(out["starts"] > 0, out["wkts"] / out["starts"] * 100, np.nan)

    out = out[out["balls"] >= min_balls]
    return out.sort_values(["wicket_pct_per_start", "econ_vs_batter"], ascending=[False, True])


def best_batters_vs_bowler_in_slot(df: pd.DataFrame, bowler: str, start_over0: int, start_ball0: int, min_balls: int) -> pd.DataFrame:
    if "bowler" not in df.columns or df["bowler"].dropna().empty:
        return pd.DataFrame()

    starts = df[
        (df["bowler"] == bowler) &
        (df["legal_ball_bowled"] == 1) &
        (df["over"] == start_over0) &
        (df["ball_in_over_legal_idx"] == start_ball0)
    ].copy()
    if starts.empty:
        return pd.DataFrame()

    rows = []
    end_over = start_over0 + 2

    for (mid, inn), gstarts in starts.groupby(["match_id", "innings"], sort=False):
        inn_df = df[(df["match_id"] == mid) & (df["innings"] == inn)].sort_values("seq")
        base_mask = (inn_df["over"] >= start_over0) & (inn_df["over"] <= end_over)

        for _, srow in gstarts.iterrows():
            start_seq = int(srow["seq"])
            win_df = inn_df[base_mask & (inn_df["seq"] >= start_seq)]
            sub = win_df[win_df["bowler"] == bowler]
            if sub.empty:
                continue
            grp = sub.groupby("batter", as_index=False).agg(
                runs=("runs_batter", "sum"),
                balls=("ball_faced", "sum"),
                outs=("batter_out", "sum"),
            )
            grp["starts"] = 1
            rows.append(grp)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True).groupby("batter", as_index=False).agg(
        starts=("starts", "sum"),
        runs=("runs", "sum"),
        balls=("balls", "sum"),
        outs=("outs", "sum"),
    )
    out = out[out["batter"].notna() & (out["batter"] != "")]
    out["sr_vs_bowler"] = np.where(out["balls"] > 0, out["runs"] / out["balls"] * 100, np.nan)
    out["dismissal_pct_per_start"] = np.where(out["starts"] > 0, out["outs"] / out["starts"] * 100, np.nan)

    out = out[out["balls"] >= min_balls]
    return out.sort_values(["sr_vs_bowler", "dismissal_pct_per_start"], ascending=[False, True])


def matchup_batter_vs_bowler(df: pd.DataFrame, batter: str, bowler: str, start_over0: int, start_ball0: int):
    if "bowler" not in df.columns or df["bowler"].dropna().empty:
        return {"ok": 0, "reason": "No bowler field present in dataset."}

    starts = df[
        (df["batter"] == batter) &
        (df["ball_faced"] == 1) &
        (df["over"] == start_over0) &
        (df["ball_in_over_legal_idx"] == start_ball0)
    ].copy()

    if starts.empty:
        return {"ok": 0, "reason": "No historical starts for this batter at this over/ball slot."}

    starts_n = 0
    bruns = bballs = bouts = 0.0
    cruns = cballs = cwkts = 0.0

    for (mid, inn), group_starts in starts.groupby(["match_id", "innings"], sort=False):
        inn_df = df[(df["match_id"] == mid) & (df["innings"] == inn)].sort_values("seq")
        end_over = start_over0 + 2
        base_mask = (inn_df["over"] >= start_over0) & (inn_df["over"] <= end_over)

        for _, srow in group_starts.iterrows():
            starts_n += 1
            start_seq = int(srow["seq"])
            win_df = inn_df[base_mask & (inn_df["seq"] >= start_seq)]

            sub = win_df[(win_df["batter"] == batter) & (win_df["bowler"] == bowler)]
            bruns += sub["runs_batter"].sum()
            bballs += sub["ball_faced"].sum()
            bouts += sub["batter_out"].sum()

            sub2 = win_df[(win_df["bowler"] == bowler) & (win_df["batter"] == batter)]
            cruns += sub2["runs_total"].sum()
            cballs += sub2["legal_ball_bowled"].sum()
            cwkts += sub2["bowler_wicket"].sum()

    sr = (bruns / bballs * 100) if bballs > 0 else np.nan
    dps = (bouts / starts_n * 100) if starts_n > 0 else np.nan
    econ = (cruns / (cballs / 6.0)) if cballs > 0 else np.nan
    wps = (cwkts / starts_n * 100) if starts_n > 0 else np.nan

    return {
        "ok": 1,
        "starts": starts_n,
        "batter_runs": bruns, "batter_balls": bballs, "batter_outs": bouts,
        "batter_sr": sr, "batter_dismissal_pct_per_start": dps,
        "bowler_runs_conceded": cruns, "bowler_legal_balls": cballs, "bowler_wkts": cwkts,
        "bowler_econ": econ, "bowler_wicket_pct_per_start": wps,
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Moneycontrol CricPulse", layout="wide")

MC_LOGO = """
<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="#1a4fa3"/>
      <stop offset="1" stop-color="#0b2d66"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="30" fill="url(#g)"/>
  <path d="M18 42V22h6l8 10 8-10h6v20h-6V31l-8 10-8-10v11z" fill="#fff"/>
</svg>
""".strip()

CRICKET_SVG = """
data:image/svg+xml;utf8,
<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>
<circle cx='32' cy='32' r='28' fill='%23b40000'/>
<path d='M20 18c6 8 6 20 0 28' fill='none' stroke='%23ffffff' stroke-width='2'/>
<path d='M44 18c-6 8-6 20 0 28' fill='none' stroke='%23ffffff' stroke-width='2'/>
<path d='M24 20l-3 3m6-1l-3 3m6-1l-3 3' stroke='%23ffffff' stroke-width='1.5'/>
<path d='M40 20l3 3m-6-1l3 3m-6-1l3 3' stroke='%23ffffff' stroke-width='1.5'/>
</svg>
""".strip()

@st.cache_data(show_spinner=True)
def load_all(zip_path: str):
    df_ = parse_zip_to_deliveries(zip_path)
    bat_ = build_batter_windows_3overs(df_)
    bowl_ = build_bowler_windows_3overs(df_)
    return df_, bat_, bowl_

df, batter_win, bowler_win = load_all(DEFAULT_ZIP_PATH)

all_batters = sorted([x for x in df["batter"].dropna().unique().tolist() if x])
has_bowler = ("bowler" in df.columns) and (not df["bowler"].dropna().empty)
all_bowlers = sorted([x for x in df["bowler"].dropna().unique().tolist() if x]) if has_bowler else []

# Defaults (your request)
default_batter = pick_default(all_batters, "RG Sharma")
default_bowler = pick_default(all_bowlers, "MA Starc")

if "batter_sel" not in st.session_state:
    st.session_state["batter_sel"] = default_batter
if "bowler_sel" not in st.session_state:
    st.session_state["bowler_sel"] = default_bowler
if "over_ui" not in st.session_state:
    st.session_state["over_ui"] = 2
if "ball_ui" not in st.session_state:
    st.session_state["ball_ui"] = 1  # UI: 1–6

# Precompute team colors based on current selections
batter_team_now = get_primary_batting_team(df, st.session_state["batter_sel"])
bowler_team_now = get_primary_bowling_team(df, st.session_state["bowler_sel"]) if has_bowler else None
B_COL = team_color(batter_team_now)
W_COL = team_color(bowler_team_now)
M_LEFT = B_COL
M_RIGHT = W_COL

# CSS: style the *actual column containers* (works for tables too)
st.markdown(
    """
<style>

/* ======================================================
   LIGHT MODE (default)
   ====================================================== */

/* --- App background gradient --- */
.stApp {
  background:
    radial-gradient(1200px 600px at 20% 10%, rgba(0, 126, 230, 0.22), rgba(255, 255, 255, 0) 60%),
    radial-gradient(900px 500px at 85% 20%, rgba(0, 180, 140, 0.18), rgba(255, 255, 255, 0) 55%),
    linear-gradient(180deg, #ffffff 0%, #f6f8fb 100%);
  background-attachment: fixed;
}

/* Make Streamlit containers transparent */
div[data-testid="stAppViewContainer"],
section.main,
section.main > div,
div[data-testid="stMainBlockContainer"] {
  background: transparent !important;
}

/* Sidebar */
section[data-testid="stSidebar"] > div {
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(6px);
}

/* Cards */
.mc-card {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.88);
  box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

/* KPIs */
.mc-kpi {
  font-size: 36px;
  font-weight: 800;
  line-height: 1;
}

/* Header */
.mc-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 6px 0 18px 0;
}
.mc-header h1 {
  font-size: 28px;
  margin: 0;
  line-height: 1.15;
}
.mc-subtitle {
  color: rgba(0,0,0,0.62);
  font-size: 14px;
}

/* ======================================================
   DARK MODE (automatic)
   ====================================================== */

@media (prefers-color-scheme: dark) {

  /* App background */
  .stApp {
    background:
      radial-gradient(1200px 600px at 20% 10%, rgba(56, 189, 248, 0.18), transparent 60%),
      radial-gradient(900px 500px at 85% 20%, rgba(34, 197, 94, 0.16), transparent 55%),
      linear-gradient(180deg, #0b1220 0%, #020617 100%);
  }

  /* Text */
  html, body, .stMarkdown, .stText, .stDataFrame {
    color: #e5e7eb;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] > div {
    background: rgba(2, 6, 23, 0.96);
    backdrop-filter: blur(6px);
  }

  /* Cards */
  .mc-card {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
  }

  /* DataFrames / tables */
  div[data-testid="stDataFrame"] {
    background: rgba(2, 6, 23, 0.9) !important;
  }

  /* Inputs */
  input, textarea, select {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
  }

  /* Slider */
  div[data-baseweb="slider"] {
    background: transparent;
  }

  /* Header subtitle */
  .mc-subtitle {
    color: rgba(229,231,235,0.65);
  }
}

/* Reduce top padding slightly */
div[data-testid="stMainBlockContainer"] {
  padding-top: 1.2rem;
}

</style>
""",
    unsafe_allow_html=True
)

# -------------------------
# Top bar: brand left, controls right
# -------------------------
top_left, top_right = st.columns([0.62, 0.38], vertical_alignment="top")
with top_left:
    # Moneycontrol logo + title
    logo_path = "moneycontrol-logo-vector.png"  # keep this file next to app.py
    c1, c2 = st.columns([1.2, 4.8], vertical_alignment="center")
    with c1:
        if Path(logo_path).exists():
            st.image(logo_path, use_container_width=True)
        else:
            # Fallback to inline SVG if the png is not found
            st.markdown(MC_LOGO, unsafe_allow_html=True)
    with c2:
        st.markdown(
            '''
            <div class="mc-header">
              <div>
                <h1>Moneycontrol CricPulse</h1>
                <div class="mc-subtitle">Cricket Intelligence Platform</div>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
with top_right:
    st.session_state["over_ui"] = st.slider(
        "Over (0–19)",
        min_value=0, max_value=19,
        value=int(st.session_state["over_ui"]),
        step=1
    )
    st.session_state["ball_ui"] = st.radio(
        "Ball in over (1–6, legal)",
        options=[1, 2, 3, 4, 5, 6],
        index=[1, 2, 3, 4, 5, 6].index(int(st.session_state["ball_ui"])),
        horizontal=True
    )

over_ui = int(st.session_state["over_ui"])
ball_ui_1to6 = int(st.session_state["ball_ui"])
start_over0 = over_ui
start_ball0 = ball_ui_1to6 - 1  # internal 0–5

# -------------------------
# Main split: Batter left, Bowler right
# -------------------------
st.markdown('<div id="main_anchor"></div>', unsafe_allow_html=True)
left, right = st.columns(2, vertical_alignment="top")

# ---- Batter (LEFT) ----
with left:
    st.session_state["batter_sel"] = st.selectbox(
        "Batter",
        options=all_batters,
        index=all_batters.index(st.session_state["batter_sel"]) if st.session_state["batter_sel"] in all_batters else 0,
        key="batter_sel_widget"
    )

    batter_team = get_primary_batting_team(df, st.session_state["batter_sel"])

    st.markdown(
        f"""
<div class="mc-card">
  <h3 style="margin:0;">{safe_text(st.session_state["batter_sel"])}</h3>
  <div class="mc-meta"><b>Team:</b> {safe_text(batter_team) if batter_team else "Unknown"} • <b>Slot:</b> Over {over_ui}, Ball {ball_ui_1to6} • <b>Window:</b> 3 overs</div>
</div>
""",
        unsafe_allow_html=True
    )

    base = batter_win[
        (batter_win["batter"] == st.session_state["batter_sel"]) &
        (batter_win["start_over"] == start_over0) &
        (batter_win["start_ball_in_over"] == start_ball0)
    ].copy()

    if base.empty:
        st.warning("No historical starts for this batter at the selected over/ball slot.")
    else:
        tot = base.agg({"starts": "sum", "runs": "sum", "balls": "sum", "outs": "sum"})
        sr = (tot["runs"] / tot["balls"] * 100) if tot["balls"] > 0 else np.nan
        dps = (tot["outs"] / tot["starts"] * 100) if tot["starts"] > 0 else np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("SR", f"{sr:.1f}" if not np.isnan(sr) else "NA")
        k2.metric("Dismissal % (per start)", f"{dps:.1f}%" if not np.isnan(dps) else "NA")
        k3.metric("Historical starts (sample)", f"{int(tot['starts'])}")
        k4.metric("Balls in sample", f"{int(tot['balls'])}")

        st.markdown("#### Opposition split (same slot)")
        vs = base.groupby("bowling_team", as_index=False).agg(
            starts=("starts", "sum"),
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            outs=("outs", "sum"),
        )
        vs["sr"] = np.where(vs["balls"] > 0, vs["runs"] / vs["balls"] * 100, np.nan)
        vs["dismissal_%"] = np.where(vs["starts"] > 0, vs["outs"] / vs["starts"] * 100, np.nan)
        vs = vs.sort_values("starts", ascending=False).rename(columns={"starts": "Historical starts (sample)"})
        st.dataframe(vs, use_container_width=True)

    st.markdown("#### Best batters overall (same slot)")
    slot_bat = batter_win[
        (batter_win["start_over"] == start_over0) &
        (batter_win["start_ball_in_over"] == start_ball0)
    ].copy()
    agg_bat = slot_bat.groupby("batter", as_index=False).agg(
        starts=("starts", "sum"), runs=("runs", "sum"), balls=("balls", "sum"), outs=("outs", "sum")
    )
    agg_bat["sr"] = np.where(agg_bat["balls"] > 0, agg_bat["runs"] / agg_bat["balls"] * 100, np.nan)
    agg_bat["dismissal_%"] = np.where(agg_bat["starts"] > 0, agg_bat["outs"] / agg_bat["starts"] * 100, np.nan)
    agg_bat = agg_bat[agg_bat["balls"] >= MIN_BALLS_BATTER_LEADERBOARD].sort_values("sr", ascending=False).head(50)
    agg_bat = agg_bat.rename(columns={"starts": "Historical starts (sample)"})
    st.dataframe(agg_bat, use_container_width=True)

    st.markdown("#### Top 5 bowlers vs this batter (same slot)")
    if not has_bowler:
        st.info("Bowler field not available; cannot compute bowlers vs batter.")
    else:
        bb = best_bowlers_vs_batter_in_slot(df, st.session_state["batter_sel"], start_over0, start_ball0, MIN_BALLS_BOWLER_LISTS)
        if bb.empty:
            st.info("No sufficient sample (min 6 legal balls).")
        else:
            show = bb[["bowler", "balls", "econ_vs_batter", "wkts", "wicket_pct_per_start", "starts"]].head(5)
            show = show.rename(columns={"starts": "Historical starts (sample)"})
            st.dataframe(show, use_container_width=True)

# ---- Bowler (RIGHT) ----
with right:
    if not has_bowler:
        st.warning("Bowler data is not available in this dataset (missing/empty `bowler`).")
        st.stop()

    st.session_state["bowler_sel"] = st.selectbox(
        "Bowler",
        options=all_bowlers,
        index=all_bowlers.index(st.session_state["bowler_sel"]) if st.session_state["bowler_sel"] in all_bowlers else 0,
        key="bowler_sel_widget"
    )

    bowler_team = get_primary_bowling_team(df, st.session_state["bowler_sel"])

    st.markdown(
        f"""
<div class="mc-card">
  <h3 style="margin:0;">{safe_text(st.session_state["bowler_sel"])}</h3>
  <div class="mc-meta"><b>Team:</b> {safe_text(bowler_team) if bowler_team else "Unknown"} • <b>Slot:</b> Over {over_ui}, Ball {ball_ui_1to6} • <b>Window:</b> 3 overs</div>
</div>
""",
        unsafe_allow_html=True
    )

    base_b = bowler_win[
        (bowler_win["bowler"] == st.session_state["bowler_sel"]) &
        (bowler_win["start_over"] == start_over0) &
        (bowler_win["start_ball_in_over"] == start_ball0)
    ].copy()

    if base_b.empty:
        st.warning("No historical starts for this bowler at the selected over/ball slot.")
    else:
        tot = base_b.agg({"starts": "sum", "runs_conceded": "sum", "legal_balls": "sum", "wkts": "sum"})
        overs_b = tot["legal_balls"] / 6.0
        econ = (tot["runs_conceded"] / overs_b) if overs_b > 0 else np.nan
        wps = (tot["wkts"] / tot["starts"] * 100) if tot["starts"] > 0 else np.nan

        total_wkts = float(df[df["bowler"] == st.session_state["bowler_sel"]]["bowler_wicket"].sum())
        wshare = (tot["wkts"] / total_wkts * 100) if total_wkts > 0 else np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Economy", f"{econ:.2f}" if not np.isnan(econ) else "NA")
        k2.metric("Wicket % (per start)", f"{wps:.1f}%" if not np.isnan(wps) else "NA")
        k3.metric("Wickets share of total", f"{wshare:.1f}%" if not np.isnan(wshare) else "NA")
        k4.metric("Historical starts (sample)", f"{int(tot['starts'])}")

        st.markdown("#### Opposition split (same slot)")
        ovs = base_b.groupby("batting_team", as_index=False).agg(
            starts=("starts", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            legal_balls=("legal_balls", "sum"),
            wkts=("wkts", "sum")
        )
        ovs["overs"] = ovs["legal_balls"] / 6.0
        ovs["econ"] = np.where(ovs["overs"] > 0, ovs["runs_conceded"] / ovs["overs"], np.nan)
        ovs["wicket_%"] = np.where(ovs["starts"] > 0, ovs["wkts"] / ovs["starts"] * 100, np.nan)
        ovs = ovs.sort_values("starts", ascending=False).rename(columns={"starts": "Historical starts (sample)"})
        st.dataframe(ovs, use_container_width=True)

    st.markdown("#### Best bowlers overall (same slot)")
    slot_bowl = bowler_win[
        (bowler_win["start_over"] == start_over0) &
        (bowler_win["start_ball_in_over"] == start_ball0)
    ].copy()
    agg_bowl = slot_bowl.groupby("bowler", as_index=False).agg(
        starts=("starts", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        legal_balls=("legal_balls", "sum"),
        wkts=("wkts", "sum"),
        total_wkts=("total_wkts", "max"),
    )
    agg_bowl["overs_bowled"] = agg_bowl["legal_balls"] / 6.0
    agg_bowl["econ"] = np.where(agg_bowl["overs_bowled"] > 0, agg_bowl["runs_conceded"] / agg_bowl["overs_bowled"], np.nan)
    agg_bowl["wicket_%"] = np.where(agg_bowl["starts"] > 0, agg_bowl["wkts"] / agg_bowl["starts"] * 100, np.nan)
    agg_bowl["wickets_share_pct"] = np.where(agg_bowl["total_wkts"] > 0, agg_bowl["wkts"] / agg_bowl["total_wkts"] * 100, np.nan)

    agg_bowl = agg_bowl[agg_bowl["legal_balls"] >= MIN_BALLS_BOWLER_LISTS]
    agg_bowl = agg_bowl.sort_values(["wickets_share_pct", "econ"], ascending=[False, True]).head(50)
    agg_bowl = agg_bowl.rename(columns={"starts": "Historical starts (sample)"})
    st.dataframe(agg_bowl, use_container_width=True)

    st.markdown("#### Top 5 batters vs this bowler (same slot)")
    bestb = best_batters_vs_bowler_in_slot(df, st.session_state["bowler_sel"], start_over0, start_ball0, MIN_BALLS_BOWLER_LISTS)
    if bestb.empty:
        st.info("No sufficient sample (min 6 balls faced).")
    else:
        showb = bestb[["batter", "balls", "sr_vs_bowler", "outs", "dismissal_pct_per_start", "starts"]].head(5)
        showb = showb.rename(columns={"starts": "Historical starts (sample)", "dismissal_pct_per_start": "dismissal_% (per start)"})
        st.dataframe(showb, use_container_width=True)


# -------------------------
# Matchup (inherits selections; NO dropdowns)
# -------------------------
st.markdown('<div id="matchup_anchor"></div>', unsafe_allow_html=True)

batter_sel = st.session_state["batter_sel"]
bowler_sel = st.session_state["bowler_sel"]

batter_team = get_primary_batting_team(df, batter_sel)
bowler_team = get_primary_bowling_team(df, bowler_sel)

st.markdown(
    f"""
<div class="mc-card">
  <h3 style="margin:0;">Matchup</h3>
  <div class="mc-meta"><b>{safe_text(batter_sel)}</b> ({safe_text(batter_team)}) vs <b>{safe_text(bowler_sel)}</b> ({safe_text(bowler_team)}) •
  <b>Slot:</b> Over {over_ui}, Ball {ball_ui_1to6} • <b>Window:</b> 3 overs</div>
</div>
""",
    unsafe_allow_html=True
)

res = matchup_batter_vs_bowler(df, batter_sel, bowler_sel, start_over0, start_ball0)
if res.get("ok", 0) == 0:
    st.warning(res.get("reason", "Could not compute matchup."))
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batter SR vs bowler", f"{res['batter_sr']:.1f}" if not np.isnan(res["batter_sr"]) else "NA")
    c2.metric("Batter dismissal % (per start)", f"{res['batter_dismissal_pct_per_start']:.1f}%" if not np.isnan(res["batter_dismissal_pct_per_start"]) else "NA")
    c3.metric("Bowler economy vs batter", f"{res['bowler_econ']:.2f}" if not np.isnan(res["bowler_econ"]) else "NA")
    c4.metric("Bowler wicket % (per start)", f"{res['bowler_wicket_pct_per_start']:.1f}%" if not np.isnan(res["bowler_wicket_pct_per_start"]) else "NA")

    st.markdown("#### Top bowlers vs this batter (same slot)")
    bb = best_bowlers_vs_batter_in_slot(df, batter_sel, start_over0, start_ball0, MIN_BALLS_BOWLER_LISTS)
    if bb.empty:
        st.info("No sufficient sample (min 6 legal balls).")
    else:
        show = bb[["bowler", "balls", "econ_vs_batter", "wkts", "wicket_pct_per_start", "starts"]].head(10)
        show = show.rename(columns={"starts": "Historical starts (sample)"})
        st.dataframe(show, use_container_width=True)

    st.markdown("#### Top batters vs this bowler (same slot)")
    bestb = best_batters_vs_bowler_in_slot(df, bowler_sel, start_over0, start_ball0, MIN_BALLS_BOWLER_LISTS)
    if bestb.empty:
        st.info("No sufficient sample (min 6 balls faced).")
    else:
        showb = bestb[["batter", "balls", "sr_vs_bowler", "outs", "dismissal_pct_per_start", "starts"]].head(10)
        showb = showb.rename(columns={"starts": "Historical starts (sample)", "dismissal_pct_per_start": "dismissal_% (per start)"})
        st.dataframe(showb, use_container_width=True)
