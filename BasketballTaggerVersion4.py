import streamlit as st
import pandas as pd
from datetime import datetime, date
import re
import io
import json
from typing import List, Dict

st.set_page_config(page_title="Basketball Tagging App", layout="wide")

# ---------------------------
# Helpers & Session State
# ---------------------------
def init_state():
    st.session_state.setdefault("plays", [])                 # list[str]
    st.session_state.setdefault("log", [])                   # list[dict]
    st.session_state.setdefault("selected_play", None)       # str | None
    st.session_state.setdefault("opponent", "")
    st.session_state.setdefault("game_date", date.today())
    st.session_state.setdefault("quarter", "")
    st.session_state.setdefault("new_play", "")
    st.session_state.setdefault("pending_ft_play", None)     # str | None
    st.session_state.setdefault("__exports_ready", False)

def safe_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s or "file"

def event_row(play: str, quarter: str, etype: str, result: str, points: int, payload: Dict=None):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "opponent": st.session_state["opponent"],
        "game_date": str(st.session_state["game_date"]),
        "quarter": quarter,
        "play": play,
        "event_type": etype,    # "shot" | "foul" | "ft" | "to"
        "result": result,       # human text for quick scan
        "points": points,
        "extra": payload or {}, # e.g., {"ftm":2,"fta":2} or {"pt":2,"made":True}
    }

def add_log(row: Dict):
    st.session_state["log"].append(row)

def compute_metrics_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces the table you asked for:
    Play | Attempts | Points | PPP | Frequency | Success Rate
    Definitions (default / "Simple"):
      - Attempts = every time you tag an outcome for that play (includes fouls; excludes pure FT rows)
      - Points = shot points + FT points
      - PPP = Points / Attempts
      - Frequency = Attempts / total Attempts
      - Success Rate = FG made / (FG made + FG missed) [fouls & TO not in denominator]
    """
    if df.empty:
        return pd.DataFrame(columns=["Play", "Attempts", "Points", "PPP", "Frequency", "Success Rate"])

    # Derive helpers
    shots = df[df["event_type"] == "shot"]
    fts   = df[df["event_type"] == "ft"]
    fouls = df[df["event_type"] == "foul"]
    tos   = df[df["event_type"] == "to"]

    # Attempts (include shots + fouls + turnovers if enabled as attempts? -> we keep baseline as shots+fouls+to if to enabled)
    # We'll treat "attempts" as any non-FT tag you made for a play (matches the live tagging rhythm).
    attempts = df[df["event_type"].isin(["shot", "foul", "to"])].groupby("play").size().rename("Attempts")

    # Points = shot points + FT points
    points = df.groupby("play")["points"].sum().rename("Points")

    # Success Rate = FG% (made shots / shot attempts)
    made_mask = (shots["extra"].apply(lambda x: x.get("made", False)))
    made_counts = shots[made_mask].groupby("play").size()
    shot_attempts = shots.groupby("play").size()
    def succ_rate(p):
        made = int(made_counts.get(p, 0))
        atts = int(shot_attempts.get(p, 0))
        return (made / atts) if atts else 0.0

    # Assemble
    base = pd.concat([attempts, points], axis=1).reset_index().rename(columns={"play":"Play"})
    total_attempts = base["Attempts"].sum() if not base.empty else 1
    base["PPP"] = base["Points"] / base["Attempts"]
    base["Frequency"] = base["Attempts"] / (total_attempts if total_attempts else 1)
    base["Success Rate"] = base["Play"].map(succ_rate)

    # Sort by PPP then Attempts
    base = base.sort_values(by=["PPP", "Attempts"], ascending=[False, False]).reset_index(drop=True)
    return base

def compute_metrics_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced metrics per play (NBA-style):
      - FGA, 3PA, eFG% = (FGM + 0.5*3PM)/FGA
      - FTA, FT Rate = FTA/FGA
      - TO, TO Rate = TO / (FGA + 0.44*FTA + TO)
      - Possessions ≈ FGA + 0.44*FTA + TO
      - PPP = Points / Possessions
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Play","FGA","FGM","3PA","3PM","FTA","FTM","TO","Points",
            "Poss","PPP","eFG%","FT Rate","TO Rate"
        ])

    shots = df[df["event_type"] == "shot"].copy()
    fts   = df[df["event_type"] == "ft"].copy()
    tos   = df[df["event_type"] == "to"].copy()

    if not shots.empty:
        shots["is3"] = shots["extra"].apply(lambda x: 1 if x.get("pt", 0) == 3 else 0)
        shots["made"] = shots["extra"].apply(lambda x: 1 if x.get("made", False) else 0)
    else:
        shots["is3"] = []
        shots["made"] = []

    # Counts
    per_play = df.groupby("play")["points"].sum().rename("Points")
    fga = shots.groupby("play").size().rename("FGA")
    fgm = shots.groupby("play")["made"].sum().rename("FGM") if not shots.empty else pd.Series(dtype=int)
    p3a = shots.groupby("play")["is3"].sum().rename("3PA") if not shots.empty else pd.Series(dtype=int)
    p3m = shots[shots["is3"]==1].groupby("play")["made"].sum().rename("3PM") if not shots.empty else pd.Series(dtype=int)
    fta = fts.groupby("play").apply(lambda g: int(sum([r.get("fta",0) for r in g["extra"]]))).rename("FTA") if not fts.empty else pd.Series(dtype=int)
    ftm = fts.groupby("play").apply(lambda g: int(sum([r.get("ftm",0) for r in g["extra"]]))).rename("FTM") if not fts.empty else pd.Series(dtype=int)
    to  = tos.groupby("play").size().rename("TO") if not tos.empty else pd.Series(dtype=int)

    # Combine
    adv = pd.concat([fga, fgm, p3a, p3m, fta, ftm, to, per_play], axis=1).fillna(0).reset_index().rename(columns={"play":"Play"})
    # Possessions & rates
    adv["Poss"] = adv["FGA"] + 0.44*adv["FTA"] + adv["TO"]
    adv["PPP"]  = adv["Points"] / adv["Poss"].replace({0: float("nan")})
    adv["eFG%"] = (adv["FGM"] + 0.5*adv["3PM"]) / adv["FGA"].replace({0: float("nan")})
    adv["FT Rate"] = adv["FTA"] / adv["FGA"].replace({0: float("nan")})
    adv["TO Rate"] = adv["TO"] / adv["Poss"].replace({0: float("nan")})
    adv = adv.fillna(0.0).sort_values(by=["PPP","Poss"], ascending=[False,False]).reset_index(drop=True)
    return adv

def filtered_df(df: pd.DataFrame, plays: List[str], quarter: str):
    if df.empty:
        return df
    out = df.copy()
    if plays:
        out = out[out["play"].isin(plays)]
    if quarter and quarter != "All":
        out = out[out["quarter"] == quarter]
    return out

init_state()

# ---------------------------
# Sidebar: Game Setup
# ---------------------------
st.sidebar.header("Game Setup")
st.session_state["opponent"] = st.sidebar.text_input("Opponent", value=st.session_state["opponent"])
st.session_state["game_date"] = st.sidebar.date_input("Game Date", value=st.session_state["game_date"])
st.session_state["quarter"] = st.sidebar.selectbox(
    "Quarter", ["", "1", "2", "3", "4", "OT"],
    index=(["", "1", "2", "3", "4", "OT"].index(st.session_state["quarter"])
           if st.session_state["quarter"] in ["", "1", "2", "3", "4", "OT"] else 0)
)

ready_to_tag = bool(st.session_state["opponent"] and st.session_state["game_date"] and st.session_state["quarter"])

# ---------------------------
# Sidebar: Settings (Pro)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
enable_to = st.sidebar.toggle("Enable Turnovers", value=True, help="Adds a Turnover button.")
enable_ft = st.sidebar.toggle("Enable Free-Throw workflow", value=True, help="After a foul, record FTs to the same play.")
ppp_mode = st.sidebar.selectbox(
    "PPP Method",
    ["Simple (Points/Attempts)", "NBA Possessions (Points / (FGA + 0.44*FTA + TO))"],
    help="Switch between your original PPP and an NBA-style possession model."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Playbook")
st.session_state["new_play"] = st.sidebar.text_input("New Play Name", value=st.session_state["new_play"])

def add_play():
    raw = st.session_state["new_play"].strip()
    if not raw:
        return
    existing_lower = {p.lower() for p in st.session_state["plays"]}
    if raw.lower() in existing_lower:
        st.sidebar.warning("Play already exists.")
        return
    st.session_state["plays"].append(raw)
    st.session_state["new_play"] = ""

if st.sidebar.button("ADD NEW PLAY", use_container_width=True):
    add_play()

# Import playbook CSV (one play per line)
pb_upload = st.sidebar.file_uploader("Import Playbook (CSV, one play per row)", type=["csv"])
if pb_upload is not None:
    try:
        lines = pd.read_csv(pb_upload, header=None)[0].dropna().astype(str).tolist()
        added = 0
        existing_lower = {p.lower() for p in st.session_state["plays"]}
        for name in lines:
            if name.strip().lower() not in existing_lower:
                st.session_state["plays"].append(name.strip())
                existing_lower.add(name.strip().lower())
                added += 1
        st.sidebar.success(f"Imported {added} plays.")
    except Exception as e:
        st.sidebar.error(f"Playbook import failed: {e}")

with st.sidebar.expander("Manage Playbook"):
    if st.session_state["plays"]:
        to_remove = st.multiselect("Remove plays", st.session_state["plays"], key="rm_multiselect")
        if st.button("Remove selected"):
            st.session_state["plays"] = [p for p in st.session_state["plays"] if p not in to_remove]
            st.success(f"Removed {len(to_remove)} plays.")
    else:
        st.caption("No plays yet.")

st.sidebar.markdown("---")
if st.sidebar.button("Reset Game (clears log & selections)", type="secondary"):
    st.session_state["log"] = []
    st.session_state["selected_play"] = None
    st.session_state["pending_ft_play"] = None
    st.session_state["__exports_ready"] = False
    st.success("Game state cleared.")

# ---------------------------
# Main Header
# ---------------------------
st.title("🏀 Basketball Tagging Application")

if not ready_to_tag:
    st.warning("Select Opponent, Game Date, and Quarter in the sidebar to begin tagging.")
    st.stop()
else:
    st.write(f"**Game:** vs **{st.session_state['opponent']}** | **Date:** {st.session_state['game_date']} | **Quarter:** {st.session_state['quarter']}")

# ---------------------------
# Tagging UI
# ---------------------------
if not st.session_state["plays"]:
    st.info("Add at least one play in the sidebar to start tagging.")
else:
    st.subheader("Select a Play")
    cols_per_row = 4
    rows = (len(st.session_state["plays"]) + cols_per_row - 1) // cols_per_row
    idx = 0
    for r in range(rows):
        row_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            if idx >= len(st.session_state["plays"]): break
            play = st.session_state["plays"][idx]
            if row_cols[c].button(play, key=f"play_btn_{idx}", use_container_width=True):
                st.session_state["selected_play"] = play
            idx += 1

# Action bar
if st.session_state["selected_play"]:
    st.markdown(f"**Tagging:** `{st.session_state['selected_play']}`")
    a, b, c, d, e, f, g = st.columns(7)

    if a.button("Made 2", use_container_width=True):
        add_log(event_row(st.session_state["selected_play"], st.session_state["quarter"], "shot", "Made 2", 2, {"pt":2,"made":True}))
    if b.button("Made 3", use_container_width=True):
        add_log(event_row(st.session_state["selected_play"], st.session_state["quarter"], "shot", "Made 3", 3, {"pt":3,"made":True}))
    if c.button("Missed 2", use_container_width=True):
        add_log(event_row(st.session_state["selected_play"], st.session_state["quarter"], "shot", "Missed 2", 0, {"pt":2,"made":False}))
    if d.button("Missed 3", use_container_width=True):
        add_log(event_row(st.session_state["selected_play"], st.session_state["quarter"], "shot", "Missed 3", 0, {"pt":3,"made":False}))
    if e.button("Foul", use_container_width=True):
        add_log(event_row(st.session_state["selected_play"], st.session_state["quarter"], "foul", "Foul", 0, {}))
        if enable_ft:
            st.session_state["pending_ft_play"] = st.session_state["selected_play"]
    if enable_to and f.button("Turnover", use_container_width=True):
        add_log(event_row(st.session_state["selected_play"], st.session_state["quarter"], "to", "Turnover", 0, {}))
    if g.button("Undo Last", use_container_width=True):
        if st.session_state["log"]:
            st.session_state["log"].pop()
            st.toast("Last tag removed.")
        else:
            st.toast("No tags to undo.", icon="⚠️")

# Pending FT panel
if enable_ft and st.session_state["pending_ft_play"]:
    with st.expander(f"Record Free Throws for: {st.session_state['pending_ft_play']}", expanded=True):
        col1, col2, col3 = st.columns([2,2,1])
        fta = col1.number_input("FTA", min_value=1, max_value=3, value=2, step=1)
        ftm = col2.number_input("FTM", min_value=0, max_value=3, value=2, step=1)
        if col3.button("Add FT"):
            ftm = min(ftm, fta)
            add_log(event_row(st.session_state["pending_ft_play"], st.session_state["quarter"], "ft", f"FT {ftm}/{fta}", ftm, {"fta": int(fta), "ftm": int(ftm)}))
            st.session_state["pending_ft_play"] = None
            st.success("Free throws recorded.")

# ---------------------------
# Filters
# ---------------------------
st.markdown("---")
st.subheader("Filters")
all_plays = sorted(set([r["play"] for r in st.session_state["log"]])) if st.session_state["log"] else []
f1, f2, f3 = st.columns([3,2,2])
play_filter = f1.multiselect("Plays", options=all_plays, default=[], help="Leave empty = all plays")
quarter_filter = f2.selectbox("Quarter", options=["All","","1","2","3","4","OT"], index=0)
apply_to_export = f3.toggle("Apply filters to exports", value=False)

# Build DataFrames (filtered for display)
log_df_full = pd.DataFrame(st.session_state["log"])
log_df_view = filtered_df(log_df_full, play_filter, quarter_filter)

# Metrics table (baseline)
st.subheader("📊 Per Play Metrics")
base_df = compute_metrics_base(log_df_view)
if base_df.empty:
    st.info("No data yet — tag some plays to see metrics.")
else:
    st.dataframe(
        base_df.style.format({
            "PPP": "{:.2f}",
            "Frequency": "{:.1%}",
            "Success Rate": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Quick visuals
    left, right = st.columns(2)
    with left:
        st.caption("PPP by Play")
        st.bar_chart(base_df.set_index("Play")["PPP"], use_container_width=True)
    with right:
        st.caption("Frequency by Play")
        st.bar_chart(base_df.set_index("Play")["Frequency"], use_container_width=True)

# Advanced metrics (NBA-style)
with st.expander("Advanced Metrics (NBA-style)"):
    adv_source = log_df_view.copy()
    adv_df = compute_metrics_advanced(adv_source)
    st.dataframe(
        adv_df.style.format({
            "PPP": "{:.2f}",
            "eFG%": "{:.1%}",
            "FT Rate": "{:.2f}",
            "TO Rate": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )

# If PPP mode switch -> show which is in effect above
if ppp_mode.startswith("NBA"):
    st.caption("Note: Above 'PPP' card uses the Simple table. The Advanced table shows NBA-style PPP. Switch modes in Settings if you prefer NBA PPP for your main analysis.")

# Play-by-play table
st.subheader("🧾 Play-by-Play Log")
if log_df_view.empty:
    st.info("No events logged yet.")
else:
    st.dataframe(log_df_view, use_container_width=True, hide_index=True)

# ---------------------------
# Exports & Snapshots
# ---------------------------
st.subheader("📥 Export")
if st.button("Prepare Exports"):
    st.session_state["__exports_ready"] = True

export_df = log_df_view if apply_to_export else log_df_full

if st.session_state.get("__exports_ready") and not export_df.empty:
    # Build per-play metrics aligned with PPP mode choice
    if ppp_mode.startswith("NBA"):
        metrics_out = compute_metrics_advanced(export_df)
        # rename PPP column so filenames stay consistent but contents are NBA style
        metrics_csv = metrics_out.to_csv(index=False).encode("utf-8")
        metrics_suffix = "metrics_nbaPPP.csv"
    else:
        metrics_out = compute_metrics_base(export_df)
        metrics_csv = metrics_out.to_csv(index=False).encode("utf-8")
        metrics_suffix = "metrics_simplePPP.csv"

    log_csv = export_df.to_csv(index=False).encode("utf-8")
    json_blob = export_df.to_json(orient="records", indent=2).encode("utf-8")

    opp = safe_filename(str(st.session_state["opponent"]))
    gdt = safe_filename(str(st.session_state["game_date"]))
    qtr = safe_filename(str(st.session_state["quarter"]))

    st.download_button(
        "Download Per-Play Metrics (CSV)",
        data=metrics_csv,
        file_name=f"{opp}_{gdt}_Q{qtr}_{metrics_suffix}",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        "Download Play-by-Play (CSV)",
        data=log_csv,
        file_name=f"{opp}_{gdt}_Q{qtr}_playbyplay.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        "Download Snapshot (JSON)",
        data=json_blob,
        file_name=f"{opp}_{gdt}_Q{qtr}_snapshot.json",
        mime="application/json",
        use_container_width=True
    )

# ---------------------------
# Import Snapshot (optional)
# ---------------------------
with st.expander("Load Snapshot (JSON)"):
    up = st.file_uploader("Upload snapshot.json created by this app", type=["json"])
    if up is not None:
        try:
            blob = json.load(up)
            if isinstance(blob, list) and all(isinstance(x, dict) for x in blob):
                st.session_state["log"] = blob
                st.success("Snapshot loaded into current game.")
            else:
                st.error("Invalid snapshot format.")
        except Exception as e:
            st.error(f"Failed to load snapshot: {e}")
