"""
Satellite Intelligence Dashboard
SIH Problem Statement 26227 — Ministry of Defence

Run locally:
    streamlit run CODE\\analyst_dashboard.py

Deployed:
    https://sih-2026-4uujvnafnxwkkf8lznapb8.streamlit.app/

FIXES applied:
  - os.system() REMOVED → direct Python function calls (real FAISS search)
  - @st.cache_resource for CLIP model + FAISS index (load once, reuse)
  - Proper st.spinner + st.progress for search feedback
  - Cloud-safe: if index not found, clear error shown (not silent fail)
  - Streamlit sleep handled with keep-alive tip in sidebar
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

SIH_DIR      = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).upper() == "CODE" else SCRIPT_DIR
DATASET_DIR  = os.path.join(SIH_DIR, "Dataset")
INDEX_DIR    = os.path.join(DATASET_DIR, "Index")
TILES_DIR    = os.path.join(DATASET_DIR, "Tiles")
PREVIEWS_DIR = os.path.join(DATASET_DIR, "change_previews")
SEARCH_DIR   = os.path.join(DATASET_DIR, "search_results")

INDEX_FILE    = os.path.join(INDEX_DIR, "tiles.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "tile_metadata.json")
STRETCH_FILE  = os.path.join(INDEX_DIR, "stretch_bounds.json")
REVIEW_QUEUE_FILE = os.path.join(INDEX_DIR, "review_queue.json")
CLUSTERS_FILE     = os.path.join(INDEX_DIR, "tile_clusters.json")
DECISIONS_FILE    = os.path.join(DATASET_DIR, "analyst_decisions.json")

os.makedirs(SEARCH_DIR, exist_ok=True)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Intelligence — SIH 26227",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1f2c; padding: 12px; border-radius: 8px;
                border: 1px solid #2d3748; }
    .mod-badge {
        background-color: #1e3a8a; color: #93c5fd;
        padding: 4px 10px; border-radius: 4px;
        font-weight: 600; font-size: 0.85rem;
        display: inline-block; margin-bottom: 8px;
    }
    .search-hint {
        background: #1a2744; border-left: 3px solid #3b82f6;
        padding: 10px 14px; border-radius: 4px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── CLOUD-SAFE MODEL + INDEX LOADER ─────────────────────────────────────────
# @st.cache_resource → loads ONCE per session, not per button click

CLOUD_MODE = not os.path.exists(INDEX_FILE)   # True when deployed without Dataset/

@st.cache_resource(show_spinner="Loading CLIP model and FAISS index — please wait...")
def load_search_engine():
    """Load OpenCLIP + FAISS index once and cache for the session."""
    try:
        import torch
        import open_clip
        import faiss

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(device).eval()

        index    = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        with open(STRETCH_FILE) as f:
            bounds = json.load(f)

        return {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "index": index,
            "metadata": metadata,
            "p2": bounds["p2"],
            "p98": bounds["p98"],
            "device": device,
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tile_to_rgb(tile_path, p2, p98):
    """Convert 4-band GeoTIFF to RGB PIL image."""
    import rasterio
    with rasterio.open(tile_path) as src:
        data = src.read()
    blue, green, red = data[0], data[1], data[2]
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, p2, p98)
    rgb = ((rgb - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def run_text_search(engine, query: str, top_k: int, year_filter: str):
    """Run actual CLIP text encoding + FAISS search. Returns list of result dicts."""
    import torch

    model     = engine["model"]
    tokenizer = engine["tokenizer"]
    index     = engine["index"]
    metadata  = engine["metadata"]
    device    = engine["device"]

    # Encode query text → 512-dim vector
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        vec    = model.encode_text(tokens)
        vec    = vec / vec.norm(dim=-1, keepdim=True)   # L2 normalise
        vec_np = vec.cpu().numpy().astype(np.float32)

    # FAISS search — search more to allow year filtering
    fetch_k = min(top_k * 10, index.ntotal)
    scores, indices = index.search(vec_np, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        tile_file = meta.get("file", meta.get("tile_file", ""))
        acq_date  = meta.get("acquisition_date", meta.get("date", ""))

        # Year filter
        if year_filter != "All Dates" and year_filter not in acq_date:
            continue

        results.append({
            "rank":             len(results) + 1,
            "score":            float(score),
            "tile_file":        tile_file,
            "acquisition_date": acq_date,
            "lat_min":          meta.get("lat_min", 0),
            "lon_min":          meta.get("lon_min", 0),
        })
        if len(results) >= top_k:
            break

    return results


def run_image_search(engine, tile_name: str, top_k: int):
    """Run image-to-image FAISS search."""
    import torch

    model      = engine["model"]
    preprocess = engine["preprocess"]
    index      = engine["index"]
    metadata   = engine["metadata"]
    p2, p98    = engine["p2"], engine["p98"]
    device     = engine["device"]

    tile_path = os.path.join(TILES_DIR, tile_name)
    img_rgb   = tile_to_rgb(tile_path, p2, p98)

    with torch.no_grad():
        img_t = preprocess(img_rgb).unsqueeze(0).to(device)
        vec   = model.encode_image(img_t)
        vec   = vec / vec.norm(dim=-1, keepdim=True)
        vec_np = vec.cpu().numpy().astype(np.float32)

    scores, indices = index.search(vec_np, top_k + 1)  # +1 to skip self

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        tfile = meta.get("file", meta.get("tile_file", ""))
        if tfile == tile_name:     # skip the query tile itself
            continue
        results.append({
            "rank":             len(results) + 1,
            "score":            float(score),
            "tile_file":        tfile,
            "acquisition_date": meta.get("acquisition_date", meta.get("date", "")),
        })
        if len(results) >= top_k:
            break

    return results


# ─── DECISIONS FILE INIT ──────────────────────────────────────────────────────

if not os.path.exists(DECISIONS_FILE):
    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

def load_decisions():
    try:
        with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_decision(candidate_id, decision, notes, candidate_data):
    decisions = load_decisions()
    entry = {
        "candidate_id": candidate_id,
        "decision":     decision,
        "analyst_notes": notes,
        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "change_type":  candidate_data.get("change_type"),
        "confidence":   candidate_data.get("adjusted_confidence", candidate_data.get("confidence")),
        "before_date":  candidate_data.get("before_date"),
        "after_date":   candidate_data.get("after_date"),
        "before_tile":  candidate_data.get("before_tile", candidate_data.get("before_file")),
        "after_tile":   candidate_data.get("after_tile",  candidate_data.get("after_file")),
        "lat":  candidate_data.get("lat_min"),
        "lon":  candidate_data.get("lon_min"),
        "sensor": candidate_data.get("sensor", "Sentinel-2 L2A"),
    }
    existing_ids = [d.get("candidate_id") for d in decisions]
    if candidate_id in existing_ids:
        for i, d in enumerate(decisions):
            if d.get("candidate_id") == candidate_id:
                decisions[i] = entry
    else:
        decisions.append(entry)
    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="mod-badge">MINISTRY OF DEFENCE | DGIS</div>',
                unsafe_allow_html=True)
    st.title("🛰️ SIH PS-26227")
    st.caption("Satellite Semantic Retrieval & Change Analysis")
    st.divider()

    st.markdown("**Area Details:**")
    st.markdown("- **Location**: Ranchi, Jharkhand")
    st.markdown("- **Total Tiles**: 180 (512×512 each)")
    st.markdown("- **Years Covered**: 2020 — 2024")
    st.markdown("- **Satellite**: Sentinel-2 L2A (10m)")
    st.divider()

    decisions     = load_decisions()
    confirmed_ct  = sum(1 for d in decisions if d["decision"] == "CONFIRMED")
    rejected_ct   = sum(1 for d in decisions if d["decision"] == "REJECTED")
    st.metric("✅ Confirmed Changes",       confirmed_ct)
    st.metric("❌ Rejected (False Alarms)", rejected_ct)
    st.divider()

    if CLOUD_MODE:
        st.warning(
            "**Cloud Demo Mode**\n\n"
            "The FAISS index and tile images are not available on Streamlit Cloud "
            "(Dataset/ is excluded from git due to size).\n\n"
            "To run with full functionality, clone the repo and run locally:\n"
            "```\nstreamlit run CODE/analyst_dashboard.py\n```"
        )
    else:
        st.success("✅ Full mode — FAISS index loaded")

    st.caption("💡 **Tip:** Streamlit Cloud free tier sleeps after ~60 min of inactivity. "
               "Reload the page to wake it up — it takes ~30 seconds.")


# ─── MAIN TABS ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Change Review Queue",
    "🔍 Smart Search",
    "🌐 Find Similar Locations",
    "📑 Analyst Decisions",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHANGE REVIEW QUEUE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("📋 Change Review Queue")
    st.caption("Review and verify multi-temporal change detections — validate or dismiss flagged anomalies")

    if not os.path.exists(REVIEW_QUEUE_FILE):
        st.warning("Review queue not found. Run `python CODE/false_alarm_suppression.py` to generate it.")
    else:
        with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
            queue = json.load(f)

        if not queue:
            st.info("No candidates currently pending review.")
        else:
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                type_options = ["ALL"] + sorted(set(c["change_type"] for c in queue))
                selected_type = st.selectbox("Change Type Filter:", type_options)
            with col_f2:
                min_conf = st.slider("Min Confidence:", 0.20, 0.95, 0.35, 0.05)

            filtered = [
                c for c in queue
                if (selected_type == "ALL" or c["change_type"] == selected_type)
                and c.get("adjusted_confidence", c.get("confidence", 0)) >= min_conf
            ]
            st.write(f"**{len(filtered)}** candidates identified:")

            options = [
                f"#{i+1} | {c['change_type'].upper()} | Conf: {c.get('adjusted_confidence', c['confidence']):.3f} "
                f"| {c['before_date']} → {c['after_date']} | Objects: {c.get('num_objects_marked', 0)}"
                for i, c in enumerate(filtered)
            ]

            if options:
                sel_idx = st.selectbox("Select Target Candidate:", range(len(options)),
                                       format_func=lambda i: options[i])
                cand = filtered[sel_idx]
                cand_id = f"{cand.get('row',0)}_{cand.get('col',0)}_{cand['before_date']}_{cand['after_date']}"

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Change Type",  cand["change_type"].upper().replace("_", " "))
                m2.metric("Confidence",   f"{cand.get('adjusted_confidence', cand['confidence']):.3f}")
                m3.metric("Objects Marked", cand.get("num_objects_marked", "N/A"))
                m4.metric("First Seen",   cand.get("earliest_observation", cand["after_date"]))

                preview_file = cand.get("preview_file")
                preview_path = os.path.join(PREVIEWS_DIR, preview_file) if preview_file else None

                if preview_path and os.path.exists(preview_path):
                    st.image(preview_path, use_container_width=True,
                             caption=f"BEFORE | AFTER (marked) | HEATMAP — {preview_file}")
                else:
                    st.warning("Preview imagery unavailable on this deployment.")

                with st.expander("🔍 Full Details", expanded=False):
                    p1, p2_ = st.columns(2)
                    with p1:
                        st.markdown(f"**Before Tile**: `{cand.get('before_tile', cand.get('before_file'))}`")
                        st.markdown(f"**After Tile**:  `{cand.get('after_tile',  cand.get('after_file'))}`")
                        st.markdown(f"**Satellite**:   `{cand.get('sensor', 'Sentinel-2 L2A')}`")
                        st.markdown(f"**Embedding Distance**: `{cand.get('embedding_distance', 'N/A')}`")
                    with p2_:
                        st.markdown(f"**Lat Range**: `[{cand.get('lat_min',0):.4f}, {cand.get('lat_max',0):.4f}]`")
                        st.markdown(f"**Lon Range**: `[{cand.get('lon_min',0):.4f}, {cand.get('lon_max',0):.4f}]`")
                        st.markdown(f"**Seasonal Penalty**: `{cand.get('seasonal_penalty', 0.0)}`")
                        st.markdown(f"**Suppression Flags**: `{', '.join(cand.get('suppression_reasons', ['None']))}`")

                st.subheader("Analyst Verification")
                notes_input = st.text_area("Notes (optional):",
                    placeholder="e.g., New road confirmed near river bend.", key=f"note_{cand_id}")

                b1, b2, _ = st.columns([1, 1, 3])
                with b1:
                    if st.button("✅ Confirm Change", type="primary", use_container_width=True):
                        save_decision(cand_id, "CONFIRMED", notes_input, cand)
                        st.success("Confirmed! Logged in audit trail.")
                        st.rerun()
                with b2:
                    if st.button("❌ Reject (False Alarm)", use_container_width=True):
                        save_decision(cand_id, "REJECTED", notes_input, cand)
                        st.warning("Rejected! False alarm logged.")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SMART SEARCH  (FIXED — real CLIP+FAISS, no os.system)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🔍 Smart Search")
    st.caption("Semantic natural-language and image-to-image similarity search over satellite archives")

    # ── Cloud Mode notice ──────────────────────────────────────────────────────
    if CLOUD_MODE:
        st.error(
            "**Search unavailable in Cloud Demo Mode.**\n\n"
            "The FAISS index (`Dataset/Index/tiles.faiss`) is not present on this "
            "deployment because the Dataset folder is excluded from git (too large).\n\n"
            "**To use full search functionality**, run the app locally:\n"
            "```bash\ngit clone https://github.com/MR-ROGUE01/SIH-2026\n"
            "cd SIH-2026\n"
            "streamlit run CODE/analyst_dashboard.py\n```\n\n"
            "The dataset was generated by running `python CODE/Data_download.py` "
            "followed by the full pipeline."
        )
        st.info("👇 A preview of what the search outputs look like:")
        sample_csv = os.path.join(SEARCH_DIR, "search_results.csv")
        if os.path.exists(sample_csv):
            df_sample = pd.read_csv(sample_csv)
            st.dataframe(df_sample, use_container_width=True)
        st.stop()

    # ── Full Search Mode ───────────────────────────────────────────────────────
    engine = load_search_engine()
    if not engine.get("ok"):
        st.error(f"Failed to load search engine: {engine.get('error')}\n\n"
                 "Make sure FAISS index is built: `python CODE/build_embeddings_index.py`")
        st.stop()

    st.success(f"✅ FAISS index ready — {engine['index'].ntotal} tiles indexed")

    query_type = st.radio("Search Modality:",
                          ["Text Query (Natural Language)", "Image Query (Reference Tile)"],
                          horizontal=True)

    search_query   = ""
    image_tile_name = ""

    if query_type == "Text Query (Natural Language)":
        st.markdown("""
        <div class="search-hint">
        💡 <b>Try natural language queries like these:</b><br>
        "newly built structures near a river" &nbsp;|&nbsp;
        "large vehicle concentrations on open ground" &nbsp;|&nbsp;
        "road development area" &nbsp;|&nbsp;
        "dense urban construction site"
        </div>
        """, unsafe_allow_html=True)

        qc = st.columns(4)
        presets = [
            ("🌊 Structures near river", "newly built structures near a river"),
            ("🚜 Vehicle concentrations", "large vehicle concentrations on open ground"),
            ("🛣️ Road development", "road development area"),
            ("🏗️ Construction site", "dense urban construction site"),
        ]
        for col, (label, val) in zip(qc, presets):
            if col.button(label):
                st.session_state["sq"] = val

        search_query = st.text_input(
            "Enter search prompt:",
            value=st.session_state.get("sq", ""),
            placeholder="e.g. road construction near forest edge"
        )
    else:
        all_tiles = sorted(os.listdir(TILES_DIR)) if os.path.exists(TILES_DIR) else []
        if not all_tiles:
            st.warning("No tiles found. Run the data pipeline first.")
            st.stop()
        image_tile_name = st.selectbox("Select reference tile:", all_tiles)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        top_k = st.slider("Top Results (K):", 3, 10, 5)
    with col_opt2:
        year_filter = st.selectbox("Temporal Filter (Year):",
                                   ["All Dates", "2020", "2021", "2022", "2023", "2024"])

    if st.button("🔎 Execute Search", type="primary"):

        # ── Step-by-step progress feedback ────────────────────────────────────
        progress_bar = st.progress(0, text="Initialising search...")
        status_box   = st.empty()

        try:
            if query_type == "Text Query (Natural Language)":
                if not search_query.strip():
                    st.warning("Please enter a search query.")
                    progress_bar.empty()
                    st.stop()

                status_box.info(f"🧠 Encoding query: **\"{search_query}\"** with OpenCLIP ViT-B/32...")
                progress_bar.progress(25, text="Encoding text query with CLIP...")
                time.sleep(0.3)   # brief pause so user sees the step

                progress_bar.progress(55, text="Searching FAISS index...")
                status_box.info("⚡ Searching FAISS index for matching tiles...")

                t0      = time.perf_counter()
                results = run_text_search(engine, search_query, top_k, year_filter)
                elapsed = (time.perf_counter() - t0) * 1000

                progress_bar.progress(80, text="Loading tile images...")
                status_box.info("🖼️ Loading and rendering tile images...")

            else:
                status_box.info(f"🖼️ Encoding reference tile: **{image_tile_name}**...")
                progress_bar.progress(30, text="Encoding reference tile with CLIP...")
                time.sleep(0.3)

                progress_bar.progress(60, text="Searching FAISS index...")
                t0      = time.perf_counter()
                results = run_image_search(engine, image_tile_name, top_k)
                elapsed = (time.perf_counter() - t0) * 1000
                search_query = image_tile_name

            progress_bar.progress(100, text="Done!")
            status_box.empty()
            progress_bar.empty()

            if not results:
                st.warning("No results found for this query / year filter combination. "
                           "Try 'All Dates' or a broader query.")
            else:
                st.success(
                    f"✅ **{len(results)} matching tiles retrieved** — "
                    f"query: *\"{search_query}\"* — "
                    f"latency: **{elapsed:.1f} ms** (CLIP + FAISS, CPU)"
                )

                # Save to CSV
                df_res = pd.DataFrame(results)
                csv_path = os.path.join(SEARCH_DIR, "search_results.csv")
                df_res.to_csv(csv_path, index=False)

                # Display tile grid
                p2_v, p98_v = engine["p2"], engine["p98"]
                cols_disp   = st.columns(min(len(results), 5))
                for i, row in enumerate(results[:5]):
                    with cols_disp[i]:
                        st.markdown(f"**Rank #{row['rank']}**")
                        st.caption(f"Score: **{row['score']:.4f}**")
                        st.caption(f"Date: `{row['acquisition_date']}`")
                        t_path = os.path.join(TILES_DIR, row["tile_file"])
                        if os.path.exists(t_path):
                            try:
                                img = tile_to_rgb(t_path, p2_v, p98_v)
                                st.image(img, use_container_width=True,
                                         caption=row["tile_file"])
                            except Exception as ex:
                                st.caption(f"Preview error: {ex}")
                                st.code(row["tile_file"])
                        else:
                            st.code(row["tile_file"])

                with st.expander("📊 Full Results Table", expanded=False):
                    st.dataframe(df_res, use_container_width=True)

        except Exception as e:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Search failed: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FIND SIMILAR LOCATIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🌐 Find Similar Locations")
    st.caption("Unsupervised site discovery — retrieve locations with equivalent semantic signatures")

    sub1, sub2 = st.tabs(["🎯 Similar Sites Retrieval", "📊 Latent Terrain Clusters"])

    with sub1:
        if CLOUD_MODE:
            st.error("Similarity search requires local FAISS index — not available in cloud demo mode.")
        else:
            all_tiles = sorted(os.listdir(TILES_DIR)) if os.path.exists(TILES_DIR) else []
            chosen = st.selectbox("Select Target Tile:", all_tiles, key="disc_tile")

            if st.button("⚡ Discover Similar Sites", type="primary"):
                with st.spinner("Searching for visually similar locations..."):
                    try:
                        from clustering_discovery import find_similar_sites
                        engine_l = load_search_engine()
                        similar  = find_similar_sites(chosen, top_k=6)
                        if similar:
                            st.success(f"{len(similar)} similar locations discovered.")
                            cols = st.columns(len(similar))
                            p2_v, p98_v = engine_l["p2"], engine_l["p98"]
                            for idx, res in enumerate(similar):
                                with cols[idx]:
                                    st.markdown(f"**Match #{res['rank']}**")
                                    st.caption(f"Similarity: **{res['similarity_score']:.4f}**")
                                    st.caption(f"Date: `{res['date']}`")
                                    tp = os.path.join(TILES_DIR, res["tile_file"])
                                    if os.path.exists(tp):
                                        try:
                                            st.image(tile_to_rgb(tp, p2_v, p98_v),
                                                     use_container_width=True,
                                                     caption=res["tile_file"])
                                        except Exception:
                                            st.code(res["tile_file"])
                        else:
                            st.info("No similar locations found above threshold.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with sub2:
        st.subheader("8 Unsupervised Terrain Archetypes (KMeans k=8)")
        if os.path.exists(CLUSTERS_FILE):
            with open(CLUSTERS_FILE) as f:
                c_data = json.load(f)
            rows = [{"Cluster ID": cid,
                     "Total Tiles": info["total_tiles"],
                     "Example Tile": info["exemplar_tile"]}
                    for cid, info in c_data.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Cluster index not generated yet.")
            if not CLOUD_MODE and st.button("Initialize Clustering Pipeline"):
                with st.spinner("Running KMeans clustering..."):
                    import subprocess
                    subprocess.run([sys.executable, "CODE/clustering_discovery.py",
                                    "--build_clusters"], cwd=SIH_DIR)
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYST DECISIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📑 Analyst Decisions")
    st.caption("Complete audit trail of analyst verification decisions")

    decisions = load_decisions()
    if decisions:
        df_dec = pd.DataFrame(decisions)
        st.subheader(f"Total {len(df_dec)} decisions recorded")
        st.dataframe(df_dec, use_container_width=True)
        st.download_button(
            "📥 Download Decisions (CSV)",
            data=df_dec.to_csv(index=False).encode("utf-8"),
            file_name="analyst_decisions.csv",
            mime="text/csv",
        )
    else:
        st.info("No decisions recorded yet — review candidates in Tab 1.")

    st.divider()
    st.subheader("📊 System Info")
    st.json({
        "Location":          "Ranchi Urban & Peri-Urban, Jharkhand, India",
        "Bounding Box":      "85.15°E–85.45°E, 23.20°N–23.45°N (~868 km²)",
        "Time Period":       "2020-01-01 to 2024-12-31 (6 acquisitions)",
        "Satellite":         "Sentinel-2 MSI Level-2A (10m resolution, 4 bands)",
        "Total Tiles":       180,
        "Embedding Dims":    512,
        "Search Index":      "FAISS IndexFlatIP (Cosine Similarity)",
        "Mean Query Latency":"88.28 ms (CPU, CLIP + FAISS, 15 benchmark queries)",
        "Change Detection":  "Multi-Temporal Embedding Distance + Spectral Morphology",
        "False Alarm Filter":"Seasonal Mismatch Penalty (−0.30) + Quality Masking",
        "Cloud Mode":        CLOUD_MODE,
    })
