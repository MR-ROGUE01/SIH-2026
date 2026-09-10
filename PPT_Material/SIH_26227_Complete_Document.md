# SIH 2026 — Problem Statement 26227
## Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
### Ministry of Defence (DGIS) | Team Submission Document

---

## 1. PROBLEM STATEMENT

**PS ID**: 26227  
**Organization**: Ministry of Defence — Directorate General of Information Systems (DGIS)  
**Theme**: Smart Automation / AI in Defence  

**Problem**: Earth-observation archives are expanding rapidly. Conventional catalogues search only by metadata (date, coordinates). Analysts need to search satellite imagery *by meaning* — "show me construction sites near rivers" — and detect real changes vs. seasonal false alarms, all running **on-premises** (air-gapped, sovereign).

---

## 2. OUR SOLUTION — AT A GLANCE

| What We Built | How |
|---|---|
| 🔍 Semantic Search | OpenCLIP ViT-B/32 + FAISS vector index |
| 🛰️ Multi-Temporal Change Detection | Same-season embedding distance + spectral delta |
| 🚫 False Alarm Suppression | Seasonal penalty + quality masking |
| 🌐 Unsupervised Discovery | KMeans clustering on tile embeddings |
| 📋 Analyst Workflow | Streamlit dashboard — confirm/reject with audit trail |
| 📊 Evaluation | Sub-100ms query latency, 868 km² coverage |

**Key USP**: Fully **air-gapped** (no internet at inference), runs on RTX 3050 laptop (4GB VRAM), sovereign on-premises deployment.

---

## 3. TECHNICAL ARCHITECTURE

```
Sentinel-2 L2A GeoTIFF (6 epochs, 2020-2024)
        ↓
[Phase 2] Tiling → 180 × 512px chips (no black padding)
        ↓
[Phase 3] OpenCLIP Embeddings → FAISS IndexFlatIP (512-dim)
        ↓                              ↓
[Phase 4] Semantic Search      [Phase 5] Change Detection
  Text / Image query             Same-season comparison only
  FAISS cosine similarity        Bounding box object marking
        ↓                              ↓
                          [Phase 6] False Alarm Suppression
                            Seasonal mismatch penalty
                            Quality/nodata masking
                                       ↓
[Phase 7A] Discovery & Clustering    [Phase 7B] Analyst Dashboard
  8 KMeans terrain clusters           Streamlit 4-tab UI
  "Find Similar Sites" FAISS          Confirm / Reject audit trail
                                       ↓
                          [Phase 8] Evaluation Report
                            88ms mean latency | 868 km² | 180 tiles
```

---

## 4. DATASET DETAILS

| Parameter | Value |
|---|---|
| Satellite | Sentinel-2 MSI Level-2A |
| Resolution | 10 meters per pixel |
| Area of Interest | Ranchi, Jharkhand (85.15°E–85.45°E, 23.20°N–23.45°N) |
| Coverage | ~868 km² |
| Temporal Epochs | 6 (Jan-2020, Jun-2020, Jun-2021, Dec-2022, Jun-2023, Dec-2024) |
| Spectral Bands | B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) |
| Total Tiles | 180 clean 512×512 chips |
| Total Data | ~715 MB raw GeoTIFF |

---

## 5. PHASE-WISE RESULTS

### Phase 1 — Data Download
- ✅ 6 Sentinel-2 L2A GeoTIFFs downloaded from Copernicus Open Access Hub
- Each ~67MB, multi-band INT16 format

### Phase 2 — Tiling
- ✅ **180 clean tiles** (integer-division grid, zero black/padded tiles)
- `tile_catalogue.csv` — each tile has lat/lon bounds + acquisition date

### Phase 3 — Embeddings & FAISS Index
- ✅ **180 × 512-dim vectors** encoded via OpenCLIP ViT-B/32
- FAISS IndexFlatIP — exact cosine similarity search
- Index size: 368 KB (highly compact)

### Phase 4 — Semantic Search
- ✅ Text queries: "newly built structures near a river", "road development area", etc.
- ✅ Image queries: pass any tile → find visually similar tiles
- ✅ **Mean query latency: 88.28ms** (sub-100ms SLA met)
- 12 search result images generated (see `02_Semantic_Search_Outputs/`)

### Phase 5 — Change Detection
- ✅ **34 change events detected** across 6 temporal pairs
- Change types: `road_development`, `construction`, `clearance`, `water_extent`
- **Same-season-only comparison** (DRY_WINTER↔DRY_WINTER, MONSOON↔MONSOON) — eliminates seasonal false alarms
- **Bounding boxes** drawn on changed objects (12–30 objects per pair)
- 3-panel output: BEFORE | AFTER (marked) | CHANGE HEATMAP

### Phase 6 — False Alarm Suppression
- ✅ Seasonal mismatch penalty: −0.30 for cross-season pairs
- ✅ Quality masking: nodata fraction penalty
- Result: **9 high-confidence candidates** in analyst review queue
- All 34 raw candidates preserved in full audit log (nothing deleted)

### Phase 7A — Discovery & Clustering
- ✅ **8 KMeans terrain clusters** (vegetation, urban, water, barren, etc.)
- "Find Similar Sites": one click → FAISS returns visually similar locations
- Silhouette Score: 0.094 (expected low for heterogeneous satellite data)

### Phase 7B — Analyst Dashboard
- ✅ Streamlit 4-tab UI running at `localhost:8501`
- Tab 1: Change Review Queue — ranked by confidence, confirm/reject
- Tab 2: Smart Search — text + image queries in real-time
- Tab 3: Find Similar Locations — one-click site discovery
- Tab 4: Analyst Decisions — full audit trail, CSV export

### Phase 8 — Evaluation Report
- ✅ Mean Query Latency: **88.28 ms**
- ✅ Median Query Latency: **74.81 ms**
- ✅ Coverage: **868 km²**
- ✅ Architecture: Air-gapped, on-premises, CPU+GPU compatible

---

## 6. CHANGE DETECTION — TOP RESULTS

| Rank | Change Type | Confidence | Period | Objects Marked |
|---|---|---|---|---|
| #1 | Road Development | 0.741 | Jun-2021 → Jun-2023 | 16 |
| #2 | Road Development | 0.629 | Jun-2021 → Jun-2023 | 30 |
| #3 | Road Development | 0.599 | Jun-2021 → Jun-2023 | 19 |
| #4 | Road Development | 0.569 | Jun-2021 → Jun-2023 | 21 |
| #5 | Road Development | 0.564 | Jun-2021 → Jun-2023 | 24 |
| #6 | Construction | 0.558 | Jun-2021 → Jun-2023 | 22 |
| #7 | Road Development | 0.554 | Dec-2022 → Dec-2024 | 18 |
| #8 | Construction | 0.541 | Jun-2021 → Jun-2023 | 14 |
| #9 | Clearance | 0.521 | Jun-2021 → Jun-2023 | 12 |

*3-panel images in `01_Change_Detection_Outputs/` folder*

---

## 7. SEMANTIC SEARCH — SAMPLE QUERIES

| Query | Top Match | Score |
|---|---|---|
| "newly built structures near a river" | ranchi_2024_12_tile | High |
| "road development area" | ranchi_2023_06_tile | High |
| "construction site" | ranchi_2022_12_tile | High |
| "large vehicle concentrations on open ground" | ranchi_2021_06_tile | Medium |
| "water body near settlement" | ranchi_2020_06_tile | Medium |

*Search result images in `02_Semantic_Search_Outputs/` folder*

---

## 8. TECH STACK

| Component | Technology |
|---|---|
| Satellite Data | Sentinel-2 L2A (Copernicus OAH) |
| Image Encoder | OpenCLIP ViT-B/32 (open-source, on-premises) |
| Vector Search | FAISS IndexFlatIP |
| Change Detection | NumPy + SciPy morphology |
| Clustering | scikit-learn KMeans |
| Dashboard | Streamlit |
| Geospatial I/O | rasterio, pyproj |
| Language | Python 3.11 |
| Hardware Tested | RTX 3050 (4GB VRAM), Windows 11 |

---

## 9. HOW TO RUN (Demo Steps for Judges)

```bash
# Step 1 — Activate environment
.venv\Scripts\activate

# Step 2 — Run semantic search
python CODE\semantic_search.py --query "road development area" --top 5

# Step 3 — Run change detection
python CODE\change_detection.py

# Step 4 — Run false alarm suppression
python CODE\false_alarm_suppression.py --final_threshold 0.30

# Step 5 — Launch analyst dashboard
streamlit run CODE\analyst_dashboard.py
# Opens at http://localhost:8501
```

---

## 10. FOLDER STRUCTURE

```
SIH/
├── CODE/
│   ├── Data_download.py          ← Phase 1: Downloads satellite data
│   ├── tile_images.py            ← Phase 2: Cuts into 512px tiles
│   ├── build_embeddings_index.py ← Phase 3: CLIP embeddings + FAISS
│   ├── semantic_search.py        ← Phase 4: Text/image search
│   ├── change_detection.py       ← Phase 5: Multi-temporal analysis
│   ├── false_alarm_suppression.py← Phase 6: Confidence calibration
│   ├── clustering_discovery.py   ← Phase 7A: KMeans + similar sites
│   ├── analyst_dashboard.py      ← Phase 7B: Streamlit UI
│   └── eval_report.py            ← Phase 8: Performance evaluation
│
├── Dataset/
│   ├── Tiles/                    ← 180 clean 512×512 GeoTIFF tiles
│   ├── Index/                    ← FAISS index + metadata + results
│   ├── change_previews/          ← 34 × 3-panel change evidence PNGs
│   └── search_results/           ← Search result images + CSV
│
└── PPT_Material/                 ← THIS FOLDER (everything for PPT)
    ├── 01_Change_Detection_Outputs/
    ├── 02_Semantic_Search_Outputs/
    ├── 03_System_Architecture/
    ├── 04_Code_Snapshots/
    └── 05_Results_Data/
```

---

## 11. PS REQUIREMENT MAPPING

| PS Requirement | Our Implementation | Status |
|---|---|---|
| PS 2.2.1 — Semantic Retrieval | OpenCLIP + FAISS text/image search | ✅ Done |
| PS 2.2.2 — Multi-Temporal Change Analysis | Same-season detection, 4 change types | ✅ Done |
| PS 2.2.3 — False Alarm Suppression | Seasonal penalty + quality masking | ✅ Done |
| PS 2.2.4 — Discovery & Clustering | KMeans 8 clusters + similar site search | ✅ Done |
| PS 2.2.5 — Analyst Review Queue | Streamlit dashboard with confirm/reject | ✅ Done |
| PS 2.2.6 — On-Premises / Air-Gapped | No external API at inference time | ✅ Done |
| PS 2.2.7 — Audit Trail & Provenance | analyst_decisions.json + CSV export | ✅ Done |
| PS 2.3 — Evaluation | eval_report.md, <100ms latency | ✅ Done |

---

*Generated: SIH 2026 | PS-26227 | Ranchi AOI | Sentinel-2 L2A*
