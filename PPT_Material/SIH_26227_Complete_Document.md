# SIH 2026 — Problem Statement 26227
## Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
### Ministry of Defence (DGIS) | Complete Technical Reference Document
### (For PPT / Presentation Use — All Details Included)

---

## 1. PROBLEM STATEMENT OVERVIEW

| Field | Details |
|---|---|
| **PS ID** | 26227 |
| **Organization** | Ministry of Defence — Directorate General of Information Systems (DGIS), Indian Army |
| **Theme** | Smart Automation / Artificial Intelligence in Defence |
| **Category** | Software |

**Problem in simple words:**
Satellite image archives are growing fast. Old search systems only find images by date, coordinates, or sensor name. Analysts need to:
1. Search images **by meaning** — e.g., "show me construction sites near rivers"
2. Detect **real changes** between years (not fake seasonal ones)
3. Do all of this **without internet** (on-premises, air-gapped, sovereign system)

---

## 2. OUR SOLUTION — COMPLETE OVERVIEW

We built a **full end-to-end satellite intelligence pipeline** with 8 phases:

```
Download → Tile → Embed → Search → Detect Changes → Filter → Cluster → Dashboard
```

| PS Requirement | Our Solution |
|---|---|
| Semantic search by text/image | OpenCLIP (ViT-B/32) + FAISS vector database |
| Multi-temporal change detection | Same-season embedding distance + spectral delta |
| False alarm suppression | Seasonal mismatch penalty + quality masking |
| Unsupervised discovery | KMeans clustering on tile embeddings |
| Analyst review queue | Streamlit 4-tab interactive dashboard |
| Air-gapped on-premises | No internet at inference, all models local |
| Audit trail & provenance | JSON + CSV decision logs, full metadata |
| Performance evaluation | Sub-100ms query latency report |

---

## 3. TECH STACK — COMPLETE (Every Library & Version)

### 3.1 Programming Language
| Language | Version |
|---|---|
| **Python** | 3.11.9 (64-bit, Windows AMD64) |

### 3.2 Deep Learning / AI / Embeddings
| Library | Version | Used For |
|---|---|---|
| **PyTorch** | 2.14.0 | Deep learning backend for CLIP model |
| **TorchVision** | 0.29.0 | Image transforms & preprocessing |
| **open-clip-torch** | 3.3.0 | OpenCLIP ViT-B/32 — satellite image + text encoder |
| **timm** | 1.0.29 | Vision transformer backbone (used by OpenCLIP) |
| **safetensors** | 0.8.0 | Model weight loading format |
| **huggingface_hub** | 1.30.0 | Model download from HuggingFace Hub |

> **Model Used**: `hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
> Pre-trained on LAION-2B dataset, supports both image and text encoding into 512-dim vectors.

### 3.3 Vector Search (FAISS)
| Library | Version | Used For |
|---|---|---|
| **faiss-cpu** | 1.15.0 | Approximate/exact nearest-neighbor vector search |

> **Index Type**: `IndexFlatIP` — exact cosine similarity (Inner Product after L2 normalization)
> **Index Size**: 180 vectors × 512 dimensions = 368 KB

### 3.4 Geospatial / Remote Sensing
| Library | Version | Used For |
|---|---|---|
| **rasterio** | 1.4.4 | Read/write GeoTIFF satellite files with CRS metadata |
| **pyproj** | 3.7.2 | Coordinate Reference System (CRS) transforms, UTM↔WGS84 |
| **shapely** | 2.1.2 | Geometry operations (bounding boxes, polygons) |
| **geopandas** | 1.1.4 | Geospatial dataframes |
| **affine** | 3.0.1 | Affine transforms for pixel↔coordinate mapping |
| **pyogrio** | 0.13.0 | Fast vector I/O backend for geopandas |
| **pystac** | 1.15.2 | SpatioTemporal Asset Catalog (STAC) for data access |
| **openeo** | 0.52.0 | Copernicus Data Space API client for download |
| **xarray** | 2025.1.1 | N-dimensional array for multi-band satellite data |

### 3.5 Scientific Computing & ML
| Library | Version | Used For |
|---|---|---|
| **numpy** | 2.4.6 | Array math, spectral delta computation |
| **scipy** | 1.17.1 | Morphological operations (label, binary_closing) for object marking |
| **scikit-learn** | 1.9.0 | KMeans clustering (8 terrain clusters), silhouette scoring |
| **pandas** | 2.3.3 | CSV reading/writing, results dataframes |
| **joblib** | 1.6.0 | Parallel processing backend |

### 3.6 Image Processing & Visualization
| Library | Version | Used For |
|---|---|---|
| **Pillow** | 12.3.0 | Image loading, RGB conversion, saving previews |
| **matplotlib** | (via scipy) | Colormap (RdYlGn) for change heatmap |

### 3.7 Web Dashboard / UI
| Library | Version | Used For |
|---|---|---|
| **streamlit** | 1.63.0 | Interactive analyst dashboard (4-tab web UI) |
| **altair** | 6.2.2 | Charts inside Streamlit |
| **pydeck** | 0.9.3 | 3D geospatial map layers |
| **pyarrow** | 25.0.1 | Fast data transfer for Streamlit dataframes |

### 3.8 Networking / Download
| Library | Version | Used For |
|---|---|---|
| **requests** | 2.34.2 | HTTP downloads from Copernicus API |
| **tqdm** | 4.70.0 | Progress bars during download |
| **urllib3** | 2.7.0 | HTTP connection pooling |
| **httpx** | 0.28.1 | Async HTTP client |

### 3.9 Utilities
| Library | Version | Used For |
|---|---|---|
| **regex** | 2026.9.10 | Text tokenization for CLIP text encoder |
| **ftfy** | 6.3.1 | Fix text encoding issues in queries |
| **PyYAML** | 6.0.3 | Configuration files |
| **python-dateutil** | 2.9.0 | Date parsing |
| **colorama** | 0.4.6 | Terminal colored output |
| **filelock** | 3.32.6 | Safe concurrent file access |

### 3.10 Hardware Environment (Tested On)
| Component | Specification |
|---|---|
| **OS** | Windows 11 (AMD64) |
| **CPU** | Intel Core i5/i7 (64-bit) |
| **GPU** | NVIDIA RTX 3050 (4GB VRAM) |
| **RAM** | 15.2 GB |
| **PyTorch Mode** | CPU (model runs on CPU — no CUDA dependency) |
| **Storage Used** | ~715 MB total (tiles + index + outputs) |

---

## 4. DATA SOURCE

| Parameter | Value |
|---|---|
| **Satellite** | Sentinel-2 MSI (MultiSpectral Instrument) |
| **Product Level** | Level-2A (Atmospherically corrected, Bottom-of-Atmosphere reflectance) |
| **Provider** | Copernicus Data Space Ecosystem (European Space Agency) |
| **API Used** | OpenEO API (via `openeo` Python library) |
| **Spectral Bands Used** | B02 (Blue, 10m), B03 (Green, 10m), B04 (Red, 10m), B08 (NIR, 10m) |
| **Pixel Data Type** | INT16 (range: −32768 to 32767) |
| **Nodata Value** | −32768 |
| **Area of Interest** | Ranchi, Jharkhand, India |
| **Bounding Box** | West: 85.15°E, South: 23.20°N, East: 85.45°E, North: 23.45°N |
| **Coverage** | ~31 km × 28 km ≈ 868 km² |
| **Temporal Epochs** | 6 dates: 2020-01, 2020-06, 2021-06, 2022-12, 2023-06, 2024-12 |
| **File Size** | ~67 MB per GeoTIFF |

---

## 5. SYSTEM ARCHITECTURE — PHASE BY PHASE

### Phase 1 — Data Download (`Data_download.py`)
- Authenticates to Copernicus Data Space via OpenEO API
- Downloads 6 Sentinel-2 L2A GeoTIFFs (2020–2024)
- Saves to `Dataset/` with timestamped filenames
- Logs download metadata to `Dataset/download_log.csv`

### Phase 2 — Tiling (`tile_images.py`)
- Reads each GeoTIFF using **rasterio**
- Applies global p2/p98 percentile stretch (p2=257, p98=2156) for consistent normalization
- Splits each image into **512×512 pixel tiles** using integer division (no edge padding/black tiles)
- Saves tiles as individual GeoTIFF files preserving CRS and geotransform
- Generates `Dataset/tile_catalogue.csv` with per-tile lat/lon bounds + acquisition date

**Result**: 180 clean tiles across 6 epochs

### Phase 3 — Embeddings + FAISS Index (`build_embeddings_index.py`)
- Loads each tile, converts 4-band INT16 → 3-band RGB uint8
- Feeds through **OpenCLIP ViT-B/32** visual encoder → 512-dim float32 vector
- L2-normalizes all vectors for cosine similarity via inner product
- Builds **FAISS IndexFlatIP** (exact nearest-neighbor, no approximation)
- Saves: `Dataset/Index/tiles.faiss`, `tile_metadata.json`, `stretch_bounds.json`

### Phase 4 — Semantic Search (`semantic_search.py`)
- **Text query**: tokenizes with CLIP tokenizer → 512-dim text embedding → FAISS search
- **Image query**: encodes tile → 512-dim image embedding → FAISS search
- Supports `--date-from`, `--date-to` metadata filters
- Saves results to `Dataset/search_results/search_results.csv`
- Generates result preview PNG (5-tile grid with scores)

**Performance**: Mean latency **88.28ms**, Median **74.81ms**

### Phase 5 — Change Detection (`change_detection.py`)
- Groups tiles by **season** (DRY_WINTER: Nov–Feb, MONSOON: Jun–Sep)
- Compares ONLY same-season pairs (eliminates seasonal false alarms)
- Computes **embedding cosine distance** between temporal pairs
- Applies **spectral delta** on normalized band values (band-wise difference)
- Detects change types:
  - `road_development` — linear spectral changes
  - `construction` — high spectral variance increase
  - `clearance` — vegetation loss signature
  - `water_extent` — NIR/SWIR water index shift
- Draws **red bounding boxes** using `scipy.ndimage.label` morphological segmentation
- Generates **3-panel PNG**: BEFORE | AFTER (marked objects) | CHANGE HEATMAP
- Exports `Dataset/Index/change_candidates.json`

**Result**: 34 change events detected

### Phase 6 — False Alarm Suppression (`false_alarm_suppression.py`)
- Reads `change_candidates.json`
- Applies **seasonal mismatch penalty** (−0.30 for cross-season pairs)
- Applies **quality factor** = 1 − nodata_fraction (noisy tiles penalized)
- `adjusted_confidence = raw_confidence × quality_factor − seasonal_penalty`
- Threshold: candidates below 0.30 moved to audit log, not review queue
- Outputs: `review_queue.json` (9 high-confidence), `change_candidates_audited.json` (all 34)

### Phase 7A — Discovery & Clustering (`clustering_discovery.py`)
- Loads 180 embedding vectors from FAISS index
- Runs **KMeans (k=8)** clustering — groups tiles by visual/spectral similarity
- Silhouette Score: 0.094 (acceptable for heterogeneous satellite data)
- Assigns terrain labels (urban dense, vegetation, water, barren, mixed, etc.)
- **"Find Similar Sites"**: given any tile → FAISS top-k cosine search → similar locations
- Saves cluster assignments to `Dataset/Index/tile_clusters.json`

### Phase 7B — Analyst Dashboard (`analyst_dashboard.py`)
- **Framework**: Streamlit 1.63.0
- **4 tabs**:
  1. 📋 **Change Review Queue** — ranked by confidence, confirm/reject buttons, audit trail
  2. 🔍 **Smart Search** — text + image queries, date filter, live preview
  3. 🌐 **Find Similar Locations** — one-click similar site discovery
  4. 📑 **Analyst Decisions** — decision log table, CSV download
- Saves analyst decisions to `Dataset/analyst_decisions.json`
- Sidebar: operational parameters, confirmed/rejected counters

### Phase 8 — Evaluation Report (`eval_report.py`)
- Runs 15 benchmark queries (text + image)
- Measures per-query latency via `time.perf_counter()`
- Reads system info via Windows `ctypes` (no psutil dependency)
- Outputs: `Dataset/evaluation_report.json`, `Dataset/evaluation_report.md`

---

## 6. PS REQUIREMENT MAPPING (Complete)

| PS Section | Requirement | Our Implementation | Status |
|---|---|---|---|
| 2.2.1 | Semantic text retrieval | CLIP text encoder + FAISS search | ✅ Done |
| 2.2.1 | Multimodal image-to-image retrieval | CLIP image encoder + FAISS search | ✅ Done |
| 2.2.1 | Metadata filters (date, location) | `--date-from`, `--date-to` CLI args | ✅ Done |
| 2.2.2 | Multi-temporal change analysis | Same-season comparison, 4 change types | ✅ Done |
| 2.2.2 | Earliest observation estimation | Backward time walk on change chain | ✅ Done |
| 2.2.2 | Object marking / localization | Bounding boxes via scipy morphology | ✅ Done |
| 2.2.3 | False alarm suppression | Seasonal + quality penalty calibration | ✅ Done |
| 2.2.3 | Confidence calibration | Adjusted confidence score output | ✅ Done |
| 2.2.4 | Unsupervised discovery | KMeans 8-cluster terrain grouping | ✅ Done |
| 2.2.4 | "Find similar sites" | FAISS cosine search from any tile | ✅ Done |
| 2.2.5 | Ranked analyst review queue | Streamlit Tab 1 with confidence sort | ✅ Done |
| 2.2.5 | Confirm / Reject workflow | Buttons + JSON audit trail | ✅ Done |
| 2.2.6 | On-premises / air-gapped operation | No external API at inference time | ✅ Done |
| 2.2.7 | Audit trail & provenance | analyst_decisions.json + CSV export | ✅ Done |
| 2.2.7 | Sensor & scene metadata | Lat/lon, date, sensor in every record | ✅ Done |
| 2.3 | Query latency < 100ms | Mean: 88.28ms, Median: 74.81ms | ✅ Done |
| 2.3 | Reproducible evaluation | eval_report.md with fixed parameters | ✅ Done |

---

## 7. KEY RESULTS & NUMBERS

| Metric | Value |
|---|---|
| Total satellite tiles | 180 (clean, no black tiles) |
| Tile size | 512 × 512 pixels |
| Spatial resolution | 10 meters/pixel |
| Coverage area | 868 km² |
| Temporal range | 2020–2024 (6 epochs) |
| Embedding dimensions | 512 |
| FAISS index size | 368 KB |
| Changes detected (raw) | 34 |
| High-confidence candidates | 9 (in review queue) |
| Terrain clusters | 8 (KMeans) |
| Mean query latency | 88.28 ms |
| Median query latency | 74.81 ms |
| Search preview images | 12 |
| Change evidence images | 34 (3-panel PNG each) |
| Total system footprint | ~715 MB |

---

## 8. CHANGE DETECTION TOP RESULTS

| Rank | Change Type | Confidence | Before Date | After Date | Objects Marked | Season |
|---|---|---|---|---|---|---|
| #1 | Road Development | 0.741 | 2021-06-01 | 2023-06-01 | 16 | MONSOON |
| #2 | Road Development | 0.629 | 2021-06-01 | 2023-06-01 | 30 | MONSOON |
| #3 | Road Development | 0.599 | 2021-06-01 | 2023-06-01 | 19 | MONSOON |
| #4 | Road Development | 0.569 | 2021-06-01 | 2023-06-01 | 21 | MONSOON |
| #5 | Road Development | 0.564 | 2021-06-01 | 2023-06-01 | 24 | MONSOON |
| #6 | Construction | 0.558 | 2021-06-01 | 2023-06-01 | 22 | MONSOON |
| #7 | Road Development | 0.554 | 2022-12-01 | 2024-12-01 | 18 | DRY_WINTER |
| #8 | Construction | 0.541 | 2021-06-01 | 2023-06-01 | 14 | MONSOON |
| #9 | Clearance | 0.521 | 2021-06-01 | 2023-06-01 | 12 | MONSOON |

---

## 9. SEMANTIC SEARCH — SAMPLE QUERIES TESTED

| Query Text | Search Type | Files in `02_Semantic_Search_Outputs/` |
|---|---|---|
| "newly built structures near a river" | Text | `search_newly_built_structures_near_a_river.png` |
| "road development area" | Text | `search_road_development_area.png` |
| "construction site" | Text | `search_construction_site.png` |
| "large vehicle concentrations on open ground" | Text | `search_large_vehicle_concentrations_on_open_gro.png` |
| "dense urban area" | Text | `search_dense_urban_area.png` |
| "open agricultural land" | Text | `search_open_agricultural_land.png` |
| "water body near settlement" | Text | `search_water_body_near_settlement.png` |
| "road build in last 2 year" | Text | `search_road_build_in_last_2_year.png` |
| ranchi_2020_01_tile_0_0.tif | Image | `search_ranchi_2020_01_tile_0_0.tif.png` |
| ranchi_2020_01_tile_2_3.tif | Image | `search_ranchi_2020_01_tile_2_3.tif.png` |
| ranchi_2020_06_tile_0_1.tif | Image | `search_ranchi_2020_06_tile_0_1.tif.png` |
| ranchi_2024_12_tile_3_4.tif | Image | `search_ranchi_2024_12_tile_3_4.tif.png` |

---

## 10. FOLDER STRUCTURE (GitHub Repo)

```
SIH-PS-26227/                          ← GitHub Repository Root
│
├── README.md                           ← Project overview
├── .gitignore                          ← Excludes venv, raw TIFs, __pycache__
│
├── CODE/                               ← All Python source code
│   ├── Data_download.py                ← Phase 1: Copernicus API download
│   ├── tile_images.py                  ← Phase 2: 512×512 tiling
│   ├── build_embeddings_index.py       ← Phase 3: CLIP embeddings + FAISS
│   ├── semantic_search.py              ← Phase 4: Text/image search
│   ├── change_detection.py             ← Phase 5: Multi-temporal analysis
│   ├── false_alarm_suppression.py      ← Phase 6: Confidence calibration
│   ├── clustering_discovery.py         ← Phase 7A: KMeans + similar sites
│   ├── analyst_dashboard.py            ← Phase 7B: Streamlit UI
│   └── eval_report.py                  ← Phase 8: Performance evaluation
│
└── PPT_Material/                       ← PPT / Presentation assets
    ├── README.md                       ← Guide (what goes on which slide)
    ├── SIH_26227_Complete_Document.md  ← THIS FILE — full reference
    ├── 01_Change_Detection_Outputs/    ← 14 × 3-panel change PNGs
    ├── 02_Semantic_Search_Outputs/     ← 12 search result images
    ├── 03_System_Architecture/         ← ARCHITECTURE.md pipeline diagram
    ├── 04_Code_Snapshots/              ← Copies of all 9 .py files
    └── 05_Results_Data/                ← CSVs, evaluation_report.md
```

---

## 11. HOW TO RUN — STEP BY STEP (For Judges / Demo)

```bash
# Prerequisites
Python 3.11, Git

# Step 1 — Clone repo
git clone https://github.com/MR-ROGUE01/SIH-PS-26227.git
cd SIH-PS-26227

# Step 2 — Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Step 3 — Install all dependencies
pip install torch torchvision open-clip-torch faiss-cpu rasterio pyproj numpy pandas pillow streamlit scikit-learn scipy openeo pystac requests tqdm

# Step 4 — Download satellite data (needs internet + Copernicus account)
python CODE\Data_download.py

# Step 5 — Tile the images
python CODE\tile_images.py

# Step 6 — Build CLIP embeddings + FAISS index
python CODE\build_embeddings_index.py

# Step 7 — Run semantic search
python CODE\semantic_search.py --query "road development area" --top 5

# Step 8 — Run change detection
python CODE\change_detection.py

# Step 9 — Run false alarm suppression
python CODE\false_alarm_suppression.py --final_threshold 0.30

# Step 10 — Launch analyst dashboard
streamlit run CODE\analyst_dashboard.py
# Opens at http://localhost:8501
```

---

## 12. INNOVATION POINTS (For PPT Highlights)

1. **Same-Season Comparison** — Original approach to eliminate seasonal false alarms (crop cycles, vegetation changes) by only comparing DRY_WINTER↔DRY_WINTER and MONSOON↔MONSOON pairs

2. **Foundation Model for Remote Sensing** — Used OpenCLIP (trained on LAION-2B) without fine-tuning, demonstrating zero-shot transfer to satellite imagery

3. **Dual-Modal Search** — Both natural language text AND satellite tile image can be used as search queries against same FAISS index

4. **Bounding Box Object Marking** — scipy.ndimage morphological segmentation automatically marks changed objects with red bounding boxes in 3-panel evidence images

5. **Full Air-Gapped Operation** — After initial model download, zero internet required. No cloud API calls at inference time. Sovereign, on-premises deployment

6. **Sub-100ms Query Latency** — 88ms mean latency on CPU-only hardware (no GPU required at inference)

7. **Complete Audit Trail** — Every analyst decision (confirm/reject) stored with timestamp, confidence, location, sensor metadata — full provenance chain preserved

---

## 13. GITHUB REPOSITORY

**URL**: https://github.com/MR-ROGUE01/SIH-PS-26227

**Branch**: `main`

**Key files for judges**:
- Source code → `CODE/` folder
- Output images → `PPT_Material/01_Change_Detection_Outputs/`
- Search results → `PPT_Material/02_Semantic_Search_Outputs/`
- Full results data → `PPT_Material/05_Results_Data/`

---

*Document Version: 1.0 | SIH 2026 | PS-26227 | Team Submission*
