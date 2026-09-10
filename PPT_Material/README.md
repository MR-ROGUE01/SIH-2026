# PPT_Material — README
# SIH PS-26227 | Ministry of Defence | Satellite Intelligence System

Is folder mein PPT ke liye sab kuch hai. Neeche dekho kya kahan hai.

---

## FOLDER STRUCTURE

### 01_Change_Detection_Outputs/
- **Kya hai**: 3-panel change evidence images
- **Format**: BEFORE | AFTER (red bounding boxes on changed objects) | CHANGE HEATMAP
- **PPT slide use**: "Change Detection Results" slide pe daalo
- **Best images for PPT** (highest confidence wale):
  - change_1_2_MONSOON_2021-06-01_to_2023-06-01.png  ← Road development
  - change_0_3_MONSOON_2021-06-01_to_2023-06-01.png  ← Construction
  - change_0_0_DRY_WINTER_2022-12-01_to_2024-12-01.png ← Clearance

### 02_Semantic_Search_Outputs/
- **Kya hai**: Text query se nikale satellite tile results
- **PPT slide use**: "Semantic Search Demo" slide
- **Best for PPT**:
  - search_newly_built_structures_near_a_river.png
  - search_road_development_area.png
  - search_construction_site.png

### 03_System_Architecture/
- **Kya hai**: ARCHITECTURE.md — full pipeline diagram
- **PPT slide use**: "System Architecture" slide

### 04_Code_Snapshots/
- **Kya hai**: Saare 9 Python source files
- **PPT slide use**: "Technical Implementation" slide pe key snippets

### 05_Results_Data/
- **Kya hai**: CSV files + evaluation report
  - search_results.csv — semantic search results
  - analyst_queue.csv — top 9 change candidates with confidence scores
  - tile_catalogue.csv — all 180 tiles with coordinates
  - evaluation_report.md — performance metrics

### SIH_26227_Complete_Document.md
- **Kya hai**: FULL document — problem statement se lekar results tak
- **PPT slide use**: Source of truth for all slides

---

## PPT SLIDES SUGGESTED ORDER (SIH Format)

1. **Title Slide** — PS 26227, Team Name, Institute
2. **Problem Statement** — Section 2.1 from document
3. **Our Approach** — System architecture diagram
4. **Tech Stack** — Table from Section 8
5. **Phase Results** — Phases 1-8 summary
6. **Semantic Search Demo** — search result images from 02/
7. **Change Detection Results** — 3-panel images from 01/
8. **Analyst Dashboard** — Screenshot of Streamlit UI (take manually)
9. **False Alarm Suppression** — Confidence table
10. **PS Requirement Mapping** — Table from Section 11
11. **Performance Metrics** — 88ms latency, 868km², 180 tiles
12. **How to Run** — Demo steps from Section 9

---

## DASHBOARD SCREENSHOT LENE KA TARIKA

1. Terminal mein chalaao:
   ```
   .venv\Scripts\python.exe -m streamlit run CODE\analyst_dashboard.py
   ```
2. Browser mein `http://localhost:8501` khuljayega
3. Har tab ka screenshot lo (Windows: Win+Shift+S)
4. Save karo is folder mein: `PPT_Material/06_Dashboard_Screenshots/`

---

*Total files in PPT_Material: ~35 images + 4 CSV + 2 MD + 9 Python files*
