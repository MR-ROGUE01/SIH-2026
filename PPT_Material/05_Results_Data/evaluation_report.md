# MoD/DGIS Problem Statement 26227 — Evaluation Report

**Organization**: Ministry of Defence (MoD) | Indian Army (DGIS)  
**Title**: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery  
**Generated**: 2026-09-10 17:54:15  
**Operational Status**: Sovereign On-Premises Air-Gapped Operation  

---

## 1. Geospatial & Archive Coverage
- **Area of Interest**: Ranchi Urban & Peri-Urban Envelope, Jharkhand, India
- **Coordinates**: Lat [23.20°N, 23.45°N], Lon [85.15°E, 85.45°E]
- **Indexed Ground Area**: **~868.0 km²**
- **Sensor Platform**: Sentinel-2 MSI Level-2A (10m Resolution, 4 Spectral Bands)
- **Temporal Span**: 2020 to 2024 (6 Seasonal Epochs)
- **Total Valid Tile Chips**: **180** (100% clean full-frame 512×512 tiles, zero black padding)

---

## 2. Model & Index Performance
- **Embedding Architecture**: OpenCLIP ViT-B-32 (512-dimensional multimodal latent space)
- **Vector Index**: FAISS `IndexFlatIP` (Exact Inner Product / Cosine Similarity)
- **Mean Query Latency**: **88.28 ms** (Sub-second retrieval)
- **Median Query Latency**: **74.81 ms**

---

## 3. Storage Footprint
- **Tile Imagery**: 271.7 MB
- **Vector Index & Metadata**: 0.5 MB
- **Total Operational Footprint**: **715.23 MB** (Extremely lightweight, deployable on tactical field laptops)

---

## 4. Hardware Environment
- **OS**: Windows 10
- **Processor**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- **RAM**: 15.2 GB
- **Sovereignty**: Complete local operation without external APIs or cloud dependencies
