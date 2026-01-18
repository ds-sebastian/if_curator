<div align="center">

# 🖼️ if-curator
### Immich to Frigate Curator

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Immich](https://img.shields.io/badge/Immich-v1.106%2B-violet?style=for-the-badge)](https://immich.app)
[![Frigate](https://img.shields.io/badge/Frigate-Ready-green?style=for-the-badge)](https://frigate.video)

*A specialized tool to extract **high-quality, diverse** training images from your Immich library for Frigate's Face Recognition (ArcFace) and Object/State Classification models.*

</div>


---

> [!WARNING]  
> This is 100% vibe-coded.





## ⚡ Why This Tool?

> **"Diversity matters far more than volume."** — *Frigate Developer Tips*

Training AI models on "bulk" data is often harmful. If you feed the model 50 images from the same 10-second video clip, it learns to recognize the *lighting and background*, not the actual *face* or *object*.

`if-curator` solves this using **AI-powered diversity selection**:

| Mode | Embedding Model | Algorithm |
| :--- | :--- | :--- |
| **👤 Face** | InsightFace (ArcFace) | Farthest Point Sampling |
| **🐶 Object** | SigLIP (Vision Transformer) | Farthest Point Sampling |

Both use the same **Farthest Point Sampling (FPS)** algorithm that mathematically selects images until redundancy starts, ensuring optimal diversity whether that's 20 or 150 images.

> [!WARNING]
> **Regarding Object Classification**
>
> Frigate **does not support** uploading custom images for object classification training via the UI or API.
> This tool currently prepares the dataset (crops and categorizes images) for training external models (like YOLO) manually.

---

## ✨ Features

### 🎯 Unified Selection Strategies
Both Face and Object modes offer the same powerful options:
- **Auto (Objective Diversity) [Recommended]**: Dynamically selects images until redundancy starts
- **Standard (30 images)**: Balanced set using Smart Diversity
- **Broad (100 images)**: Extensive set using Smart Diversity
- **Custom Count**: You choose the limit

### 👤 Face Recognition Prep
- Uses **InsightFace** (ArcFace/Buffalo_L) embeddings
- Extracts faces using Immich's metadata
- **Auto-Diversity** picks the optimal set size based on visual distinctness

### 📦 Object/State Classification Prep
- Uses **SigLIP** (Vision Transformer) embeddings for semantic diversity
- **YOLOv9c** to detect and crop specific objects (dogs, cars, etc.)
- Captures variation in poses, lighting, and backgrounds
- *Note: As mentioned, Frigate upload is pending support.*

---

## 🚀 Installation

### Prerequisites
- **Python 3.12+**
- **[uv](https://astral.sh/uv/)** (highly recommended)
- **Immich Server** (v1.106+)

### Setup

```bash
git clone <repository_url>
cd if-curator
uv sync
```

### 🏎️ GPU Support (Recommended)
For faster embedding computation, install with GPU extras:

```bash
uv sync --extra gpu
```
*Automatically detects CUDA (NVIDIA), ROCm (AMD), or MPS (macOS).*

---

## 💻 Usage

Run the command-line interface:

```bash
uv run -m if_curator
```

### Interactive Flow
The tool will guide you through:
1.  **Select Person/Subject**: Choose from your Immich people.
2.  **Training Mode**: Face (Recognition) or Object (Classification).
3.  **Strategy**: Auto, Standard, Broad, etc.

```text
Using SigLIP (visual embeddings) for diversity analysis...
Computing embeddings for 69 images... ████████████████ 100%
Auto-diversity selected 38 optimally diverse images.
```

---

## 🛠️ Configuration

The tool prompts for your Immich URL and API Key on the first run and saves them to `.immich_config.json`.

| Variable | Description |
| :--- | :--- |
| `IMMICH_URL` | Full URL to Immich (e.g. `http://192.168.1.10:2283`) |
| `API_KEY` | Your Immich API Key |
| `FORCE_CPU` | Set to `true` to disable GPU acceleration |

---

## 🧠 Technical Details

- **InsightFace**: Face detection and embedding (ArcFace)
- **SigLIP**: Visual embeddings via `transformers` (OpenAI CLIP alternative)
- **YOLOv9c**: State-of-the-art object detection for cropping
- **Rich**: Beautiful terminal UI

