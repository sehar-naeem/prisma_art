# ✏️ PrismaArt Studio — AI Pencil Sketch, Comic & Chroma Pop-Art Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PrismaArt Studio** is an interactive AI & Computer Vision web application that transforms standard digital photographs into fine art, including **Charcoal Pencil Sketches**, **Detailed Ink Pen Drawings**, **Comic Book / Manga Anime Art**, **ChromaCluster (K-Means Pop-Art)**, and **Vintage Film Noir**.

Built by fusing classical mathematical image processing (Gaussian Convolutions, Color Dodge Blending, Difference of Gaussians (DoG), Sobel Gradient Operators) with Unsupervised Machine Learning (**K-Means Vector Quantization**) using NumPy, SciPy, Scikit-Learn, Pillow, and Streamlit.

---

## ✨ Artistic Modes

- ✏️ **Charcoal Pencil Sketch**: Uses inverse Gaussian blurring and mathematical **Color Dodge** blending to replicate hand-sketched pencil pressure and graphite shading.
- 🖋️ **Detailed Ink Pen Drawing**: Extracts high-contrast contour boundaries using **Sobel spatial gradient operators** and percentile-adaptive thresholding.
- 💥 **Comic Book / Anime Art**: Features a two-pass **Difference of Gaussians (DoG)** ink-line extractor coupled with edge-preserving median smoothing for smooth cel-shaded graphic novel artwork.
- 🎨 **ChromaCluster (K-Means Pop-Art)**: Performs **Unsupervised K-Means Color Quantization** to reduce millions of colors into $K$ optimal centroid clusters, with an optional Andy Warhol silk-screen ink outline and live hex color swatch pills.
- 🎞️ **Vintage Film Noir**: Emulates 1940s Hollywood cinema with non-linear gamma contrast mapping and Gaussian silver-halide film grain noise.

---

## 🔬 AI & Computer Vision Principles Used

1. **Grayscale Intensity Mapping**: Conversion of RGB color space to single-channel luminous intensity ($Y = 0.299R + 0.587G + 0.114B$).
2. **Color Dodge Algorithm**:
   $$\text{Sketch}(x, y) = \min\left(255, \frac{\text{Gray}(x, y) \times 255}{255 - \text{BlurredInvert}(x, y)}\right)$$
3. **Difference of Gaussians (DoG) for Line Inking**:
   $$\text{DoG}(x, y) = G_{\sigma_1}(x, y) - \gamma \cdot G_{\sigma_2}(x, y)$$
4. **Unsupervised K-Means Vector Quantization**:
   $$J = \sum_{k=1}^K \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$
   Groups millions of 3D RGB pixel vectors into $K$ centroid colors, creating retro poster art while preserving structural color boundaries.
5. **Edge-Preserving Cel-Shading**: Multi-pass median filtering that flattens interior textures (skin, fabric) while protecting silhouette lines.

---

## 🚀 Quickstart Guide

### 1. Clone & Navigate
```bash
git clone https://github.com/<your-username>/PrismaArt-AI-Studio.git
cd PrismaArt-AI-Studio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) (or port 8502) in your browser.

---

## 👩‍💻 Author
Crafted with ❤️ by **Sehar Naeem** — AI / ML Intern at Inovegen.
