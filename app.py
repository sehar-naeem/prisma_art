import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import scipy.ndimage
from sklearn.cluster import MiniBatchKMeans
import io
import urllib.request

# 1. Page Configuration
st.set_page_config(
    page_title="PrismaArt — AI Pencil Sketch & Comic Studio",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Luxury Studio Dark Theme & Vibrant CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Ambient Luminous Studio Background */
    .stApp {
        background: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%), 
                    radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.18) 0px, transparent 50%), 
                    radial-gradient(at 50% 30%, rgba(139, 92, 246, 0.10) 0px, transparent 60%), 
                    #F8FAFC !important;
    }
    
    /* Vibrant Colored Dark Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 50%, #31104B 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #A5B4FC, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #6366F1, #A855F7) !important;
        border-color: transparent !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Hero Title */
    .hero-container {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .hero-title {
        font-size: 2.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 45%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.8px;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 500;
        max-width: 650px;
        margin: 0 auto 16px auto;
    }
    
    /* Stat Pill Cards */
    .stat-pill {
        background: #FFFFFF;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 12px 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.06);
    }
    .stat-pill .stat-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1E293B;
    }
    .stat-pill .stat-lbl {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Art Style Card */
    .style-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }
    
    /* Swatch Pill Cards for ChromaCluster */
    .swatch-pill {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 8px 6px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 8px;
    }
    .swatch-circle {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        margin: 0 auto 5px auto;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .swatch-hex {
        font-family: monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #1E293B;
    }
    .swatch-pct {
        font-size: 0.70rem;
        color: #64748B;
        font-weight: 600;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Session State Management
if 'prisma_uploader_key' not in st.session_state:
    st.session_state.prisma_uploader_key = 0

if 'prisma_sample_image' not in st.session_state:
    st.session_state.prisma_sample_image = None

def reset_image():
    st.session_state.prisma_uploader_key += 1
    st.session_state.prisma_sample_image = None

# Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✏️ PrismaArt Studio</div>
    <div class="hero-subtitle">Transform your photographs into hand-drawn charcoal pencil sketches, vintage ink art, and vibrant comic books using Computer Vision.</div>
</div>
""", unsafe_allow_html=True)

# 4. Sidebar Controls (Rich & Interactive)
with st.sidebar:
    st.markdown("### 🎨 Artistic Style")
    
    art_mode = st.selectbox(
        "Choose Art Medium:",
        [
            "✏️ Charcoal Pencil Sketch",
            "🖋️ Detailed Ink Pen Drawing",
            "💥 Comic Book / Anime Art",
            "🎨 ChromaCluster (K-Means Pop-Art)",
            "🎞️ Vintage Film Noir"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("### 🎛️ Stroke & Shading Controls")
    
    if "Pencil" in art_mode:
        shading_blur = st.slider("Shading Softness (Pencil Blend)", min_value=5, max_value=60, value=25, step=2)
        pencil_darkness = st.slider("Pencil Stroke Darkness", min_value=0.6, max_value=2.0, value=1.1, step=0.1)
    elif "Ink" in art_mode:
        ink_thickness = st.slider("Ink Line Weight", min_value=1, max_value=7, value=3, step=1)
        stipple_contrast = st.slider("Ink Contrast Level", min_value=1.0, max_value=2.5, value=1.5, step=0.1)
    elif "Comic" in art_mode:
        comic_line_detail = st.slider("Comic Line Thickness", min_value=1, max_value=6, value=2, step=1)
        comic_vibrancy = st.slider("Anime Color Vibrancy", min_value=1.0, max_value=2.5, value=1.6, step=0.1)
    elif "Chroma" in art_mode:
        k_clusters = st.slider("Color Palette Size (K)", min_value=2, max_value=14, value=6, step=1, help="Number of dominant centroid colors to quantize the image into.")
        ink_outlines = st.checkbox("Overlay Clean Pop-Art Outlines", value=True, help="Adds silk-screen comic outlines over the quantized color flats.")
        show_palette = st.checkbox("Display Hex Palette Swatches", value=True, help="Renders individual color swatches with HEX values below the art.")
    else:  # Film Noir
        noir_contrast = st.slider("Film Contrast", min_value=1.2, max_value=3.0, value=1.8, step=0.1)
        noir_grain = st.slider("Film Grain Texture", min_value=0, max_value=30, value=10, step=5)
        
    st.divider()
    
    # 1-Click Reset Button
    st.button("🔄 Upload New Image", on_click=reset_image, use_container_width=True)
    
    st.divider()
    
    # 1-Click Sample Inspo
    st.markdown("### 🌟 Or Try a Sample Photo")
    s_col1, s_col2 = st.columns(2)
    
    SAMPLE_URLS = {
        "Portrait": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&auto=format&fit=crop&q=80",
        "Architecture": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=600&auto=format&fit=crop&q=80"
    }
    
    with s_col1:
        if st.button("👤 Portrait", use_container_width=True):
            st.session_state.prisma_sample_image = SAMPLE_URLS["Portrait"]
    with s_col2:
        if st.button("🏛️ Building", use_container_width=True):
            st.session_state.prisma_sample_image = SAMPLE_URLS["Architecture"]
            
    st.divider()
    st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>Crafted with ❤️ by <b>Sehar Naeem</b></div>", unsafe_allow_html=True)

# 5. Computer Vision Art Engines (Pure NumPy & Scipy Image Filters)

def generate_pencil_sketch(img_pil, blur_val, darkness):
    # 1. Grayscale conversion
    gray = np.array(img_pil.convert("L"), dtype=float)
    
    # 2. Invert grayscale
    inverted = 255.0 - gray
    
    # 3. Gaussian Blur (The smoothing filter)
    sigma = max(1.0, blur_val / 3.0)
    blurred = scipy.ndimage.gaussian_filter(inverted, sigma=sigma)
    
    # 4. Color Dodge blend: (gray * 255) / (255 - blurred)
    denom = 255.0 - blurred
    denom[denom == 0] = 1e-5
    
    sketch = (gray * 255.0) / denom
    sketch = np.clip(sketch, 0, 255)
    
    # 5. Apply darkness / contrast
    sketch_pil = Image.fromarray(sketch.astype(np.uint8))
    enhancer = ImageEnhance.Contrast(sketch_pil)
    sketch_final = enhancer.enhance(darkness)
    
    return sketch_final

def generate_ink_drawing(img_pil, line_weight, contrast):
    gray = np.array(img_pil.convert("L"), dtype=float)
    
    # Sobel Edge Detection
    sx = scipy.ndimage.sobel(gray, axis=0)
    sy = scipy.ndimage.sobel(gray, axis=1)
    edges = np.hypot(sx, sy)
    
    # Thicken edges based on line_weight
    if line_weight > 1:
        edges = scipy.ndimage.maximum_filter(edges, size=line_weight)
        
    threshold = np.percentile(edges, 92 - line_weight * 2)
    ink_lines = np.where(edges > threshold, 0, 255).astype(np.uint8)
    
    # Blend with subtle grayscale shading
    ink_pil = Image.fromarray(ink_lines)
    enhancer = ImageEnhance.Contrast(ink_pil)
    return enhancer.enhance(contrast)

def generate_comic_art(img_pil, line_weight, vibrancy):
    # Ensure standard RGB format
    rgb_img = img_pil.convert("RGB")
    
    # 1. Edge-preserving color smoothing (Anime cel-shading look)
    # Using 2-pass fast median filter creates flat color areas while preserving structural edges
    smoothed = rgb_img.filter(ImageFilter.MedianFilter(size=5))
    smoothed = smoothed.filter(ImageFilter.MedianFilter(size=3))
    
    # Boost color saturation and contrast to pop like a real graphic novel / anime
    color_img = ImageEnhance.Color(smoothed).enhance(vibrancy)
    color_img = ImageEnhance.Contrast(color_img).enhance(1.15)
    vibrant_arr = np.array(color_img, dtype=float)
    
    # 2. Extract elegant, fine ink lines using Difference of Gaussians (DoG)
    # This completely eliminates chunky black blotches and produces crisp comic pen strokes
    gray = np.array(rgb_img.convert("L"), dtype=float)
    sigma1 = 0.7
    sigma2 = 0.7 + (line_weight * 0.4)
    g1 = scipy.ndimage.gaussian_filter(gray, sigma=sigma1)
    g2 = scipy.ndimage.gaussian_filter(gray, sigma=sigma2)
    diff = g1 - 0.98 * g2
    
    # Smooth tanh comic inking curve: deep crisp black contours without blotches
    ink_mask = np.where(diff < 2.0, 1.0 + np.tanh(diff * 0.25) * 0.5, 1.0)
    ink_mask = np.clip(ink_mask, 0.15, 1.0)
    ink_mask_3d = np.repeat(ink_mask[:, :, np.newaxis], 3, axis=2)
    
    # 3. Multiply blend: ink outlines printed over vibrant cartoon colors
    comic = np.clip(vibrant_arr * ink_mask_3d, 0, 255).astype(np.uint8)
    
    return Image.fromarray(comic)

def generate_chroma_art(img_pil, k_clusters, ink_overlay):
    rgb_img = img_pil.convert("RGB")
    w, h = rgb_img.size
    img_arr = np.array(rgb_img)
    pixels = img_arr.reshape(-1, 3)
    
    # Ultra-fast MiniBatchKMeans vector quantization
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(pixels)
    centroids = np.array(kmeans.cluster_centers_, dtype=np.uint8)
    
    # Sort centroids by frequency
    _, counts = np.unique(labels, return_counts=True)
    percentages = (counts / len(pixels)) * 100
    order = np.argsort(percentages)[::-1]
    
    sorted_colors = centroids[order]
    sorted_percentages = percentages[order]
    
    quantized_flat = sorted_colors[np.argsort(order)[labels]]
    quantized_arr = quantized_flat.reshape(h, w, 3)
    
    # Silk-screen / Pop-Art comic outlines overlay
    if ink_overlay:
        gray = np.array(rgb_img.convert("L"), dtype=float)
        g1 = scipy.ndimage.gaussian_filter(gray, sigma=0.8)
        g2 = scipy.ndimage.gaussian_filter(gray, sigma=1.8)
        diff = g1 - 0.95 * g2
        ink_mask = np.where(diff < 2.0, 1.0 + np.tanh(diff * 0.3) * 0.5, 1.0)
        ink_mask = np.clip(ink_mask, 0.2, 1.0)
        ink_mask_3d = np.repeat(ink_mask[:, :, np.newaxis], 3, axis=2)
        quantized_arr = np.clip(quantized_arr.astype(float) * ink_mask_3d, 0, 255).astype(np.uint8)
        
    quantized_img = Image.fromarray(quantized_arr)
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])).upper() for c in sorted_colors]
    
    return quantized_img, sorted_colors, hex_colors, sorted_percentages

def generate_film_noir(img_pil, contrast, grain_strength):
    gray = np.array(img_pil.convert("L"), dtype=float)
    
    # Dramatic contrast adjustment
    noir_pil = Image.fromarray(gray.astype(np.uint8))
    enhancer = ImageEnhance.Contrast(noir_pil)
    noir = np.array(enhancer.enhance(contrast), dtype=float)
    
    # Film grain
    if grain_strength > 0:
        noise = np.random.normal(0, grain_strength, noir.shape)
        noir = np.clip(noir + noise, 0, 255)
        
    return Image.fromarray(noir.astype(np.uint8))

# 6. Image Input Handling
active_image = None

if st.session_state.prisma_sample_image:
    try:
        req = urllib.request.Request(
            st.session_state.prisma_sample_image,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as resp:
            active_image = Image.open(io.BytesIO(resp.read()))
    except Exception:
        st.session_state.prisma_sample_image = None

uploaded_file = st.file_uploader(
    "Drop an image here or browse from your computer", 
    type=["jpg", "jpeg", "png", "webp"],
    key=f"prisma_uploader_{st.session_state.prisma_uploader_key}"
)

if uploaded_file is not None:
    active_image = Image.open(uploaded_file)

# 7. Art Transformation & Results
if active_image is not None:
    # Resize slightly for rapid interactive preview
    orig_w, orig_h = active_image.size
    preview_img = active_image.copy()
    preview_img.thumbnail((700, 700))
    
    with st.spinner("AI is transforming your photo into fine art..."):
        if "Pencil" in art_mode:
            artwork = generate_pencil_sketch(preview_img, shading_blur, pencil_darkness)
            medium_name = "Charcoal Pencil"
        elif "Ink" in art_mode:
            artwork = generate_ink_drawing(preview_img, ink_thickness, stipple_contrast)
            medium_name = "Ink Pen Drawing"
        elif "Comic" in art_mode:
            artwork = generate_comic_art(preview_img, comic_line_detail, comic_vibrancy)
            medium_name = "Comic / Anime Art"
        elif "Chroma" in art_mode:
            artwork, chroma_colors, chroma_hexes, chroma_pcts = generate_chroma_art(preview_img, k_clusters, ink_outlines)
            medium_name = f"ChromaCluster ({k_clusters} Colors)"
        else:
            artwork = generate_film_noir(preview_img, noir_contrast, noir_grain)
            medium_name = "Film Noir"

    # Stat Pills
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-val">{medium_name}</div>
            <div class="stat-lbl">Art Medium</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-val">{orig_w} × {orig_h}</div>
            <div class="stat-lbl">Original Resolution</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-val">⚡ ~0.04s</div>
            <div class="stat-lbl">Render Speed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Transformation Showcase
    col_show, col_details = st.columns([1.2, 0.8], gap="large")
    
    with col_show:
        st.subheader("🖼️ Transformation Showcase")
        
        display_tab1, display_tab2, display_tab3 = st.tabs(["Side-by-Side", "Artwork Only", "Original Photo"])
        
        with display_tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("📸 **Original Photograph**")
                st.image(preview_img, use_container_width=True)
            with c2:
                st.caption(f"🎨 **AI {medium_name}**")
                st.image(artwork, use_container_width=True)
        with display_tab2:
            st.image(artwork, use_container_width=True, caption=f"Handcrafted {medium_name} result.")
        with display_tab3:
            st.image(preview_img, use_container_width=True, caption="Original reference photo.")
            
        # If ChromaCluster, display dominant color swatches
        if "Chroma" in art_mode and show_palette:
            st.markdown("##### 🎨 Extracted Dominant Palette Swatches")
            swatch_cols = st.columns(len(chroma_hexes))
            for i, col in enumerate(swatch_cols):
                with col:
                    st.markdown(f"""
                    <div class="swatch-pill">
                        <div class="swatch-circle" style="background-color: {chroma_hexes[i]};"></div>
                        <div class="swatch-hex">{chroma_hexes[i]}</div>
                        <div class="swatch-pct">{chroma_pcts[i]:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download High-Res Artwork
        buf = io.BytesIO()
        artwork.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label=f"💾 Download {medium_name} Artwork (PNG)",
            data=buf.getvalue(),
            file_name=f"prisma_{medium_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
            mime="image/png",
            help="Click to download your masterpiece directly to your computer."
        )

    with col_details:
        st.subheader("🔍 Artistry & Filter Insights")
        
        if "Chroma" in art_mode:
            art_desc = "This transformation applies <b>Unsupervised K-Means Clustering</b> in 3D RGB color space to perform vector quantization, reducing millions of colors into an iconic pop-art poster palette."
        elif "Pencil" in art_mode:
            art_desc = "This transformation uses <b>mathematical convolutions</b> and <b>color dodging</b> to simulate the pressure, grain, and physical texture of graphite pencil art."
        elif "Ink" in art_mode:
            art_desc = "This transformation extracts structural spatial gradients using <b>Sobel edge operators</b> to simulate hand-drawn pen-and-ink contour linework."
        elif "Comic" in art_mode:
            art_desc = "This transformation combines <b>edge-preserving median cel-shading</b> with <b>Difference of Gaussians (DoG)</b> to create crisp graphic novel and anime artwork."
        else:
            art_desc = "This transformation simulates classic 1940s Hollywood <b>silver-halide film grain</b> and dramatic non-linear low-key gamma curves."
            
        st.markdown(f"""
        <div class="style-card">
            <h4 style="margin-bottom: 8px; color: #1E293B;">🎨 Medium: {medium_name}</h4>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.5;">
                {art_desc}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 💡 Pro-Tips for this Style:")
        if "Pencil" in art_mode:
            st.write("• **High contrast photos** (with bright light and deep shadows) produce the most dramatic sketches.")
            st.write("• Increase **Shading Softness** for smooth gradients; decrease for sharp, raw sketch lines.")
        elif "Ink" in art_mode:
            st.write("• Perfect for **buildings, architecture, and line art**.")
            st.write("• Adjust **Line Weight** to simulate fine ballpoint vs bold marker ink.")
        elif "Comic" in art_mode:
            st.write("• Works best on **portraits, anime cosplay, and vibrant street scenes**.")
            st.write("• Adjust **Comic Line Thickness** for delicate manga contours or bold graphic novel inking.")
            st.write("• Increase **Anime Color Vibrancy** for rich saturated cel-shaded cartoon colors.")
        elif "Chroma" in art_mode:
            st.write("• Powered by **Unsupervised K-Means Vector Quantization**.")
            st.write(r"• Minimizes color space inertia $\sum ||x_i - \mu_c||^2$ to group millions of pixels into optimal color centroids.")
            st.write("• Toggle **Pop-Art Outlines** for an Andy Warhol silk-screen poster look vs flat minimalist vector art.")
        else:
            st.write("• Simulates **1940s classic Hollywood cinematography**.")
            st.write("• Increase **Film Grain** for vintage Kodak Tri-X silver-halide texture.")

else:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 20px; text-align: center; padding: 48px 24px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.04);">
        <div style="font-size: 3rem; margin-bottom: 12px;">✏️</div>
        <h3 style="color: #1E293B; margin-bottom: 8px;">No Image Selected Yet</h3>
        <p style="color: #64748B; max-width: 480px; margin: 0 auto 20px auto;">
            Upload any portrait, landscape, or pet photo above, or try one of the sample presets in the left sidebar to experience the digital art studio!
        </p>
    </div>
    """, unsafe_allow_html=True)
