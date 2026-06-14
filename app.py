import tempfile
import re
from pathlib import Path

import cv2
import requests
import streamlit as st

from analyze import analyze, VEHICLE_CLASSES

st.set_page_config(
    page_title="Drone Araç Analizi",
    page_icon="🚁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern Dark Dashboard UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Ana arka plan */
.stApp {
    background: #0a0c0f !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1115 !important;
    border-right: 1px solid #1e2229 !important;
}
[data-testid="stSidebar"] * {
    color: #a0a8b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Sidebar input alanları */
[data-testid="stSidebar"] input {
    background: #1a1d24 !important;
    border: 1px solid #2a2f3a !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: #1e6fff !important;
}

/* Toggle */
[data-testid="stToggle"] span {
    background: #1e6fff !important;
}

/* Divider */
hr {
    border-color: #1e2229 !important;
}

/* Başlık alanı */
.main-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2229;
    margin-bottom: 2rem;
}
.main-title {
    font-size: 28px;
    font-weight: 600;
    color: #f0f4ff;
    letter-spacing: -0.02em;
    margin: 0;
}
.main-subtitle {
    font-size: 13px;
    color: #4a5568;
    margin: 6px 0 0 0;
    font-family: 'DM Mono', monospace;
}

/* Metric kartları */
[data-testid="stMetric"] {
    background: #0f1318 !important;
    border: 1px solid #1e2229 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #4a5568 !important;
}
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 600 !important;
    color: #f0f4ff !important;
}

/* Toplam metrik özel */
[data-testid="stMetric"]:first-child {
    border-color: #1e4fff44 !important;
    background: #0a1128 !important;
}
[data-testid="stMetric"]:first-child [data-testid="stMetricValue"] {
    color: #4d9fff !important;
}

/* Görsel kartlar */
.img-card {
    background: #0f1318;
    border: 1px solid #1e2229;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.img-card-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5568;
    margin-bottom: 1rem;
    font-weight: 500;
}

/* Görseller */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #1e2229 !important;
}

/* Tablo */
[data-testid="stTable"] table {
    background: #0f1318 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #1e2229 !important;
    font-size: 13px !important;
}
[data-testid="stTable"] th {
    background: #171c24 !important;
    color: #4a5568 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid #1e2229 !important;
    padding: 12px 16px !important;
}
[data-testid="stTable"] td {
    color: #c8d0e0 !important;
    border-bottom: 1px solid #131820 !important;
    padding: 10px 16px !important;
}
[data-testid="stTable"] tr:last-child td {
    border-bottom: none !important;
}

/* Ana buton */
[data-testid="stButton"] > button[kind="primary"] {
    background: #1e6fff !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #3d80ff !important;
    transform: translateY(-1px) !important;
}

/* İndirme butonu */
[data-testid="stDownloadButton"] > button {
    background: #141920 !important;
    border: 1px solid #1e2229 !important;
    border-radius: 8px !important;
    color: #6b7a99 !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    padding: 0.4rem 0.8rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #1e6fff !important;
    color: #4d9fff !important;
    background: #0a1128 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f1318 !important;
    border: 1px dashed #1e2229 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #1e6fff88 !important;
}

/* Info box (DeepSeek raporu) */
[data-testid="stAlert"] {
    background: #0c1420 !important;
    border: 1px solid #1e3a6e !important;
    border-radius: 12px !important;
    color: #7aaddd !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #1e6fff !important;
}

/* Caption */
[data-testid="stCaptionContainer"] {
    color: #2a3040 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* Section başlıkları */
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
}

/* Warning mesajı */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-color: #3a2a1e !important;
    background: #1a1008 !important;
    color: #c8a87a !important;
}

/* Genel text */
p, label, span {
    color: #8892a4 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <p class="main-title">🚁 Drone Araç Analizi</p>
    <p class="main-subtitle">YOLOv8n · VisDrone fine-tune · v2 model · mAP50@car 0.718</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")

    model_path = st.text_input(
        "Model yolu",
        value=r"runs\detect\runs\train\visdrone_v2\weights\best.pt",
        help="Fine-tune edilmiş YOLOv8 model dosyası",
    )

    conf_threshold = st.slider(
        "Güven eşiği",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Düşük → daha fazla tespit, yüksek → daha temiz sonuç",
    )

    st.divider()
    st.markdown("**DeepSeek**")
    use_deepseek = st.toggle("Rapor oluştur", value=True)
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

    st.divider()
    st.markdown("**Sınıf renkleri**")
    for label in ["🟢 Car", "🟠 Van", "🔵 Truck", "🟣 Bus", "🩵 Motor", "🟡 Bicycle"]:
        st.markdown(f"<span style='font-size:13px'>{label}</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DeepSeek rapor fonksiyonu
# ---------------------------------------------------------------------------
def generate_report(counts: dict, total: int, ollama_url: str) -> str:
    active = {k: v for k, v in counts.items() if v > 0}
    siniflar = ", ".join([f"{k}: {v}" for k, v in active.items()])

    prompt = f"""You are a traffic analyst. Write a short professional traffic report (3-4 sentences) based on drone image analysis results:

Total vehicles detected: {total}
Breakdown: {siniflar}

Cover: overall traffic density, dominant vehicle type and its implication, one brief recommendation."""

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 512},
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return clean if clean else raw

    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama bağlantısı kurulamadı. `ollama serve` çalışıyor mu?"
    except Exception as e:
        return f"⚠️ DeepSeek hatası: {e}"

# ---------------------------------------------------------------------------
# Görüntü yükleme
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drone görüntüsü yükle",
    type=["jpg", "jpeg", "png"],
    help="JPG veya PNG formatında drone/UAV görüntüsü",
)

if uploaded_file is None:
    st.info("👆 Bir drone görüntüsü yükleyerek başlayın.")
    st.stop()

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_analysis = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

if not run_analysis:
    st.image(uploaded_file, caption="Yüklenen görüntü", use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# Analiz
# ---------------------------------------------------------------------------
with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

with st.spinner("Analiz yapılıyor…"):
    try:
        out_dir_obj = tempfile.TemporaryDirectory()
        out_dir = out_dir_obj.name

        counts, det_path, heat_path = analyze(
            image_path=tmp_path,
            model_path=model_path,
            conf=conf_threshold,
            output_dir=out_dir,
        )
        det_img  = cv2.cvtColor(cv2.imread(det_path),  cv2.COLOR_BGR2RGB)
        heat_img = cv2.cvtColor(cv2.imread(heat_path), cv2.COLOR_BGR2RGB)

        _, det_buf  = cv2.imencode(".jpg", cv2.imread(det_path))
        _, heat_buf = cv2.imencode(".jpg", cv2.imread(heat_path))
        det_bytes  = det_buf.tobytes()
        heat_bytes = heat_buf.tobytes()

    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
st.divider()

total = sum(counts.values())
active = {k: v for k, v in counts.items() if v > 0}

metric_cols = st.columns(len(active) + 1)
metric_cols[0].metric("Toplam Araç", total)
for i, (label, cnt) in enumerate(active.items(), start=1):
    metric_cols[i].metric(label.capitalize(), cnt)

# ---------------------------------------------------------------------------
# Görseller
# ---------------------------------------------------------------------------
st.divider()
col_det, col_heat = st.columns(2)

with col_det:
    st.markdown("#### 🎯 Araç Tespiti")
    st.image(det_img, use_container_width=True)
    st.download_button(
        label="↓ tespit görselini indir",
        data=det_bytes,
        file_name=f"tespit_{uploaded_file.name}",
        mime="image/jpeg",
        use_container_width=True,
    )

with col_heat:
    st.markdown("#### 🌡️ Yoğunluk Haritası")
    st.image(heat_img, use_container_width=True)
    st.download_button(
        label="↓ heatmap indir",
        data=heat_bytes,
        file_name=f"heatmap_{uploaded_file.name}",
        mime="image/jpeg",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Tablo
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Sınıf Bazında Dağılım")

if active:
    st.table({
        "Sınıf":    list(active.keys()),
        "Adet":     list(active.values()),
        "Oran (%)": [f"{v/total*100:.1f}%" for v in active.values()],
    })
else:
    st.warning("Hiç araç tespit edilemedi. Güven eşiğini düşürmeyi deneyin.")

# ---------------------------------------------------------------------------
# DeepSeek raporu
# ---------------------------------------------------------------------------
if use_deepseek and total > 0:
    st.divider()
    st.subheader("DeepSeek Traffic Report")
    with st.spinner("DeepSeek analiz yazıyor…"):
        report = generate_report(counts, total, ollama_url)
    st.info(report)

st.caption("YOLOv8n · VisDrone fine-tune · 30 epoch · mAP50: 0.285 · mAP50@car: 0.718 · DeepSeek-R1 7B")