import tempfile
from pathlib import Path

import cv2
import requests
import streamlit as st

from analyze import analyze, VEHICLE_CLASSES

# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drone Araç Analizi",
    page_icon="🚁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.title("🚁 Drone Araç Analizi")
st.markdown("YOLOv8 fine-tuned on VisDrone · Araç tespiti, sayım ve yoğunluk haritası")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — ayarlar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")

    model_path = st.text_input(
        "Model yolu (best.pt)",
        value=r"runs\detect\runs\train\visdrone_v1\weights\best.pt",
        help="Fine-tune edilmiş YOLOv8 model dosyası",
    )

    conf_threshold = st.slider(
        "Güven eşiği (confidence)",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Düşük → daha fazla tespit, yüksek → daha temiz sonuç",
    )

    st.divider()
    st.markdown("**🤖 DeepSeek Ayarları**")
    use_deepseek = st.toggle("DeepSeek raporu oluştur", value=True)
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

    st.divider()
    st.markdown("**Sınıf renk kodları**")
    for label in ["🟢 Car", "🟠 Van", "🔵 Truck", "🟣 Bus", "🩵 Motor", "🟡 Bicycle"]:
        st.markdown(f"- {label}")

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

        # <think>...</think> bloklarını temizle
        import re
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
        with tempfile.TemporaryDirectory() as out_dir:
            counts, det_path, heat_path = analyze(
                image_path=tmp_path,
                model_path=model_path,
                conf=conf_threshold,
                output_dir=out_dir,
            )
            det_img  = cv2.cvtColor(cv2.imread(det_path),  cv2.COLOR_BGR2RGB)
            heat_img = cv2.cvtColor(cv2.imread(heat_path), cv2.COLOR_BGR2RGB)

    except FileNotFoundError as e:
        st.error(f"❌ Hata: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📊 Tespit Sonuçları")

total = sum(counts.values())
active = {k: v for k, v in counts.items() if v > 0}

metric_cols = st.columns(len(active) + 1)
metric_cols[0].metric("🚗 Toplam Araç", total)
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

with col_heat:
    st.markdown("#### 🌡️ Yoğunluk Haritası")
    st.image(heat_img, use_container_width=True)

# ---------------------------------------------------------------------------
# Tablo
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📋 Sınıf Bazında Dağılım")

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
    st.subheader("🤖 DeepSeek Traffic Report")
    with st.spinner("DeepSeek analiz yazıyor…"):
        report = generate_report(counts, total, ollama_url)
    st.info(report)

st.caption("YOLOv8n · VisDrone fine-tune · 15 epoch · mAP50@car: 0.556 · DeepSeek-R1 7B")