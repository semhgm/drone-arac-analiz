# 🚁 Drone Araç Analizi

> Görüntü İşleme Dersi — Final Projesi  
> Drone (UAV) görüntülerinden araç tespiti, sayım ve yoğunluk haritası

---

## 📌 Proje Özeti

YOLOv8n modeli VisDrone veri seti üzerinde fine-tune edilerek drone görüntülerinden araç tespiti, sınıf bazında sayım ve yoğunluk haritası üretimi yapılmaktadır. Streamlit arayüzü üzerinden görüntü yüklenip analiz sonuçları görselleştirilebilmekte; DeepSeek-R1 (Ollama) entegrasyonu ile otomatik trafik raporu oluşturulmaktadır.

---

## 🛠️ Teknoloji Stack

| Katman | Teknoloji |
|--------|-----------|
| Model | YOLOv8n (ultralytics) |
| Veri seti | VisDrone2019-DET |
| Görüntü işleme | OpenCV |
| Arayüz | Streamlit |
| LLM | DeepSeek-R1 7B (Ollama) |
| Geliştirme | MacBook Air M4 (MPS) |
| Eğitim | HP Victus — RTX 4050, CUDA 12.1 |

---

## 📁 Klasör Yapısı

```
drone-arac-analiz/
├── analyze.py          # Tespit + sayım + heatmap pipeline
├── app.py              # Streamlit arayüzü
├── train.py            # YOLOv8 fine-tune scripti
├── test_inference.py   # Pretrained model testi
├── test_mps.py         # Apple GPU testi
├── requirements.txt
├── .gitignore
└── output/             # Analiz çıktıları (gitignore'd)
```

---

## 🚀 Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/kullanici/drone-arac-analiz.git
cd drone-arac-analiz

# Sanal ortam
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt
```

---

## 🏋️ Model Eğitimi

```bash
python train.py
```

**Eğitim parametreleri (v1):**
- Model: YOLOv8n
- Epochs: 15 | Image size: 416 | Batch: 8
- Dataset: VisDrone (6471 train, 548 val)
- Device: RTX 4050 (CUDA) | Süre: ~20 dk

**v1 Sonuçları:**

| Metrik | Değer |
|--------|-------|
| mAP50 (genel) | 0.159 |
| mAP50 (car) | 0.556 |
| mAP50-95 | 0.084 |

---

## 🔍 Kullanım

### Komut satırı

```bash
python analyze.py --image yol/goruntu.jpg --model runs/.../best.pt
```

### Streamlit arayüzü

```bash
# Ollama çalışıyor olmalı (DeepSeek raporu için)
ollama serve

streamlit run app.py
```

Arayüzde:
1. Drone görüntüsü yükle (JPG/PNG)
2. Confidence eşiğini ayarla
3. "Analiz Et" butonuna bas
4. Tespit görseli, heatmap ve DeepSeek raporu görüntülenir

---

## 🎯 Tespit Edilen Sınıflar

| ID | Sınıf | Renk |
|----|-------|------|
| 3 | car | 🟢 Yeşil |
| 4 | van | 🟠 Turuncu |
| 5 | truck | 🔵 Mavi |
| 8 | bus | 🟣 Mor |
| 9 | motor | 🩵 Açık mavi |
| 2 | bicycle | 🟡 Sarı |

---

## 🤖 DeepSeek Entegrasyonu

[Ollama](https://ollama.com) üzerinden yerel olarak çalışan DeepSeek-R1 7B modeli, tespit sonuçlarını otomatik olarak analiz edip İngilizce trafik raporu üretir.

```bash
# DeepSeek modelini indir
ollama pull deepseek-r1:7b
```

---

## 📊 Örnek Çıktılar

- **Araç tespiti:** Sınıf bazında renkli bounding box'lar
- **Yoğunluk haritası:** JET colormap overlay (mavi→kırmızı yoğunluk)
- **Trafik raporu:** DeepSeek-R1 ile otomatik oluşturulan metin analizi