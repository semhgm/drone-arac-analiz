# 🚁 Drone Görüntü Analizi — Araç Tespiti & Yoğunluk Analizi

Drone (UAV) görüntülerinden **araç tespiti**, **sınıf bazında sayım** ve **yoğunluk haritası (heatmap)** üreten bir görüntü işleme uygulaması. Görüntü İşleme dersi kapsamında geliştirilmiştir.

---

## ✨ Özellikler

- 🎯 **Araç Tespiti** — YOLOv8 (VisDrone üzerinde fine-tune) ile yukarıdan bakış araç tespiti
- 🔢 **Sayım** — Tespit edilen araçların sınıf bazında (car, van, truck, bus) sayımı
- 🔥 **Yoğunluk Haritası** — Araç yoğunluğunun ısı haritası ile görselleştirilmesi
- 🖥️ **Arayüz** — Streamlit tabanlı: görüntü yükle → analiz et → sonuçları gör

---

## 🛠️ Teknoloji

| Bileşen | Kullanım |
|---------|----------|
| YOLOv8 (ultralytics) | Nesne tespiti & eğitim |
| OpenCV | Görüntü işleme, heatmap |
| Streamlit | Web arayüzü |
| VisDrone | Eğitim veri seti |
| PyTorch (MPS) | Apple Silicon GPU hızlandırma |

---

## 🚀 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/<kullanici-adi>/drone-arac-analiz.git
cd drone-arac-analiz

# Sanal ortam
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt
```

---

## 📦 Kullanım

### 1. Hızlı tespit testi
```bash
python test_inference.py
```

### 2. Model eğitimi (fine-tune)
```bash
python train.py
```

### 3. Arayüzü başlat
```bash
streamlit run app.py
```

---

## 📁 Proje Yapısı

```
drone-arac-analiz/
├── README.md
├── requirements.txt
├── test_inference.py     # Pretrained model ile hızlı test
├── train.py              # VisDrone fine-tune
├── analyze.py            # Tespit + sayım + heatmap pipeline
├── app.py                # Streamlit arayüzü
└── runs/                 # Eğitim çıktıları (model ağırlıkları, metrikler)
```

---

## 📊 Veri Seti

[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) — Tianjin Üniversitesi AISKYEYE ekibi tarafından oluşturulan, drone tabanlı tespit/takip için büyük ölçekli açık veri seti. Ultralytics tarafından otomatik indirilip YOLO formatına çevrilir.

---

## 📝 Lisans

Eğitim/akademik amaçlı geliştirilmiştir.