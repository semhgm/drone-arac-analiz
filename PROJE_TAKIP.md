# Drone Görüntü Analizi Projesi — Takip Dosyası

> Görüntü İşleme Dersi | Final-year proje | Deadline: ~2 hafta
> Konu: Drone görüntülerinden araç tespiti, sayımı ve yoğunluk analizi

---

## 🎯 Proje Özeti

Drone (UAV) görüntülerinden:
1. **Araç tespiti** (YOLOv8 fine-tune, VisDrone)
2. **Sayım** (kaç araç, sınıf bazında)
3. **Yoğunluk haritası** (heatmap — nerede yoğunlaşma var)
4. **Basit arayüz** (Streamlit: görüntü yükle → analiz et → sonuç göster)

LLM entegrasyonu: opsiyonel, zaman kalırsa eklenir.

---

## 🛠️ Teknoloji Stack

- Python 3.x + venv
- `ultralytics` (YOLOv8)
- `opencv-python` (görüntü işleme, heatmap)
- `streamlit` (arayüz)
- Veri seti: **VisDrone** (ultralytics otomatik indirir)
- Donanım: MacBook Air M4 (MPS / Apple Silicon GPU)

---

## 📋 Yapılacaklar (TODO)

### Aşama 0 — Ortam Kurulumu
- [ ] Yeni proje klasörü + venv
- [ ] Gerekli paketler kurulumu
- [ ] MPS (Apple GPU) çalışıyor mu testi

### Aşama 1 — Veri Seti & Baseline
- [ ] VisDrone.yaml ile veri setini otomatik indir
- [ ] Hazır YOLOv8 modeliyle (pretrained) ilk inference testi
- [ ] Sonuçları gözlemle

### Aşama 2 — Fine-Tune
- [ ] YOLOv8n/s ile VisDrone üzerinde eğitim
- [ ] Araç sınıflarına odak (car, van, truck, bus)
- [ ] Eğitim metriklerini (mAP) kaydet

### Aşama 3 — Analiz Pipeline
- [ ] Tespit + bounding box çizimi
- [ ] Sınıf bazında sayım
- [ ] Yoğunluk haritası (heatmap)

### Aşama 4 — Arayüz
- [ ] Streamlit: görüntü yükle
- [ ] Analiz sonuçlarını göster (görsel + sayılar + heatmap)

### Aşama 5 — Rapor & Teslim
- [ ] Sonuçların ekran görüntüleri
- [ ] Kısa rapor / sunum

---

## 📌 İlerleme Günlüğü (nereye kadar geldik)

| Tarih | Aşama | Ne yapıldı | Not |
|-------|-------|------------|-----|
| 29.05.2026 | Planlama | Stack ve yol haritası belirlendi | LLM opsiyonel bırakıldı |

---

## 💡 Notlar & Kararlar

- Veri seti Telegram yerine **VisDrone** (akademik, etiketli, kabul görmüş)
- VisDrone YOLO formatına ultralytics tarafından otomatik çevriliyor
- İlk eğitim için YOLOv8n (nano) → hızlı; sonra v8s denenebilir
