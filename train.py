"""
YOLOv8n — VisDrone Fine-Tune Script
====================================
Drone görüntülerinden araç tespiti için fine-tuning.

VisDrone sınıfları (10 adet):
  0: pedestrian  1: people     2: bicycle   3: car
  4: van         5: truck      6: tricycle  7: awning-tricycle
  8: bus         9: motor

Strateji:
  - İlk çalıştırmada VisDrone (~2.3 GB) otomatik indirilir
  - Kısa eğitim (20 epoch) → demo için yeterli, M4'te ~30-45 dk
  - Sadece araç sınıflarını takip ediyoruz ama tüm sınıflarla eğitiyoruz
    (VisDrone.yaml değiştirilmeden kullanılır, filtreleme inference'ta yapılır)
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

# ─── Ayarlar ──────────────────────────────────────────────────────────────────
MODEL_BASE   = "yolov8n.pt"      # Pretrained başlangıç ağırlıkları
DATA_YAML    = "VisDrone.yaml"   # Ultralytics otomatik indirir + dönüştürür
EPOCHS     = 15
DEVICE = "cuda"      # "mps" yerine
IMAGE_SIZE = 416
BATCH_SIZE = 8  
PROJECT_DIR  = "runs/train"      # Sonuçlar buraya kaydedilir
RUN_NAME     = "visdrone_v1"     # runs/train/visdrone_v1/

# Araç sınıfları (analiz aşamasında bu ID'leri kullanacağız)
VEHICLE_CLASSES = {
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    8: "bus",
    9: "motor",
}

# ─── Eğitim ───────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("VisDrone Fine-Tune Başlıyor")
    print(f"  Model   : {MODEL_BASE}")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  Device  : {DEVICE}")
    print("=" * 60)
    print()
    print("ℹ️  İlk çalıştırmada VisDrone (~2.3 GB) indirilecek.")
    print("   Lütfen bekle...\n")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        # Performans / M4 optimizasyonları
        workers=4,           # Veri yükleme thread sayısı
        cache=False,         # RAM'e cache etme (2.3 GB büyük olabilir)
        patience=10,         # Early stopping: 10 epoch iyileşme yoksa dur
        save=True,           # En iyi model kaydedilsin
        save_period=5,       # Her 5 epoch'ta checkpoint
        exist_ok=True,       # Aynı isimli run varsa üstüne yaz
        verbose=True,
    )

    print()
    print("=" * 60)
    print("✅ Eğitim tamamlandı!")
    print(f"   Sonuçlar : {PROJECT_DIR}/{RUN_NAME}/")
    print(f"   En iyi model : {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print()

    # mAP özetini yazdır
    metrics = results  # train() sonucu metrics objesi döner
    try:
        print(f"   mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    except Exception:
        print("   (Metrikler runs/train/visdrone_v1/results.csv dosyasında)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()

