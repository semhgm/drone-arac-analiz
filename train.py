"""
YOLOv8n — VisDrone Fine-Tune Script v2
====================================
Drone görüntülerinden araç tespiti için fine-tuning.

VisDrone sınıfları (10 adet):
  0: pedestrian  1: people     2: bicycle   3: car
  4: van         5: truck      6: tricycle  7: awning-tricycle
  8: bus         9: motor

Strateji:
  - VisDrone zaten indirildi (v1'den kalma)
  - v2: 30 epoch, 640px → belirgin mAP artışı hedefleniyor
  - Filtreleme inference'ta yapılır (pedestrian/people dahil edilmez)
"""

from ultralytics import YOLO
from pathlib import Path

# ─── Ayarlar ──────────────────────────────────────────────────────────────────
MODEL_BASE   = "yolov8n.pt"
DATA_YAML    = "VisDrone.yaml"
EPOCHS       = 30
DEVICE       = "cuda"
IMAGE_SIZE   = 640
BATCH_SIZE   = 8
PROJECT_DIR  = "runs/train"
RUN_NAME     = "visdrone_v2"

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
    print("VisDrone Fine-Tune v2 Başlıyor")
    print(f"  Model   : {MODEL_BASE}")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  ImgSize : {IMAGE_SIZE}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  Device  : {DEVICE}")
    print("=" * 60)

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        # Optimizasyon
        optimizer="AdamW",
        lr0=0.001,
        mosaic=1.0,        # küçük nesne augmentation
        close_mosaic=5,    # son 5 epoch'ta mosaic kapat (stabilite)
        workers=4,
        cache=False,
        patience=10,
        save=True,
        save_period=5,
        exist_ok=True,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("✅ Eğitim tamamlandı!")
    print(f"   En iyi model : {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    try:
        print(f"   mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    except Exception:
        print("   (Metrikler runs/train/visdrone_v2/results.csv dosyasında)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()