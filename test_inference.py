"""
Pretrained YOLOv8 ile hızlı tespit testi.
Pipeline'ın çalıştığını doğrulamak için kullanılır (fine-tune ÖNCESİ).
"""
from ultralytics import YOLO

# Hazır (pretrained) nano model — ilk indirmede otomatik iner
model = YOLO("yolov8n.pt")

# Ultralytics'in örnek görselinde test (otomatik iner)
results = model("https://ultralytics.com/images/bus.jpg", device="mps")

# Sonucu kaydet
results[0].save(filename="test_sonuc.jpg")

print("Tespit edilen nesneler:")
for box in results[0].boxes:
    cls = int(box.cls)
    print(f"  - {model.names[cls]}: {float(box.conf):.2f}")