import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    8: "bus",
    9: "motor",
}

DEFAULT_MODEL = r"runs\detect\runs\train\visdrone_v1\weights\best.pt"


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def detect(model: YOLO, image_path: str, conf: float = 0.25):
    results = model(image_path, conf=conf)[0]
    return results


def count_vehicles(results) -> dict:
    counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id in VEHICLE_CLASSES:
            counts[VEHICLE_CLASSES[cls_id]] += 1
    return counts


def draw_detections(image: np.ndarray, results, alpha: float = 1.0) -> np.ndarray:
    COLOR_MAP = {
        "car":     (0, 255, 0),
        "van":     (255, 165, 0),
        "truck":   (0, 0, 255),
        "bus":     (255, 0, 255),
        "motor":   (0, 255, 255),
        "bicycle": (255, 255, 0),
    }
    out = image.copy()
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in VEHICLE_CLASSES:
            continue
        label = VEHICLE_CLASSES[cls_id]
        conf  = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = COLOR_MAP.get(label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(out, text, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def build_heatmap(image: np.ndarray, results) -> np.ndarray:
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = x2 - x1
        bh = y2 - y1
        radius = max(int(max(bw, bh) * 1.5), 20)
        cv2.circle(heat, (cx, cy), radius, 1.0, -1)

    # Gaussian blur — yumuşat
    blur_k = max(int(min(h, w) * 0.05) | 1, 21)  # tek sayı olmalı
    heat = cv2.GaussianBlur(heat, (blur_k, blur_k), 0)

    # Normalize + colormap uygula
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_uint8 = (heat * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # Orijinal görüntüyle blend
    overlay = cv2.addWeighted(image, 0.5, heatmap_color, 0.6, 0)
    return overlay


def print_summary(counts: dict, image_path: str):
    total = sum(counts.values())
    print(f"\n{'='*40}")
    print(f"Görüntü    : {image_path}")
    print(f"Toplam araç: {total}")
    print(f"{'-'*40}")
    for label, cnt in counts.items():
        if cnt > 0:
            print(f"  {label:<10}: {cnt}")
    print(f"{'='*40}\n")


def analyze(image_path: str, model_path: str, conf: float = 0.25, output_dir: str = "output"):
    Path(output_dir).mkdir(exist_ok=True)
    stem = Path(image_path).stem

    print(f"[1/4] Model yükleniyor : {model_path}")
    model = load_model(model_path)

    print(f"[2/4] Tespit yapılıyor : {image_path}")
    results = detect(model, image_path, conf)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü açılamadı: {image_path}")

    print("[3/4] Bounding box'lar çiziliyor...")
    detected_img = draw_detections(image, results)
    det_path = str(Path(output_dir) / f"{stem}_detections.jpg")
    cv2.imwrite(det_path, detected_img)

    print("[4/4] Heatmap üretiliyor...")
    heatmap_img = build_heatmap(image, results)
    heat_path = str(Path(output_dir) / f"{stem}_heatmap.jpg")
    cv2.imwrite(heat_path, heatmap_img)

    counts = count_vehicles(results)
    print_summary(counts, image_path)
    print(f"✅ Tespit görseli → {det_path}")
    print(f"✅ Heatmap         → {heat_path}")

    return counts, det_path, heat_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone araç analizi")
    parser.add_argument("--image",  required=True,         help="Analiz edilecek görüntü yolu")
    parser.add_argument("--model",  default=DEFAULT_MODEL, help="best.pt yolu")
    parser.add_argument("--conf",   type=float, default=0.25, help="Güven eşiği (0-1)")
    parser.add_argument("--output", default="output",      help="Çıktı klasörü")
    args = parser.parse_args()

    analyze(args.image, args.model, args.conf, args.output)