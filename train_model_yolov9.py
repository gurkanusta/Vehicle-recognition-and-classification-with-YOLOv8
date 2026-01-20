from ultralytics import YOLO
import torch


def main():

    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Tespit Edildi: {device_name} (YOLOv9C HIZLI Modu)")


        batch_size = 23
    else:
        device = 'cpu'
        print("⚠️ GPU bulunamadı. CPU üzerinden batch=8 ile devam edilecek.")
        batch_size = 8


    print("Model yükleniyor: YOLOv9c (Compact - En İyi Denge)...")

    MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt'
    model = YOLO(MODEL_URL)


    print("\n=======================================================")
    print(f"      📈 YOLOv9C HIZLI EĞİTİMİ BAŞLIYOR (150 EPOCH) 📈")
    print(f"      Batch Size: {batch_size}")
    print("=======================================================")

    results = model.train(
        data='traffic.yaml',
        epochs=150,
        imgsz=640,
        batch=batch_size,
        name='yolov9c_comparison',
        patience=30,
        device=device,
        workers=16,
        cache=True,
        exist_ok=True,
        amp=True
    )

    print("\n✅ EĞİTİM BAŞARILIYLA TAMAMLANDI!")
    print("Karşılaştırma modelin burada: runs/detect/yolov9c_comparison/weights/best.pt")


if __name__ == '__main__':
    main()