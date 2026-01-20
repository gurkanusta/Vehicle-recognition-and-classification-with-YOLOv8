from ultralytics import YOLO
import torch


def main():

    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Tespit Edildi: {device_name} (Canavar Hazır!)")




        batch_size = 20
    else:
        device = 'cpu'
        print("⚠️ GPU bulunamadı. CPU moduna geçiliyor.")
        batch_size = 2


    print("Model yükleniyor: YOLOv8m ")
    model = YOLO('yolov8m.pt')


    print("\n=======================================================")
    print(f"      ⭐⭐⭐ NİHAİ EĞİTİM BAŞLIYOR (300 EPOCH) ⭐⭐⭐")
    print(f"      Model: YOLOv8")
    print(f"      Batch Size: {batch_size}")
    print("=======================================================")

    results = model.train(
        data='traffic.yaml',
        epochs=300,
        imgsz=800,
        batch=batch_size,
        name='tez_yolov8m',
        patience=50,
        device=device,
        workers=8,
        cache=True,
        exist_ok=True,
        amp=True,

    )

    print("\n✅ TEZ EĞİTİMİ BAŞARILIYLA TAMAMLANDI!")
    print("Nihai modelin yolu: runs/detect/tez_yolov8mv2/weights/best.pt")


if __name__ == '__main__':
    main()