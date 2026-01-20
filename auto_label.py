from ultralytics import YOLO
import os
import torch
import glob


MODEL_PATH = 'runs/detect/tez_yolov8m/weights/best.pt'

INPUT_IMAGE_FOLDER = 'new_images_to_label/'
OUTPUT_FOLDER = 'auto_label_results/'


CONFIDENCE_THRESHOLD = 0.45


def auto_label():
    try:
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Hazır Güçlü Model (YOLOv8x) yükleniyor... Cihaz: {device}")
        model = YOLO(MODEL_PATH)

        if not os.path.exists(INPUT_IMAGE_FOLDER):
            print(f"HATA: '{INPUT_IMAGE_FOLDER}' klasörü yok.")
            return

        num_images = len([f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"🚀 {num_images} resim için süper model ile etiketleme başladı...")


        results = model.predict(
            source=INPUT_IMAGE_FOLDER,
            save=True,
            save_txt=True,
            project=OUTPUT_FOLDER,
            name='labels',
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            classes=[2, 3, 5, 7],
            exist_ok=True,
            verbose=False
        )

        print("\n✅ Tespit bitti. Şimdi sınıf numaraları senin setine göre dönüştürülüyor...")


        mapping = {2: 0, 5: 1, 7: 2, 3: 3}


        txt_files = glob.glob(os.path.join(OUTPUT_FOLDER, 'labels', 'labels', '*.txt'))

        for txt_file in txt_files:
            new_lines = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        coco_id = int(parts[0])
                        if coco_id in mapping:
                            # ID'yi değiştir
                            parts[0] = str(mapping[coco_id])
                            new_lines.append(" ".join(parts) + "\n")


            with open(txt_file, 'w') as f:
                f.writelines(new_lines)

        print("\n İşlem tamam")
        print(f"Etiketler burada: {OUTPUT_FOLDER}/labels/labels/")


    except Exception as e:
        print(f"Hata: {e}")


if __name__ == '__main__':
    auto_label()