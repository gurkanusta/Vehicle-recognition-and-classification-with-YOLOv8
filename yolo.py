from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import Counter


model = YOLO('runs/detect/tez_yolov8m/weights/best.pt')
video_folder = 'videolar/'
output_folder = 'yolo_lu_videolar'
show_video = True
frame_skip = 2


os.makedirs(output_folder, exist_ok=True)


CLASS_NAMES = {0: 'Otomobil', 1: 'Otobus', 2: 'Kamyon', 3: 'Motosiklet'}
VEHICLE_CLASSES = [0, 1, 2, 3]


MIN_FRAMES_TO_COUNT = 8
HISTORY_LEN = 40
SWITCH_THRESHOLD = 0.70


if not os.path.exists(video_folder):
    print(f"HATA: '{video_folder}' klasörü bulunamadı!")
    exit()

videos = sorted([f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))])
current_video_idx = 0

print("------------------------------------------------")
print(f"Videolar '{output_folder}' klasörüne kaydedilecek.")
print("KONTROLLER: [N] Sonraki | [B] Önceki | [Q] Çıkış")
print("------------------------------------------------")

while 0 <= current_video_idx < len(videos):
    vid = videos[current_video_idx]
    video_path = os.path.join(video_folder, vid)
    cap = cv2.VideoCapture(video_path)


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_fps = fps / frame_skip if frame_skip > 0 else fps

    save_path = os.path.join(output_folder, f"islenmis_{vid}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, save_fps, (width, height))
    # -------------------------------------------

    print(f"🎬 {vid} işleniyor ve kaydediliyor... ({current_video_idx + 1}/{len(videos)})")

    track_history = {}
    counted_ids = {}
    total_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    next_video = False
    prev_video = False
    quit_program = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            next_video = True
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue


        results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES, conf=0.15, imgsz=800,
                              tracker="bytetrack.yaml")

        overlay = frame.copy()
        instant_counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().numpy().astype(int)

            for i in range(len(xyxys)):
                x1, y1, x2, y2 = xyxys[i]
                cls_id = classes[i]
                track_id = track_ids[i]


                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(cls_id)

                if len(track_history[track_id]) > HISTORY_LEN:
                    track_history[track_id].pop(0)

                current_hist = track_history[track_id]
                counter = Counter(current_hist)
                most_common = counter.most_common(1)[0]
                dominant_class = most_common[0]
                dominant_ratio = most_common[1] / len(current_hist)

                if len(current_hist) >= MIN_FRAMES_TO_COUNT:
                    if track_id not in counted_ids:
                        counted_ids[track_id] = dominant_class
                        total_counts[dominant_class] += 1
                    else:
                        previous_class = counted_ids[track_id]
                        if previous_class != dominant_class and dominant_ratio > SWITCH_THRESHOLD:
                            total_counts[previous_class] -= 1
                            total_counts[dominant_class] += 1
                            counted_ids[track_id] = dominant_class

                display_class = counted_ids.get(track_id, cls_id)
                if display_class in instant_counts:
                    instant_counts[display_class] += 1

                class_name = CLASS_NAMES.get(display_class, 'Bilinmeyen')


                color = (0, 255, 0)

                if display_class == 1:
                    color = (0, 0, 255)
                elif display_class == 2:
                    color = (0, 165, 255)
                elif display_class == 3:
                    color = (0, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{class_name} #{track_id}"
                cv2.putText(frame, label_text, (x1, max(35, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        grand_total = sum(total_counts.values())


        panel_w, panel_h = 350, 300
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 2)
        cv2.putText(frame, "TRAFIK ANALIZI", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "CANLI", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "TOPLAM", (270, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.line(frame, (20, 75), (panel_w, 75), (150, 150, 150), 1)

        y_offset = 110
        for cls_id in VEHICLE_CLASSES:
            name = CLASS_NAMES.get(cls_id)
            c_inst = instant_counts.get(cls_id, 0)
            c_total = total_counts.get(cls_id, 0)


            row_color = (255, 255, 255)

            if cls_id == 1:
                row_color = (0, 0, 255)
            elif cls_id == 2:
                row_color = (0, 165, 255)
            elif cls_id == 3:
                row_color = (0, 255, 255)

            cv2.putText(frame, f"{name}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, row_color, 2)
            cv2.putText(frame, f"{c_inst:02d}", (210, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(frame, f"{c_total:03d}", (280, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            if cls_id != 3:
                cv2.line(frame, (30, y_offset + 15), (panel_w - 10, y_offset + 15), (50, 50, 50), 1)
            y_offset += 35

        cv2.line(frame, (20, y_offset - 5), (panel_w, y_offset - 5), (255, 255, 255), 2)
        cv2.putText(frame, f"TOPLAM ARAC: {grand_total}", (30, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        out.write(frame)


        if show_video:
            cv2.imshow("Tez Akilli Trafik Sayimi", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                quit_program = True;
                break
            elif key == ord('n'):
                next_video = True;
                break
            elif key == ord('b'):
                prev_video = True;
                break

    cap.release()
    out.release()

    if quit_program:
        break
    elif next_video:
        current_video_idx += 1
    elif prev_video:
        current_video_idx = max(0, current_video_idx - 1)
    else:
        current_video_idx += 1

cv2.destroyAllWindows()
print("İşlem tamamlandı.")