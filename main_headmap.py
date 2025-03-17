import os
import random
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from tracker import Tracker

# Dosya yolları
video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'out_headmap.mp4')
json_output_path = os.path.join('.', 'tracked_objects.json')
heatmap_output_path = os.path.join('.', 'heatmap.png')

# JSON dosyasını sıfırla
if os.path.exists(json_output_path):
    os.remove(json_output_path)

tracked_data = []  # JSON için takip edilen kişilerin verisi
heatmap_coords = []  # Isı haritası için koordinatlar

# Video aç
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")
tracker = Tracker()

# Rastgele renkler oluştur
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

while ret:
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            # Takip edilen kişilerin merkez noktalarını kaydet (Isı haritası için)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            heatmap_coords.append((center_x, center_y))

            # JSON kaydı için takip edilen bilgileri sakla
            track_info = {
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y]
            }
            tracked_data.append(track_info)

            # Takip edilen kişiyi çerçeve içinde göster
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (colors[track_id % len(colors)]), 2)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

# JSON dosyasına kaydet
with open(json_output_path, "w") as json_file:
    json.dump(tracked_data, json_file, indent=4)

print(f"JSON verisi kaydedildi: {json_output_path}")

# Isı haritasını oluştur
def create_heatmap(heatmap_coords, width, height, heatmap_output_path):
    heatmap_img = np.zeros((height, width), dtype=np.float32)

    for x, y in heatmap_coords:
        if 0 <= x < width and 0 <= y < height:
            heatmap_img[y, x] += 1

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_img, cmap="jet", alpha=0.6)
    plt.axis("off")
    plt.title("Movement Heatmap")
    plt.savefig(heatmap_output_path, bbox_inches='tight', dpi=300)
    plt.close()

# Videonun boyutlarını al
height, width = frame.shape[:2]
create_heatmap(heatmap_coords, width, height, heatmap_output_path)

print(f"Isı haritası kaydedildi: {heatmap_output_path}")

