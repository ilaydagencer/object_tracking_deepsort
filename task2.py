import torch
from ultralytics import YOLO  # YOLO sınıfını buradan import et
from torch.nn import Sequential, Conv2d  # Conv sınıfını da import et
from ultralytics.nn.tasks import DetectionModel

# Güvenlik ayarları
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv2d])

# Modeli yükle
model = YOLO("yolov8n.pt")

