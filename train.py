from ultralytics import YOLO

model = YOLO("yolo26n.pt")   # if this won't download, change to "yolo11n.pt"
results = model.train(
    data="yolo_10/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=2,
    device="cpu",
    workers=0,
    name="yolo_10_train",
)