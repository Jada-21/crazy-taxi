import json, shutil
from pathlib import Path

ndjson = Path("yolotraffic.ndjson")
frames = Path("frames")
out = Path("yolo_10")

chosen = [f"frame_{i:05d}" for i in range(1200, 1190, -1)]  # 01200 ... 01191
val_names = set(chosen[8:])                                  # last 2 go to val

records = {}
with ndjson.open() as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("type") == "image":
            records[Path(rec["file"]).stem] = rec

for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

for name in chosen:
    split = "val" if name in val_names else "train"
    boxes = records[name].get("annotations", {}).get("boxes", [])
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    shutil.copy(frames / f"{name}.jpg", out / "images" / split / f"{name}.jpg")
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{name} -> {split}: {len(boxes)} boxes")

yaml_text = f"""path: {out.resolve()}
train: images/train
val: images/val

nc: 3
names:
  0: human
  1: bus
  2: semi
"""
(out / "dataset.yaml").write_text(yaml_text)
print("Done. Wrote", out / "dataset.yaml")