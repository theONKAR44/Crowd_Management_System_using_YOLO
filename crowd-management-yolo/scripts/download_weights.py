"""Download the YOLOv3 (Darknet) config and weights into ./models.

    python scripts/download_weights.py

The weights are the original COCO-trained YOLOv3 model by Joseph Redmon
(https://pjreddie.com/darknet/yolo/). If pjreddie.com is unavailable, download
yolov3.cfg and yolov3.weights from another trusted mirror and place them in
the models/ folder manually.
"""

import os
import sys
import urllib.request

FILES = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
}
EXPECTED_WEIGHTS_BYTES = 248_007_048  # about 237 MB
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    pct = 100 * done / total
    sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")
    sys.stdout.flush()


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in FILES.items():
        target = os.path.join(MODELS_DIR, name)
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            print(f"{name}: already present, skipping")
            continue
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, target, _progress)
            print()
        except Exception as exc:  # network errors, HTTP errors, ...
            if os.path.exists(target):
                os.remove(target)
            print(f"\nFailed to download {name}: {exc}", file=sys.stderr)
            print(f"Download it manually from {url} and save it to {target}", file=sys.stderr)
            return 1
    weights = os.path.join(MODELS_DIR, "yolov3.weights")
    size = os.path.getsize(weights)
    if size != EXPECTED_WEIGHTS_BYTES:
        print(f"Warning: yolov3.weights is {size} bytes, expected {EXPECTED_WEIGHTS_BYTES}. "
              "The download may be incomplete.", file=sys.stderr)
    print(f"Done. Model files are in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
