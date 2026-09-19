# Crowd Management System using YOLO and DeepSORT

Real-time head detection, tracking and crowd-density alerting from CCTV, webcam or video feeds.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?logo=opencv&logoColor=white)
![YOLOv3](https://img.shields.io/badge/Detector-YOLOv3-00A67E)
![Tracker](https://img.shields.io/badge/Tracker-DeepSORT-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

Crowd crushes are a recurring danger at stadiums, stations and public events. In June 2025, 11 people died in a crowd crush outside Chinnaswamy Stadium in Bengaluru during a victory celebration, and India has seen several similar tragedies. Manual monitoring does not scale to that many faces. This project turns ordinary cameras into an early-warning system: it counts and tracks heads, measures how full each zone is, and raises graded alerts **before** a zone reaches capacity.

It was built as an academic project at the Department of Information Technology, National Institute of Technology Karnataka (NITK), Surathkal.

## Features

- **Head detection** with YOLOv3 (COCO "person" class, head region taken as the upper 30% of the person box) and Non-Maximum Suppression
- **Persistent tracking** with DeepSORT: Kalman-filter motion model, appearance descriptors, Hungarian assignment, tentative / confirmed / deleted track states
- **Zone monitoring**: draw polygonal zones at runtime, give each a capacity, and get a live count, occupancy percentage and optional persons per m²
- **Three-level alerts**: normal, **warning at 60%** and **danger at 80%** of capacity, with a per-zone cooldown to prevent alert fatigue, a JSON-lines log and CSV export
- **Live density heatmap** with temporal decay, so lingering crowds show up as hotspots
- **Multiple inputs**: webcam, video file or RTSP/HTTP stream, with hot-switching between sources
- **Optional pre-processing** for difficult footage: Gaussian blur and CLAHE contrast enhancement
- Designed for standard hardware (the report measured 10-15 FPS on a laptop-class CPU at moderate crowd sizes); optional OpenCV CUDA backend

## How it works

![System architecture](assets/architecture.png)

Every frame goes through the same pipeline (`crowd_management/pipeline.py`):

| Step | What happens | Module |
|------|--------------|--------|
| 1. Input | Read a frame from a webcam, file or stream | `video.py` |
| 2. Pre-process | Resize (default width 640), optional blur and CLAHE | `preprocess.py` |
| 3. Detect | YOLOv3 finds people; the upper 30% of each box becomes a head box; NMS removes duplicates | `detector.py` |
| 4. Track | DeepSORT gives every head a stable ID across frames, even through short occlusions | `tracker.py` |
| 5. Analyse | Point-in-polygon test counts heads per zone; status is Safe, Warning or Danger | `zones.py` |
| 6. Heatmap | Gaussian blobs at head positions accumulate with decay: `H(t+1) = 0.95 * H(t) + sum of blobs` | `heatmap.py` |
| 7. Alert | Escalations alert immediately; repeats respect the cooldown; every alert is logged | `alerts.py` |
| 8. Display | Circles and IDs, zone outlines, heatmap overlay, HUD | `visualize.py` |

## Results reported in the project report

These numbers come from the experiments described in the project report (Intel Core i7-10750H, 16 GB RAM, optional GTX 1650, Windows 10, Python 3.8, OpenCV 4.5.5, PyTorch 1.9). Tracking was evaluated with the MOTChallenge dataset and the density tests used custom recordings of 5 to 100 people.

**Tracking method comparison**

| Method | FPS | MOTA | ID switches | Processing time (ms) |
|--------|----:|-----:|------------:|---------------------:|
| Simple centroid | 15.2 | 65.3% | 42 | 18.3 |
| SORT | 13.8 | 78.6% | 18 | 26.7 |
| **DeepSORT** | 12.1 | **86.4%** | **8** | 35.4 |

**Performance at different crowd densities**

| Heads in view | FPS | Detection rate | Track consistency | CPU usage |
|--------------:|----:|---------------:|------------------:|----------:|
| 1-10 | 14.8 | 98.2% | 94.3% | 45% |
| 11-30 | 12.3 | 95.7% | 91.8% | 68% |
| 31-60 | 9.6 | 92.1% | 88.5% | 82% |
| 61-100 | 6.8 | 87.3% | 83.2% | 95% |

> **Note:** This repository is a clean re-implementation of the pipeline described in the report. The numbers above were measured in the original experiments and have not been regenerated with this code, so results on your hardware and footage will differ.

## Quick start

**1. Install**

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/crowd-management-yolo-deepsort.git
cd crowd-management-yolo-deepsort

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

**2. Download the YOLOv3 model (about 237 MB)**

```bash
python scripts/download_weights.py
```

This places `yolov3.cfg` and `yolov3.weights` in `models/`. If the download fails, get both files from the [official YOLO page](https://pjreddie.com/darknet/yolo/) and put them in that folder.

**3. Run**

```bash
python main.py --source 0                       # webcam
python main.py --source crowd.mp4               # video file
python main.py --source rtsp://user:pass@192.168.1.10/stream1
python main.py --source cam1.mp4 cam2.mp4       # press S to switch sources
python main.py --source crowd.mp4 --no-display --output annotated.mp4   # headless
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Switch to the next video source |
| `R` | Reset tracks, heatmap and alert state |
| `C` | Save a snapshot to `captures/` |
| `H` | Toggle the heatmap |
| `Z` | Draw a new zone: click points, `Enter` to finish, `Backspace` to undo, `Esc` to cancel |
| `X` | Clear all zones |
| `+` / `-` | Raise or lower zone capacity by 5 |
| `[` / `]` | Lower or raise the danger threshold by 5% |

Zones you draw are saved to `configs/zones.json` and reloaded next time. See `configs/zones.example.json` for the format, including the optional `area_m2` field that enables persons-per-m² density.

### Useful options

Run `python main.py --help` for the full list. The ones you will use most:

| Option | Meaning | Default |
|--------|---------|---------|
| `--capacity N` | Capacity of the default full-frame zone | 30 |
| `--warning` / `--danger` | Alert thresholds as a fraction of capacity | 0.6 / 0.8 |
| `--conf` | YOLO confidence threshold | 0.4 |
| `--cooldown` | Seconds between repeated alerts for a zone | 10 |
| `--clahe` / `--blur` | Help in dim light or noisy footage | off |
| `--appearance` | `histogram` (no extra install) or `mobilenet` (learned descriptor, needs `pip install torch torchvision`) | histogram |
| `--cuda` | OpenCV CUDA backend (needs a CUDA build of OpenCV) | off |
| `--sound` | Beep on warning and danger alerts | off |

Alerts are printed to the console and logged to `logs/alerts_<timestamp>.jsonl`, with a CSV export written when the session ends.

### Tuning tips

The report found the system sensitive to camera setup, so expect to tune it per camera:

- **Distant camera, tiny heads:** lower `--conf` so small heads are still detected.
- **Close camera, large heads:** raise `--conf`; a low threshold makes one person get detected two or three times.
- **Dawn, dusk or dim scenes:** try `--clahe`.
- **Slow on a CPU:** lower `--width` (for example 480) or use `--cuda`.

## Project structure

```
crowd-management-yolo-deepsort/
├── main.py                     # entry point
├── crowd_management/
│   ├── cli.py                  # live window, key controls, video output
│   ├── pipeline.py             # per-frame processing pipeline
│   ├── detector.py             # YOLOv3 head detection + NMS
│   ├── tracker.py              # DeepSORT wrapper + appearance descriptors
│   ├── zones.py                # polygon zones, thresholds, runtime zone editor
│   ├── heatmap.py              # decaying Gaussian density heatmap
│   ├── alerts.py               # 3-level alerts, cooldown, logging
│   ├── visualize.py            # drawing helpers and HUD
│   ├── preprocess.py           # resize, blur, CLAHE
│   ├── video.py                # webcam / file / stream input
│   └── config.py               # defaults
├── scripts/download_weights.py
├── configs/zones.example.json
├── tests/                      # unit and integration tests
├── assets/architecture.png
└── requirements.txt
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover head-box geometry, YOLO output parsing and NMS, point-in-polygon zones and thresholds, the heatmap maths, alert escalation and cooldown, tracker identity stability through crossings and occlusions, and a full pipeline run on a generated video. They use a stub detector, so they run without the 237 MB YOLOv3 weights; live YOLOv3 inference is not part of the automated tests.

## Limitations

- Compute cost grows with crowd size (about 7 FPS at 61 to 100 heads on the test laptop).
- Results depend heavily on parameters such as the YOLO confidence threshold, which need manual adjustment when the camera view or angle changes.
- Performance drops in dim, in-between light (dawn and dusk) and in very dense crowds, where camera clarity and angle decide how much occlusion there is.
- Head detection depends on camera angle, and training-data bias in the COCO-trained detector can affect detection across different populations.

## Future work

- Newer detectors (YOLOv5 or YOLOv7) and attention mechanisms for better occlusion handling
- Multi-camera tracking for large areas
- Behaviour analysis and anomaly detection, predictive crowd-flow analytics
- Optimisation for edge and embedded deployment

## Authors

- **Onkar Chandrashekhar Gaikwad**
- **Kushal Kumar M**
- Guide: **Dr. Janani T**, Department of Information Technology, NITK Surathkal

## Acknowledgements

- YOLOv3: J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement", 2018, and the original [Darknet](https://pjreddie.com/darknet/yolo/) weights
- DeepSORT: N. Wojke, A. Bewley and D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric", ICIP 2017
- [`deep-sort-realtime`](https://github.com/levan92/deep_sort_realtime) for the tracker implementation used here
- The open-source YOLO and DeepSORT community

## License

Released under the [MIT License](LICENSE).
