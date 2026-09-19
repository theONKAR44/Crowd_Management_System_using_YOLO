"""Command-line application: live monitoring window with runtime controls."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

import cv2

from .alerts import AlertManager
from .config import Config
from .detector import YoloV3HeadDetector
from .pipeline import CrowdMonitor, Detector
from .tracker import HeadTracker
from .video import SourceManager
from .zones import ZoneEditor, ZoneManager

WINDOW = "Crowd Monitor"
KEY_ENTER, KEY_ESC, KEY_BACKSPACE = 13, 27, 8


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        description="Real-time crowd monitoring with YOLOv3 head detection and DeepSORT tracking."
    )
    p.add_argument("--source", nargs="+", default=d.sources,
                   help="Webcam index, video file, or RTSP/HTTP URL. Give several to switch with the S key.")
    p.add_argument("--cfg", default=d.yolo_cfg, help="YOLOv3 .cfg file")
    p.add_argument("--weights", default=d.yolo_weights, help="YOLOv3 .weights file")
    p.add_argument("--conf", type=float, default=d.conf_threshold, help="Detection confidence threshold")
    p.add_argument("--nms", type=float, default=d.nms_threshold, help="NMS IoU threshold")
    p.add_argument("--input-size", type=int, default=d.input_size, help="YOLO input size (416 default)")
    p.add_argument("--width", type=int, default=d.frame_width, help="Resize frames to this width")
    p.add_argument("--cuda", action="store_true", help="Use the OpenCV CUDA backend (needs a CUDA build of OpenCV)")
    p.add_argument("--blur", action="store_true", help="Gaussian blur noise reduction")
    p.add_argument("--clahe", action="store_true", help="CLAHE contrast enhancement for poor lighting")
    p.add_argument("--appearance", choices=["histogram", "mobilenet"], default=d.appearance,
                   help="Appearance descriptor for DeepSORT")
    p.add_argument("--max-age", type=int, default=d.max_age)
    p.add_argument("--n-init", type=int, default=d.n_init)
    p.add_argument("--zones", default=d.zones_file, help="JSON file with zone definitions (created if missing)")
    p.add_argument("--capacity", type=int, default=d.default_capacity, help="Capacity of the default zone")
    p.add_argument("--warning", type=float, default=d.warning_ratio, help="Warning threshold (fraction of capacity)")
    p.add_argument("--danger", type=float, default=d.danger_ratio, help="Danger threshold (fraction of capacity)")
    p.add_argument("--cooldown", type=float, default=d.alert_cooldown, help="Seconds between repeated alerts")
    p.add_argument("--sound", action="store_true", help="Play a beep on warning and danger alerts")
    p.add_argument("--no-heatmap", action="store_true", help="Start with the heatmap off")
    p.add_argument("--log-dir", default=d.log_dir)
    p.add_argument("--output", help="Write the annotated video to this file (e.g. out.mp4)")
    p.add_argument("--no-display", action="store_true", help="Run without a window (use with --output)")
    p.add_argument("--max-frames", type=int, help="Stop after this many frames")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        sources=list(args.source),
        frame_width=args.width,
        blur=args.blur,
        clahe=args.clahe,
        yolo_cfg=args.cfg,
        yolo_weights=args.weights,
        input_size=args.input_size,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        use_cuda=args.cuda,
        appearance=args.appearance,
        max_age=args.max_age,
        n_init=args.n_init,
        zones_file=args.zones,
        default_capacity=args.capacity,
        warning_ratio=args.warning,
        danger_ratio=args.danger,
        alert_cooldown=args.cooldown,
        alert_sound=args.sound,
        log_dir=args.log_dir,
        heatmap=not args.no_heatmap,
    )


def run(
    cfg: Config,
    detector: Optional[Detector] = None,
    display: bool = True,
    output: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> int:
    """Run the monitor. `detector` can be injected (used by the tests)."""
    if detector is None:
        detector = YoloV3HeadDetector(
            cfg.yolo_cfg, cfg.yolo_weights, cfg.input_size,
            cfg.conf_threshold, cfg.nms_threshold, cfg.head_ratio, cfg.use_cuda,
        )
    tracker = HeadTracker(cfg.appearance, cfg.max_age, cfg.n_init, cfg.max_cosine_distance)
    zones = ZoneManager.load(cfg.zones_file)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_base = os.path.join(cfg.log_dir, f"alerts_{stamp}")
    alerts = AlertManager(cfg.alert_cooldown, log_base + ".jsonl", cfg.alert_sound)
    monitor = CrowdMonitor(detector, tracker, zones, alerts, cfg)
    sources = SourceManager(cfg.sources)
    editor = ZoneEditor()
    writer: Optional[cv2.VideoWriter] = None
    frames = 0

    def save_zones() -> None:
        if zones.zones:
            zones.save(cfg.zones_file)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.add_point(x, y)

    if display:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            ok, frame = sources.read()
            if not ok:
                if sources.current.is_live:
                    time.sleep(0.05)
                    continue
                break  # end of video file

            result = monitor.process_frame(frame, sources.name, editor)
            for alert in result.new_alerts:
                print(alert.message, flush=True)

            if output:
                if writer is None:
                    h, w = result.frame.shape[:2]
                    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), sources.current.fps, (w, h))
                writer.write(result.frame)

            frames += 1
            if max_frames and frames >= max_frames:
                break
            if not display:
                continue

            cv2.imshow(WINDOW, result.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if editor.active:
                if key == KEY_ENTER:
                    if editor.finish(zones, cfg.default_capacity, cfg.warning_ratio, cfg.danger_ratio):
                        save_zones()
                elif key == KEY_BACKSPACE:
                    editor.undo()
                elif key == KEY_ESC:
                    editor.cancel()
                continue
            if key == ord("q"):
                break
            elif key == ord("s"):
                if sources.switch():
                    monitor.reset()
            elif key == ord("r"):
                monitor.reset()
            elif key == ord("c"):
                os.makedirs("captures", exist_ok=True)
                path = os.path.join("captures", f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(path, result.frame)
                print(f"Saved {path}")
            elif key == ord("h"):
                monitor.toggle_heatmap()
            elif key == ord("z"):
                editor.start()
            elif key == ord("x"):
                zones.clear()
                if os.path.isfile(cfg.zones_file):
                    os.remove(cfg.zones_file)
            elif key in (ord("+"), ord("=")):
                for z in zones.zones:
                    z.capacity += 5
                save_zones()
            elif key in (ord("-"), ord("_")):
                for z in zones.zones:
                    z.capacity = max(1, z.capacity - 5)
                save_zones()
            elif key == ord("]"):
                for z in zones.zones:
                    z.danger_ratio = min(1.0, round(z.danger_ratio + 0.05, 2))
                save_zones()
            elif key == ord("["):
                for z in zones.zones:
                    z.danger_ratio = max(round(z.warning_ratio + 0.05, 2), round(z.danger_ratio - 0.05, 2))
                save_zones()
    finally:
        sources.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        if alerts.history:
            alerts.export_csv(log_base + ".csv")
            print(f"Alert log saved to {log_base}.csv")
    return frames


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.no_display and not (args.output or args.max_frames):
        print("Note: --no-display without --output or --max-frames runs until the video ends.", file=sys.stderr)
    try:
        run(cfg, display=not args.no_display, output=args.output, max_frames=args.max_frames)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
