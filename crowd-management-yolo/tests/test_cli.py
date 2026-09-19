import os

import cv2
import numpy as np
import pytest

from crowd_management.cli import build_parser, config_from_args, run
from tests.helpers import ScriptedDetector, make_head


def write_video(path, frames=20, size=(320, 240)):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
    rng = np.random.default_rng(1)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


def test_parser_defaults_match_config():
    cfg = config_from_args(build_parser().parse_args([]))
    assert cfg.warning_ratio == 0.6 and cfg.danger_ratio == 0.8
    assert cfg.input_size == 416 and cfg.head_ratio == 0.3


def test_parser_overrides():
    args = build_parser().parse_args(["--source", "a.mp4", "b.mp4", "--capacity", "50", "--no-heatmap", "--clahe"])
    cfg = config_from_args(args)
    assert cfg.sources == ["a.mp4", "b.mp4"] and cfg.default_capacity == 50
    assert cfg.heatmap is False and cfg.clahe is True


def test_end_to_end_on_video_file_headless(tmp_path):
    video = write_video(str(tmp_path / "in.avi"))
    args = build_parser().parse_args([
        "--source", video, "--capacity", "8", "--width", "320",
        "--zones", str(tmp_path / "zones.json"), "--log-dir", str(tmp_path / "logs"),
    ])
    cfg = config_from_args(args)
    heads = [make_head(30 + 35 * (i % 8), 40 + 45 * (i // 8)) for i in range(7)]  # 7/8 -> DANGER
    out = str(tmp_path / "out.mp4")

    frames = run(cfg, detector=ScriptedDetector([heads]), display=False, output=out)

    assert frames == 20  # every frame of the file processed
    assert os.path.getsize(out) > 0
    logs = list((tmp_path / "logs").glob("alerts_*.jsonl"))
    assert logs and "DANGER" in logs[0].read_text()
    assert list((tmp_path / "logs").glob("alerts_*.csv"))


def test_max_frames_stops_early(tmp_path):
    video = write_video(str(tmp_path / "in.avi"))
    cfg = config_from_args(build_parser().parse_args([
        "--source", video, "--zones", str(tmp_path / "z.json"), "--log-dir", str(tmp_path / "l"),
    ]))
    assert run(cfg, detector=ScriptedDetector([[]]), display=False, max_frames=5) == 5


def test_missing_video_source_raises(tmp_path):
    cfg = config_from_args(build_parser().parse_args(["--source", str(tmp_path / "missing.mp4")]))
    with pytest.raises(RuntimeError):
        run(cfg, detector=ScriptedDetector([[]]), display=False)


def test_missing_weights_gives_helpful_error(tmp_path):
    from crowd_management.detector import YoloV3HeadDetector

    with pytest.raises(FileNotFoundError, match="download_weights"):
        YoloV3HeadDetector(str(tmp_path / "x.cfg"), str(tmp_path / "x.weights"))
