import numpy as np

from crowd_management.detector import head_box_from_person, parse_outputs
from tests.helpers import make_row


def test_head_box_is_upper_30_percent():
    assert head_box_from_person(10, 20, 50, 100) == (10, 20, 50, 30)


def test_head_box_uses_floor():
    # 0.3 * 33 = 9.9 -> floor -> 9
    assert head_box_from_person(0, 0, 10, 33)[3] == 9


def test_head_box_never_collapses_to_zero_height():
    assert head_box_from_person(0, 0, 10, 2)[3] == 1


def test_parse_single_person():
    out = np.array([make_row(0.5, 0.5, 0.2, 0.4)])
    dets = parse_outputs([out], 640, 480)
    assert len(dets) == 1
    assert dets[0].person_xywh == (256, 144, 128, 192)
    assert dets[0].head_xywh == (256, 144, 128, 57)  # floor(0.3 * 192) = 57
    assert dets[0].confidence > 0.89


def test_non_person_classes_are_ignored():
    out = np.array([make_row(0.5, 0.5, 0.2, 0.4, class_id=2)])  # class 2 = car
    assert parse_outputs([out], 640, 480) == []


def test_low_confidence_is_filtered():
    out = np.array([make_row(0.5, 0.5, 0.2, 0.4, score=0.2)])
    assert parse_outputs([out], 640, 480, conf_threshold=0.4) == []


def test_nms_removes_duplicate_detections():
    out = np.array([make_row(0.5, 0.5, 0.2, 0.4, score=0.9), make_row(0.502, 0.502, 0.2, 0.4, score=0.6)])
    dets = parse_outputs([out], 640, 480)
    assert len(dets) == 1
    assert dets[0].confidence > 0.85  # the stronger one survives


def test_separate_people_are_both_kept():
    out = np.array([make_row(0.25, 0.5, 0.1, 0.3), make_row(0.75, 0.5, 0.1, 0.3)])
    assert len(parse_outputs([out], 640, 480)) == 2


def test_multiple_output_layers_are_combined():
    a = np.array([make_row(0.2, 0.5, 0.1, 0.3)])
    b = np.array([make_row(0.8, 0.5, 0.1, 0.3)])
    assert len(parse_outputs([a, b], 640, 480)) == 2


def test_boxes_are_clipped_to_frame():
    out = np.array([make_row(0.99, 0.5, 0.2, 0.4)])  # sticks out of the right edge
    det = parse_outputs([out], 640, 480)[0]
    x, y, w, h = det.person_xywh
    assert x + w <= 640


def test_empty_outputs():
    assert parse_outputs([], 640, 480) == []
    assert parse_outputs([np.zeros((0, 85))], 640, 480) == []
