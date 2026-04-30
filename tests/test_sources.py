from __future__ import annotations

import cv2
import numpy as np

from valuelens.core.sources import FrameContext, LiveScreenSource


class FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.requested_region = None

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        self.requested_region = (x, y, width, height)
        return self.frame


def test_live_source_uses_bgr_capture_path() -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:, :, 0] = 32
    frame[:, :, 1] = 96
    frame[:, :, 2] = 192
    capture = FakeCapture(frame)
    source = LiveScreenSource(capture)

    color, gray = source.get_frame(FrameContext(view_rect=(3, 4, 16, 12), dpr=1.0))

    assert capture.requested_region == (3, 4, 16, 12)
    assert color is frame
    assert np.array_equal(gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
