"""
Webcam capture on its own thread.

The grab is threaded for one reason: cv2's read() blocks until the camera
hands over a frame, and a 30fps webcam will therefore stall a render loop
that is trying to run faster than that. The EEG decode must not wait on a
camera. The loop asks for whatever the most recent frame is and carries on;
if the camera is slow, stalled or absent, it gets the previous frame or None
and the dashboard keeps running.
"""

import threading
import time

import numpy as np


class Camera:
    def __init__(self, index=0, width=640, height=360, mirror=True):
        self.index = index
        self.size = (int(width), int(height))
        self.mirror = mirror
        self.error = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cap = None
        self.fps = 0.0

    # ------------------------------------------------------------------
    def start(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.error = "opencv not installed (pip install opencv-python)"
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        import cv2
        # On Windows the default backend can take several seconds to open a
        # camera; DirectShow is much quicker and is what the lab machine has.
        cap = None
        for backend in (getattr(cv2, "CAP_DSHOW", None), None):
            try:
                cap = (cv2.VideoCapture(self.index, backend) if backend
                       else cv2.VideoCapture(self.index))
                if cap is not None and cap.isOpened():
                    break
                if cap is not None:
                    cap.release()
                cap = None
            except Exception:
                cap = None
        if cap is None:
            self.error = f"no camera at index {self.index}"
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self._cap = cap

        last, n, t0 = None, 0, time.time()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if self.mirror:
                frame = frame[:, ::-1]
            # cv2 is BGR; everything downstream is RGB.
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            with self._lock:
                self._frame = rgb
            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                self.fps = n / (now - t0)
                n, t0 = 0, now
            last = now
        cap.release()
        self._cap = None
        _ = last

    # ------------------------------------------------------------------
    def frame(self):
        """Most recent frame as RGB uint8, or None."""
        with self._lock:
            return self._frame

    @property
    def ok(self):
        return self.error is None and self._frame is not None

    def status(self):
        if self.error:
            return self.error
        if self._frame is None:
            return "starting camera..."
        return f"{self.fps:.0f} fps"

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
