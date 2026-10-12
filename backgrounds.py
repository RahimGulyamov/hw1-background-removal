"""Runtime background selection; built-in gradients need no downloaded assets."""
from pathlib import Path

import cv2
import numpy as np


class Backgrounds:
    def __init__(self, width, height, mode, color, image=None, directory=None):
        self.width, self.height = width, height
        self.mode = mode
        self.colors = [tuple(color), (225, 225, 225), (32, 36, 42), (38, 155, 104)]
        self.color_index = 0
        self.image_index = 0
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        t = ((x / width + y / height) / 2)[..., None]
        # RGB endpoints converted to BGR for OpenCV composition.
        self.images = []
        for name, a, b in [
            ("studio", (229, 225, 216), (118, 132, 146)),
            ("sunset", (255, 207, 155), (147, 125, 177)),
            ("ocean", (183, 225, 226), (49, 84, 119)),
        ]:
            pic = (np.float32(a[::-1]) * (1 - t) + np.float32(b[::-1]) * t).astype(np.uint8)
            self.images.append((name, pic))
        paths = ([Path(image)] if image else [])
        if directory:
            paths += sorted(p for p in Path(directory).iterdir()
                            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
        paths = list(dict.fromkeys(p.resolve() for p in paths))
        user_images = []
        for path in paths:
            pic = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if pic is None:
                raise ValueError(f"Cannot decode background: {path}")
            h, w = pic.shape[:2]
            scale = max(width / w, height / h)
            pic = cv2.resize(pic, (max(width, round(w * scale)), max(height, round(h * scale))))
            y0, x0 = (pic.shape[0] - height) // 2, (pic.shape[1] - width) // 2
            user_images.append((path.name, pic[y0:y0 + height, x0:x0 + width].copy()))
        self.images = user_images + self.images
        self.solid = np.empty((height, width, 3), np.uint8)

    @property
    def label(self):
        if self.mode == "image":
            return "image:" + self.images[self.image_index][0]
        if self.mode == "color":
            return "color:" + ",".join(map(str, self.colors[self.color_index]))
        return "blur"

    def handle_key(self, key):
        before = self.label
        if key in (ord("1"), ord("2"), ord("3")):
            self.mode = {ord("1"): "color", ord("2"): "blur", ord("3"): "image"}[key]
        elif key in (ord("n"), ord("N")):
            self.image_index = (self.image_index + 1) % len(self.images)
            self.mode = "image"
        elif key in (ord("c"), ord("C")):
            self.color_index = (self.color_index + 1) % len(self.colors)
            self.mode = "color"
        return self.label != before

    def get(self, frame):
        if self.mode == "image":
            return self.images[self.image_index][1]
        if self.mode == "blur":
            small = cv2.resize(frame, (max(1, self.width // 4), max(1, self.height // 4)))
            return cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (self.width, self.height))
        self.solid[:] = self.colors[self.color_index][::-1]
        return self.solid
