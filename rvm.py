"""Robust Video Matting, MobileNetV3 FP32, CPU only; weights from official v1.0.0."""
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).resolve().parent / "models" / "rvm_mobilenetv3_fp32.onnx"
SHA256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"


class RVM:
    def __init__(self, ratio=0.4, threads=1):
        if not MODEL.is_file():
            raise RuntimeError("RVM weights are missing: extract the complete ZIP including models/")
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("RVM weights failed SHA-256 verification; extract models/ again")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(MODEL), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.ratio = np.array([ratio], dtype=np.float32)
        self.reset()

    def reset(self):
        self.states = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]

    def infer(self, rgb):
        src = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        fgr, pha, *self.states = self.session.run(None, {
            "src": src, "downsample_ratio": self.ratio,
            **{f"r{i+1}i": value for i, value in enumerate(self.states)},
        })
        # Keep predicted foreground colours to reduce background contamination.
        foreground = np.clip(fgr[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255)
        return foreground, np.clip(pha[0, 0], 0, 1)

    def close(self):
        self.session = None
        self.states = []
