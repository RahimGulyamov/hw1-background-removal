"""Real-time video background removal on CPU."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

from backgrounds import Backgrounds


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize without stretching."""
    h, w = image.shape[:2]
    if w * height > h * width:
        crop_w = max(1, round(h * width / height))
        left = (w - crop_w) // 2
        image = image[:, left:left + crop_w]
    else:
        crop_h = max(1, round(w * height / width))
        top = (h - crop_h) // 2
        image = image[top:top + crop_h, :]
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def blend(frame: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    result = frame.astype(np.float32) * a + background.astype(np.float32) * (1 - a)
    return np.clip(result, 0, 255).astype(np.uint8)


class BackgroundRemover:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backgrounds = Backgrounds(args.width, args.height, args.mode, args.color,
                                       args.background, args.background_dir)
        if args.backend == "rvm":
            from rvm import RVM
            self.model = RVM(args.downsample_ratio, args.rvm_threads)
        else:
            try:
                import mediapipe as mp
            except ImportError as error:
                raise RuntimeError(
                    "Install the optional backend: pip install -r requirements-mediapipe.txt"
                ) from error
            self.model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=args.model)
        self.previous: np.ndarray | None = None

    def process(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = cover(raw, self.args.width, self.args.height)
        if self.args.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.args.backend == "rvm":
            foreground, alpha = self.model.infer(rgb)
            return frame, blend(foreground, self.backgrounds.get(frame), alpha), alpha
        rgb.flags.writeable = False
        result = self.model.process(rgb)
        probability = result.segmentation_mask
        if probability is None:
            raise RuntimeError("Model did not return a segmentation mask")
        if self.args.mask_filter == "guided":
            guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            probability = cv2.ximgproc.guidedFilter(guide, probability, 4, 0.001)
        elif self.args.mask_filter == "gaussian":
            probability = cv2.GaussianBlur(probability, (3, 3), 0)
        probability = np.clip(probability, 0, 1)
        # EMA only where predictions agree. Fast changes reset locally to reduce trails.
        if self.previous is not None and self.args.smooth > 0:
            weight = np.where(np.abs(probability - self.previous) < 0.2,
                              self.args.smooth, 0).astype(np.float32)
            probability = probability * (1 - weight) + self.previous * weight
        self.previous = probability.copy()
        low = self.args.threshold - self.args.feather
        alpha = np.clip((probability - low) / (2 * self.args.feather), 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)  # Smoothstep, not a hard binary edge.
        return frame, blend(frame, self.backgrounds.get(frame), alpha), alpha

    def reset(self):
        self.previous = None
        if self.args.backend == "rvm":
            self.model.reset()

    def close(self) -> None:
        self.model.close()


class FrameStats:
    """Processing and application timing for completed frames."""
    def __init__(self):
        self.pipeline: list[float] = []
        self.loop: list[float] = []
        self.recent_pipeline: deque[float] = deque(maxlen=60)
        self.recent_loop: deque[float] = deque(maxlen=60)

    def add(self, pipeline: float, loop: float) -> None:
        self.pipeline.append(pipeline)
        self.loop.append(loop)
        self.recent_pipeline.append(pipeline)
        self.recent_loop.append(loop)

    @staticmethod
    def fps(values) -> float:
        return len(values) / sum(values) if values and sum(values) > 0 else 0.0

    def summary(self) -> dict:
        if not self.pipeline:
            return {"measured_frames": 0, "status": "insufficient_frames_after_warmup"}
        latency = np.asarray(self.pipeline) * 1000
        return {
            "status": "ok",
            "measured_frames": len(self.pipeline),
            "processing_seconds": sum(self.pipeline),
            "loop_seconds": sum(self.loop),
            "processing_fps": self.fps(self.pipeline),
            "application_fps": self.fps(self.loop),
            "processing_latency_mean_ms": float(latency.mean()),
            "processing_latency_p50_ms": float(np.percentile(latency, 50)),
            "processing_latency_p95_ms": float(np.percentile(latency, 95)),
            "loop_latency_p95_ms": float(np.percentile(np.asarray(self.loop) * 1000, 95)),
        }


class RealtimeRecorder:
    """Record at a fixed FPS while preserving elapsed playback time."""
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        codec = "mp4v" if path.suffix.lower() == ".mp4" else "MJPG"
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot write {path}; try an .avi filename")
        self.fps = fps
        self.frames = 0
        self.previous = None
        self.started = None

    def push(self, frame: np.ndarray, timestamp: float) -> None:
        if self.started is None:
            self.started = timestamp
            self.previous = frame.copy()
            self.writer.write(frame)
            self.frames = 1
            return
        elapsed = timestamp - self.started
        # Zero-order hold: represent elapsed time using the last completed frame.
        while self.frames / self.fps <= elapsed:
            self.writer.write(self.previous)
            self.frames += 1
        self.previous = frame.copy()

    def close(self, timestamp: float) -> None:
        if self.previous is not None:
            self.push(self.previous, timestamp)
        self.writer.release()


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True, timeout=3).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def rgb_color(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(v) for v in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use R,G,B, e.g. 40,100,180") from error
    if len(result) != 3 or any(v < 0 or v > 255 for v in result):
        raise argparse.ArgumentTypeError("Color must contain 3 integers in [0,255]")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Webcam index")
    source.add_argument("--input", type=Path, help="Video file instead of webcam")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30, help="Requested camera FPS, not a measurement")
    parser.add_argument("--mode", choices=["color", "blur", "image"], default="color")
    parser.add_argument("--backend", choices=["rvm", "mediapipe"], default="rvm", help="RVM: better temporal matting; MediaPipe: faster")
    parser.add_argument("--downsample-ratio", type=float, default=0.4, help="RVM internal scale (output stays full resolution)")
    parser.add_argument("--rvm-threads", type=int, default=min(8, os.cpu_count() or 1), help="ONNX Runtime CPU threads")
    parser.add_argument("--color", type=rgb_color, default=(40, 100, 180), help="Background R,G,B")
    parser.add_argument("--background", type=Path, help="Background image for --mode image")
    parser.add_argument("--background-dir", type=Path, default=Path(__file__).resolve().parent / "backgrounds", help="Load images from this folder; N cycles them")
    parser.add_argument("--model", type=int, choices=[0, 1], default=1, help="0: general 256x256; 1: landscape 256x144")
    parser.add_argument("--mask-filter", choices=["guided", "gaussian", "none"], default="guided")
    parser.add_argument("--smooth", type=float, default=0.25, help="Previous-mask weight, 0 disables temporal smoothing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--feather", type=float, default=0.15, help="Half-width of soft probability transition")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--view", choices=["split", "result", "mask"], default="split")
    parser.add_argument("--headless", action="store_true", help="Disable window; for servers/benchmarks only")
    parser.add_argument("--unpaced", action="store_true", help="File input only: do not wait for source FPS")
    parser.add_argument("--warmup", type=int, default=30, help="Initial processed frames excluded from final metrics/video")
    parser.add_argument("--seconds", type=float, default=0, help="Measured wall-clock seconds after warmup; 0 = unlimited")
    parser.add_argument("--max-frames", type=int, default=0, help="Total frames including warmup; 0 = unlimited")
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--output", type=Path, help="Record shown view to .mp4 or .avi; FPS is NOT a performance metric")
    parser.add_argument("--record-fps", type=float, default=30)
    args = parser.parse_args(argv)
    if min(args.width, args.height) < 32:
        parser.error("width/height must be >=32; use at least 320x240 for the assignment")
    if not 0 <= args.smooth < 1 or not 0 < args.feather <= 0.5:
        parser.error("smooth must be in [0,1), feather in (0,0.5]")
    if not args.feather <= args.threshold <= 1 - args.feather:
        parser.error("threshold +/- feather must lie in [0,1]")
    if min(args.seconds, args.warmup, args.max_frames) < 0:
        parser.error("seconds, warmup, max-frames must be non-negative")
    if min(args.camera_fps, args.record_fps, args.opencv_threads) <= 0:
        parser.error("FPS and opencv-threads must be positive")
    if not np.isfinite([args.seconds, args.camera_fps, args.record_fps]).all():
        parser.error("seconds and FPS must be finite")
    if not 0 < args.downsample_ratio <= 1 or args.rvm_threads < 1:
        parser.error("downsample-ratio must be in (0,1], rvm-threads >=1")
    if args.background_dir and not args.background_dir.is_dir():
        parser.error(f"Background folder does not exist: {args.background_dir}")
    if args.input and not args.input.is_file():
        parser.error(f"Video does not exist: {args.input}")
    if args.background and not args.background.is_file():
        parser.error(f"Background does not exist: {args.background}")
    if args.unpaced and not args.input:
        parser.error("--unpaced is only for a video file")
    if args.output and args.output.suffix.lower() not in {".mp4", ".avi"}:
        parser.error("--output must end with .mp4 or .avi")
    if args.output and args.input and args.output.resolve() == args.input.resolve():
        parser.error("Input and output must be different files")
    return args


def run(args) -> dict:
    if sys.platform.startswith("linux") and not args.headless:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("No graphical desktop. Run on a laptop, or add --headless --input video.mp4 on a server.")
    cv2.setNumThreads(args.opencv_threads)
    cv2.ocl.setUseOpenCL(False)
    remover = BackgroundRemover(args)
    cap = cv2.VideoCapture(str(args.input) if args.input else args.camera)
    recorder = None
    stats = FrameStats()
    by_background = {}
    events = []
    total = 0
    raw_size = None
    source_fps = 0.0
    measured_start = None
    reason = "finished"
    failure = None
    window = "CPU Background Removal | 1 Color  2 Blur  3 Image  N Next  C Color  R Reset  Q Quit"
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video/camera. Check permissions, --camera index, or file codec.")
        if not args.input:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this hint.
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        playback_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 30.0
        if args.input and source_fps <= 0:
            print("Source FPS unavailable: playback pacing assumes 30 FPS", file=sys.stderr)
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if args.output:
            view_width = args.width * (2 if args.view == "split" else 1)
            recorder = RealtimeRecorder(args.output, args.record_fps, (view_width, args.height))
        print(f"CPU | {cpu_name()} | {args.backend} | processing {args.width}x{args.height} | mode={args.mode}")
        print("Warmup first; Q/Esc stops. Processing FPS and application FPS are separate.")
        print("Keys: 1=color 2=blur 3=image N=next image C=next color R=reset matting Q=quit. Focus the window; use English layout.")
        while not args.max_frames or total < args.max_frames:
            loop_start = time.perf_counter()
            if total >= args.warmup and measured_start is None:
                measured_start = loop_start
            ok, raw = cap.read()
            if not ok:
                reason = "end_of_file" if args.input else "camera_read_failed"
                if not total or not args.input:
                    raise RuntimeError("No frame received from video/camera")
                break
            raw_size = [raw.shape[1], raw.shape[0]]
            process_start = time.perf_counter()
            background_label = remover.backgrounds.label
            frame, result, mask = remover.process(raw)
            process_seconds = time.perf_counter() - process_start
            # Timer stops only after segmentation, mask refinement, and composition.
            measured = total >= args.warmup
            if args.view == "split":
                shown = np.hstack([frame, result])
            elif args.view == "mask":
                shown = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                shown = result.copy()
            elapsed = time.perf_counter() - measured_start if measured_start is not None else 0
            fps = FrameStats.fps(stats.recent_pipeline)
            app_fps = FrameStats.fps(stats.recent_loop)
            label = (f"CPU | PROC {fps:.1f} FPS | APP {app_fps:.1f} FPS | {elapsed:.1f}s"
                     if measured else f"CPU | WARMUP {total + 1}/{args.warmup}")
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 55), (20, 20, 20), -1)
            cv2.putText(shown, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            subtitle = f"{args.width}x{args.height} | {args.backend} | {background_label} | " + (
                "ORIGINAL / RESULT" if args.view == "split" else args.view.upper())
            cv2.putText(shown, subtitle, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.rectangle(shown, (0, args.height - 24), (shown.shape[1], args.height), (20, 20, 20), -1)
            cv2.putText(shown, "1 Color  2 Blur  3 Image  N Next  C Color  R Reset  Q Quit", (8, args.height - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 225), 1, cv2.LINE_AA)
            if recorder and measured:
                recorder.push(shown, time.perf_counter())
            key = -1
            if not args.headless:
                cv2.imshow(window, shown)
                key = cv2.waitKey(1) & 0xFF
            if args.input and not args.unpaced:
                remaining = 1 / playback_fps - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
            if measured:
                loop_seconds = time.perf_counter() - loop_start
                stats.add(process_seconds, loop_seconds)
                by_background.setdefault(background_label, FrameStats()).add(process_seconds, loop_seconds)
            total += 1
            if key in (27, ord("q")):
                reason = "user_quit"
                break
            if remover.backgrounds.handle_key(key):
                events.append({"after_processed_frame": total, "background": remover.backgrounds.label})
                stats.recent_pipeline.clear()
                stats.recent_loop.clear()
            if key in (ord("r"), ord("R")):
                remover.reset()
                events.append({"after_processed_frame": total, "event": "reset_matting"})
            if not args.headless and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                reason = "window_closed"
                break
            if args.seconds and measured_start is not None and time.perf_counter() - measured_start >= args.seconds:
                reason = "time_limit"
                break
        else:
            reason = "frame_limit"
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception as error:
        reason = "error"
        failure = error
    finally:
        finished_at = time.perf_counter()
        if recorder:
            recorder.close(finished_at)
        cap.release()
        remover.close()
        if not args.headless:
            cv2.destroyAllWindows()
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
        "python": platform.python_version(),
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "device": "CPU", "backend": ("RVM MobileNetV3 FP32 / ONNX Runtime CPUExecutionProvider"
                                        if args.backend == "rvm" else "MediaPipe SelfieSegmentation CPU / TFLite XNNPACK"),
        "app_version": "2.0",
        "source": "file" if args.input else "webcam",
        "source_name": args.input.name if args.input else f"camera:{args.camera}",
        "source_reported_fps": source_fps if np.isfinite(source_fps) else None,
        "source_resolution": raw_size, "processing_resolution": [args.width, args.height],
        "model_input_resolution": ([args.width, args.height] if args.backend == "rvm" else [256, 144 if args.model == 1 else 256]),
        "rvm_internal_resolution": ([int(args.width * args.downsample_ratio), int(args.height * args.downsample_ratio)]
                                    if args.backend == "rvm" else None),
        "total_processed_frames": total, "warmup_frames_excluded": min(total, args.warmup),
        "infer_every_n_frames": 1, "stop_reason": reason,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "recorded_video_frames": recorder.frames if recorder else 0,
        "measurement": stats.summary(),
        "measurement_by_background": {name: value.summary() for name, value in by_background.items()},
        "background_events": events,
        "method": {
            "processing": "N / sum(t_resize+mirror+RGB+inference+mask_refinement+composition), no capture/UI/recording",
            "application": "N / elapsed loop time, including capture, processing, HUD, video writes, imshow/waitKey and playback pacing",
            "excluded": "model/camera/window startup, warmup, EOF probe, final cleanup, JSON serialization",
            "recording": "wall-clock CFR with repeats/drops; recorded frames are NOT performance samples",
        },
    }
    if args.backend == "rvm":
        import onnxruntime as ort
        from rvm import SHA256
        metrics["versions"]["onnxruntime"] = ort.__version__
        metrics["model_sha256"] = SHA256
    else:
        import mediapipe as mp
        metrics["versions"]["mediapipe"] = mp.__version__
    if failure:
        metrics["error"] = str(failure)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["measurement"], indent=2))
    print(f"Metrics saved: {args.metrics}")
    if failure:
        raise failure
    if not stats.pipeline:
        raise RuntimeError("No measured frames: use a longer input or reduce --warmup")
    return metrics


def main():
    args = parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
