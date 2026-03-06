#!/usr/bin/env python3
"""
Benchmark: MP4 encoding latency vs multi-frame base64 encoding

Compares:
1. Current approach: Capture N frames → base64 encode each → send as multi-image
2. Video approach: Capture N frames → encode to MP4 → send as video

Tests different frame counts: 4, 8, 16, 32 frames
"""

import time
import base64
import io
import tempfile
import subprocess
import sys
from pathlib import Path

# Try to import optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not available, using PIL for frame generation")

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("Warning: av (PyAV) not available, will use ffmpeg subprocess")


def generate_test_frames(num_frames: int, width: int = 640, height: int = 480) -> list:
    """Generate synthetic test frames (simulating camera capture)."""
    frames = []
    for i in range(num_frames):
        # Create a frame with some variation (simulating real camera)
        if HAS_PIL:
            # Create RGB image with gradient + frame number
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            arr[:, :, 0] = (i * 10) % 256  # R varies by frame
            arr[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)  # G gradient
            arr[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)  # B gradient
            frame = Image.fromarray(arr, 'RGB')
            frames.append(frame)
        else:
            # Fallback: create dummy bytes
            frames.append(b'\x00' * (width * height * 3))
    return frames


def benchmark_base64_encoding(frames: list, quality: int = 85) -> dict:
    """Benchmark: Encode each frame to JPEG base64 (current approach)."""
    start = time.perf_counter()
    
    encoded_frames = []
    total_bytes = 0
    
    for frame in frames:
        buf = io.BytesIO()
        frame.save(buf, format='JPEG', quality=quality)
        jpeg_bytes = buf.getvalue()
        b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
        encoded_frames.append(f"data:image/jpeg;base64,{b64_str}")
        total_bytes += len(jpeg_bytes)
    
    elapsed = time.perf_counter() - start
    
    return {
        "method": "base64_multi_image",
        "num_frames": len(frames),
        "encoding_time_ms": elapsed * 1000,
        "total_bytes": total_bytes,
        "avg_frame_bytes": total_bytes / len(frames),
        "encoded_data": encoded_frames,  # For API comparison
    }


def benchmark_mp4_pyav(frames: list, fps: int = 10) -> dict:
    """Benchmark: Encode frames to MP4 using PyAV (in-memory)."""
    if not HAS_AV:
        return {"method": "mp4_pyav", "error": "PyAV not installed"}
    
    start = time.perf_counter()
    
    # Encode to in-memory buffer
    output_buffer = io.BytesIO()
    
    container = av.open(output_buffer, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=fps)
    stream.width = frames[0].width
    stream.height = frames[0].height
    stream.pix_fmt = 'yuv420p'
    # Fast encoding preset
    stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '23'}
    
    for frame in frames:
        av_frame = av.VideoFrame.from_image(frame)
        av_frame = av_frame.reformat(format='yuv420p')
        for packet in stream.encode(av_frame):
            container.mux(packet)
    
    # Flush
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    
    elapsed = time.perf_counter() - start
    video_bytes = output_buffer.getvalue()
    
    return {
        "method": "mp4_pyav",
        "num_frames": len(frames),
        "encoding_time_ms": elapsed * 1000,
        "total_bytes": len(video_bytes),
        "avg_frame_bytes": len(video_bytes) / len(frames),
        "video_data": video_bytes,
    }


def benchmark_mp4_ffmpeg(frames: list, fps: int = 10) -> dict:
    """Benchmark: Encode frames to MP4 using ffmpeg subprocess."""
    start = time.perf_counter()
    
    width, height = frames[0].size
    
    # Write frames to pipe, ffmpeg encodes
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name
    
    # ffmpeg command: read raw RGB from pipe, output MP4
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-f', 'mp4',
        '-movflags', '+faststart+frag_keyframe+empty_moov',
        tmp_path
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Write frames
        for frame in frames:
            raw_bytes = frame.tobytes()
            proc.stdin.write(raw_bytes)
        
        proc.stdin.close()
        proc.wait()
        
        # Read output
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        elapsed = time.perf_counter() - start
        
        return {
            "method": "mp4_ffmpeg",
            "num_frames": len(frames),
            "encoding_time_ms": elapsed * 1000,
            "total_bytes": len(video_bytes),
            "avg_frame_bytes": len(video_bytes) / len(frames),
        }
    except FileNotFoundError:
        return {"method": "mp4_ffmpeg", "error": "ffmpeg not installed"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def benchmark_webm_ffmpeg(frames: list, fps: int = 10) -> dict:
    """Benchmark: Encode frames to WebM (VP8) using ffmpeg - often faster than H.264."""
    start = time.perf_counter()
    
    width, height = frames[0].size
    
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp_path = tmp.name
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libvpx',
        '-b:v', '1M',
        '-deadline', 'realtime',
        '-cpu-used', '8',  # Fastest
        tmp_path
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        
        proc.stdin.close()
        proc.wait()
        
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        elapsed = time.perf_counter() - start
        
        return {
            "method": "webm_ffmpeg",
            "num_frames": len(frames),
            "encoding_time_ms": elapsed * 1000,
            "total_bytes": len(video_bytes),
            "avg_frame_bytes": len(video_bytes) / len(frames),
        }
    except FileNotFoundError:
        return {"method": "webm_ffmpeg", "error": "ffmpeg not installed"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("VIDEO ENCODING LATENCY BENCHMARK")
    print("Comparing multi-frame base64 vs on-the-fly video encoding")
    print("=" * 70)
    print()
    
    frame_counts = [4, 8, 16, 32]
    resolution = (640, 480)
    
    results = []
    
    for num_frames in frame_counts:
        print(f"\n{'─' * 70}")
        print(f"Testing with {num_frames} frames @ {resolution[0]}x{resolution[1]}")
        print(f"{'─' * 70}")
        
        # Generate test frames
        frames = generate_test_frames(num_frames, resolution[0], resolution[1])
        
        # Run benchmarks
        methods = [
            ("Base64 Multi-Image (current)", benchmark_base64_encoding),
            ("MP4 PyAV (in-memory)", benchmark_mp4_pyav),
            ("MP4 FFmpeg (subprocess)", benchmark_mp4_ffmpeg),
            ("WebM FFmpeg (subprocess)", benchmark_webm_ffmpeg),
        ]
        
        for name, func in methods:
            try:
                result = func(frames)
                if "error" in result:
                    print(f"  {name}: {result['error']}")
                else:
                    compression_ratio = (num_frames * resolution[0] * resolution[1] * 3) / result['total_bytes']
                    print(f"  {name}:")
                    print(f"    Encoding time: {result['encoding_time_ms']:.1f} ms")
                    print(f"    Total size:    {result['total_bytes'] / 1024:.1f} KB")
                    print(f"    Compression:   {compression_ratio:.1f}x")
                    print(f"    Per frame:     {result['encoding_time_ms'] / num_frames:.2f} ms/frame")
                    result["name"] = name
                    results.append(result)
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Encoding Time (ms)")
    print("=" * 70)
    print(f"{'Method':<30} | {'4 frames':>10} | {'8 frames':>10} | {'16 frames':>10} | {'32 frames':>10}")
    print("-" * 70)
    
    methods_summary = {}
    for r in results:
        method = r.get("name", r["method"])
        if method not in methods_summary:
            methods_summary[method] = {}
        methods_summary[method][r["num_frames"]] = r["encoding_time_ms"]
    
    for method, times in methods_summary.items():
        row = f"{method:<30}"
        for n in frame_counts:
            if n in times:
                row += f" | {times[n]:>10.1f}"
            else:
                row += f" | {'N/A':>10}"
        print(row)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Calculate overhead
    if results:
        base64_times = {r["num_frames"]: r["encoding_time_ms"] for r in results if r["method"] == "base64_multi_image"}
        mp4_times = {r["num_frames"]: r["encoding_time_ms"] for r in results if "mp4" in r["method"].lower()}
        
        if base64_times and mp4_times:
            for n in frame_counts:
                if n in base64_times and n in mp4_times:
                    overhead = list(mp4_times.values())[0] - base64_times[n]
                    print(f"  {n} frames: Video encoding adds ~{overhead:.0f}ms overhead")
                    break


if __name__ == "__main__":
    if not HAS_PIL:
        print("ERROR: PIL/Pillow is required. Install with: pip install Pillow numpy")
        sys.exit(1)
    
    run_benchmarks()


