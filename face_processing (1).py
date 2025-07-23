#!/usr/bin/env python3
"""
face_detector.py - Real-time face detection with OpenCV

Features:
- Webcam or video file input
- Haar cascade based detection
- FPS display
- Snapshot saving
- Minimal dependencies (OpenCV only)
"""

import cv2
import os
import sys
import time
import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time face detection using OpenCV Haar cascades"
    )
    parser.add_argument(
        "-i", "--input", type=str, default=0,
        help="Video source (camera index or file path)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Directory to save snapshots (optional)"
    )
    parser.add_argument(
        "-c", "--cascade", type=str, default=None,
        help="Path to Haar cascade XML file (optional)"
    )
    parser.add_argument(
        "-s", "--scale", type=float, default=1.1,
        help="Scale factor for detectMultiScale"
    )
    parser.add_argument(
        "-n", "--neighbors", type=int, default=5,
        help="Min neighbors for detectMultiScale"
    )
    return parser.parse_args()


def setup_detector(cascade_path=None):
    """Load Haar cascade classifier."""
    if cascade_path and os.path.exists(cascade_path):
        path = cascade_path
    else:
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        print("Error: Failed to load cascade classifier.", file=sys.stderr)
        sys.exit(1)
    return cascade


def setup_capture(source):
    """Initialize video capture."""
    try:
        index = int(source)
    except ValueError:
        index = source
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}", file=sys.stderr)
        sys.exit(1)
    return cap


def ensure_output_dir(directory):
    """Create output directory if needed."""
    if directory:
        os.makedirs(directory, exist_ok=True)


def detect_and_display(frame, detector, scale, neighbors):
    """Detect faces and draw rectangles; return face count."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return len(faces)


def save_snapshot(frame, directory, count):
    """Save a snapshot with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}_{count}.jpg"
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame)
    print(f"Saved snapshot: {path}")


def run():
    args = parse_args()
    detector = setup_detector(args.cascade)
    cap = setup_capture(args.input)
    ensure_output_dir(args.output)

    face_count_total = 0
    frame_count = 0
    start_time = time.time()

    print("Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error.", file=sys.stderr)
            break

        frame_count += 1
        faces = detect_and_display(frame, detector, args.scale, args.neighbors)
        face_count_total += faces

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame, f"Faces: {faces}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 0, 0), 2
        )
        cv2.putText(
            frame, f"FPS: {fps:.2f}",
            (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 0, 0), 2
        )

        cv2.imshow("Face Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and args.output:
            save_snapshot(frame, args.output, face_count_total)

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(
        f"Processed {frame_count} frames in {total_time:.2f} seconds "
        f"(Avg FPS: {avg_fps:.2f})"
    )
    print(f"Total faces detected: {face_count_total}")


if __name__ == "__main__":
    run()


# -----------------------------------------------------------------------------
# Developed by YourName © 2025
# License: MIT
# -----------------------------------------------------------------------------

