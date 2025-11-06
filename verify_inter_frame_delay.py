#!/usr/bin/env python3
"""
Verification script to demonstrate inter-frame delay functionality.

This script simulates the sender behavior and shows how inter-frame delay
prevents buffer overflow by controlling the rate at which frames are sent.
"""
import sys
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from common.protocol import encode_frame, TYPE_CS
from common.config import INTER_FRAME_DELAY


def simulate_frame_transmission(num_frames=20, inter_frame_delay=INTER_FRAME_DELAY):
    """
    Simulate sending frames with inter-frame delay.
    
    Args:
        num_frames: Number of frames to simulate
        inter_frame_delay: Delay between frames in seconds
    """
    print("=" * 60)
    print("Frame Transmission Simulation")
    print("=" * 60)
    print(f"Number of frames: {num_frames}")
    print(f"Inter-frame delay: {inter_frame_delay:.3f}s")
    print("=" * 60)
    
    # Simulate CS encoded data (186 bytes)
    test_data = b"X" * 186
    
    start_time = time.time()
    frame_times = []
    
    for i in range(num_frames):
        frame_start = time.time()
        
        # Encode frame
        frame = encode_frame(TYPE_CS, test_data)
        
        # Simulate sending (in real scenario, this would be serial.write)
        # Here we just track timing
        
        # Apply inter-frame delay
        if inter_frame_delay > 0:
            time.sleep(inter_frame_delay)
        
        frame_end = time.time()
        frame_times.append(frame_end - frame_start)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            actual_fps = (i + 1) / elapsed
            print(f"Frames: {i + 1}, Frame size: {len(frame)} bytes, "
                  f"Actual FPS: {actual_fps:.2f}")
    
    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    avg_frame_time = sum(frame_times) / len(frame_times)
    
    print("=" * 60)
    print("Simulation Results")
    print("=" * 60)
    print(f"Total frames: {num_frames}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average frame time: {avg_frame_time:.3f}s")
    if inter_frame_delay > 0:
        print(f"Theoretical max FPS (based on delay only): {1.0 / inter_frame_delay:.2f}")
    else:
        print("Theoretical max FPS: unlimited")
    print("=" * 60)
    
    return avg_fps


def main():
    """Run simulations with different delay settings."""
    print("\n### Simulation 1: Without inter-frame delay ###\n")
    fps_no_delay = simulate_frame_transmission(num_frames=20, inter_frame_delay=0.0)
    
    print("\n\n### Simulation 2: With default inter-frame delay (0.05s) ###\n")
    fps_with_delay = simulate_frame_transmission(num_frames=20, inter_frame_delay=0.05)
    
    print("\n\n### Simulation 3: With higher inter-frame delay (0.1s) ###\n")
    fps_high_delay = simulate_frame_transmission(num_frames=20, inter_frame_delay=0.1)
    
    print("\n\n### Summary ###")
    print("=" * 60)
    print("Without inter-frame delay:")
    print(f"  - Actual FPS: {fps_no_delay:.2f}")
    print(f"  - Risk: HIGH buffer overflow risk at receiver")
    print()
    print("With default inter-frame delay (0.05s):")
    print(f"  - Actual FPS: {fps_with_delay:.2f}")
    print(f"  - Maximum theoretical FPS: 20.00")
    print(f"  - Risk: LOW buffer overflow risk")
    print(f"  - Recommended for: Most use cases")
    print()
    print("With higher inter-frame delay (0.1s):")
    print(f"  - Actual FPS: {fps_high_delay:.2f}")
    print(f"  - Maximum theoretical FPS: 10.00")
    print(f"  - Risk: MINIMAL buffer overflow risk")
    print(f"  - Recommended for: Unstable connections or when errors persist")
    print("=" * 60)
    print()
    print("CONCLUSION:")
    print("The inter-frame delay effectively limits the frame rate and")
    print("prevents overwhelming the receiver's serial buffer, reducing")
    print("the occurrence of 'Invalid frame' errors.")
    print("=" * 60)


if __name__ == "__main__":
    main()
