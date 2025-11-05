# Examples

This directory contains example scripts demonstrating the LoRa CCTV system functionality.

## Available Examples

### jpeg_example.py

Demonstrates JPEG encoding and decoding without requiring hardware.

**What it does:**
1. Creates a test image with gradients and text
2. Encodes it to JPEG
3. Wraps it in the protocol frame
4. Simulates transmission
5. Decodes the frame
6. Decodes the JPEG image
7. Displays original and decoded images side-by-side

**How to run:**
```bash
cd examples
python jpeg_example.py
```

**Requirements:**
- numpy
- opencv-python

**Expected output:**
- Prints encoding/decoding statistics
- Shows compression ratio
- Calculates PSNR (image quality metric)
- Displays visual comparison

## Future Examples

- `cs_example.py` - Compressed Sensing encoding/decoding demo
- `loopback_test.py` - Test with virtual serial ports
- `benchmark.py` - Performance benchmarking
