# Badminton Point Detection System

A computer vision system that detects when a point is being played in badminton mathes versus when players are idling.

## Features

- **Point Detection**: Automatically identifies when players are actively playing a point vs when they are idling
- **Movement Analysis**: Uses MediaPipe pose detection to track player movement
- **Temporal Smoothing**: Applies smoothing to avoid false positives from brief movements
- **Visual Output**: Creates annotated videos showing the detection state
- **Data Export**: Exports detailed results to CSV for analysis

## How It Works

The system uses MediaPipe to detect human poses in each frame and calculates movement between consecutive frames. Key features:

1. **Movement Calculation**: Tracks 12 key body points (shoulders, elbows, wrists, hips, knees, ankles)
2. **State Detection**: Compares average movement against a threshold to determine if a point is being played
3. **Temporal Smoothing**: Uses a sliding window to smooth out detection and reduce noise
4. **State Transitions**: Tracks when points start and end, providing detailed statistics

## Files

- `badminton_point_detector.py` - Main production system
- `badminton_point_detector_tuning.py` - Tuning version with additional parameters
- `test_detection.py` - Analysis script
- `README.md` - This documentation

## Usage

### Basic Usage
```bash
# Activate virtual environment
source badminton_env/Scripts/activate  # On Windows
# or
source badminton_env/bin/activate      # On Linux/Mac

# Run main detection system
python badminton_point_detector.py --input your_video.mp4 --output result_video.mp4
```

### Advanced Usage
```bash
python badminton_point_detector.py \
    --input original_short.mp4 \
    --output point_detection_output.mp4 \
    --results point_detection_results.csv \
    --threshold 0.005 \
    --min-duration 30 \
    --smoothing 10
```

### Tuning Parameters

For fine-tuning the detection accuracy, use the tuning version:

```bash
# Test with debug mode to see what's happening
python badminton_point_detector_tuning.py --input original_short.mp4 --debug

# Adjust movement sensitivity (higher = more sensitive)
python badminton_point_detector_tuning.py --input original_short.mp4 --boost 1.2

# Adjust grace periods (higher = more stable)
python badminton_point_detector_tuning.py --input original_short.mp4 --idle-grace 20 --point-grace 15

# Adjust movement threshold (lower = more sensitive)
python badminton_point_detector_tuning.py --input original_short.mp4 --threshold 0.003
```

### Parameters

#### Main System Parameters
- `--input, -i`: Input video file (required)
- `--output, -o`: Output video file (optional)
- `--results, -r`: Results CSV file (default: point_detection_results.csv)
- `--threshold, -t`: Movement threshold for point detection (default: 0.005)
- `--min-duration, -d`: Minimum point duration in frames (default: 30)
- `--smoothing, -s`: Smoothing window size (default: 10)

#### Tuning System Additional Parameters
- `--boost`: Movement boost factor (default: 1.0)
- `--idle-grace`: Frames to wait before switching to idle (default: 15)
- `--point-grace`: Frames to wait before switching to point (default: 10)
- `--context`: Context window size for recent movement analysis (default: 5)
- `--debug`: Enable debug output

## Output

### Video Output
The output video shows:
- **Green indicator**: "POINT PLAYING" when a point is detected
- **Red indicator**: "IDLE" when players are not actively playing
- **Movement value**: Real-time movement measurement
- **Point counter**: Total number of points detected
- **Pose landmarks**: Overlaid on players for reference

### CSV Results
The CSV file contains:
- `frame`: Frame number
- `timestamp`: Time in seconds
- `movement`: Movement value for this frame
- `state`: Detected state (idle/point_playing)
- `point_number`: Point number (0 for idle frames)

## Example Results

From the test run on `original_short.mp4`:
- **Total frames**: 3,006
- **Points detected**: 13
- **Idle frames**: 1,874 (62.3%)
- **Point playing frames**: 1,132 (37.7%)
- **Average point duration**: 87.1 frames (2.9 seconds at 30fps)

## Requirements

- Python 3.7+
- MediaPipe
- OpenCV
- NumPy
- Pandas

Install with:
```bash
pip install -r requirements.txt
```

## Technical Details

### Movement Calculation
The system calculates movement by:
1. Detecting pose landmarks using MediaPipe
2. Computing Euclidean distance between corresponding landmarks in consecutive frames
3. Averaging movement across 12 key body points
4. Applying temporal smoothing over a configurable window

### State Detection Logic
- **Point Playing**: When average movement exceeds threshold
- **Idle**: When average movement is below threshold
- **Smoothing**: Uses deque with configurable window size to smooth transitions

### Enhanced Temporal Smoothing
The system includes sophisticated temporal smoothing to handle edge cases:

1. **Idle Grace Period**: Waits 15 frames before switching to idle (prevents false idle during brief pauses)
2. **Point Grace Period**: Waits 10 frames before switching to point (prevents false points during brief movements)
3. **Context Awareness**: Considers recent movement history to make smarter decisions
4. **Dynamic Grace Periods**: Adjusts grace periods based on movement context

### Performance
- Processes video at ~50 fps on modern hardware
- Memory efficient with streaming processing
- Configurable parameters for different video types

## Tuning Guide

### Common Tuning Scenarios

1. **Too many false points detected**:
   - Increase `--point-grace` (e.g., 15-20)
   - Decrease `--boost` (e.g., 0.8-0.9)
   - Increase `--threshold` (e.g., 0.006-0.008)

2. **Missing actual points**:
   - Decrease `--threshold` (e.g., 0.003-0.004)
   - Increase `--boost` (e.g., 1.1-1.3)
   - Decrease `--point-grace` (e.g., 5-8)

3. **False idle during active points**:
   - Increase `--idle-grace` (e.g., 20-25)
   - Decrease `--threshold` slightly

4. **Points ending too early**:
   - Increase `--idle-grace` (e.g., 20-30)
   - Check if movement threshold is too high

### Debug Mode
Use `--debug` flag to see detailed information about state transitions:
```bash
python badminton_point_detector_tuning.py --input your_video.mp4 --debug
```

This will show:
- Pending state changes
- Grace period calculations
- State transition confirmations

## Future Improvements

- Multi-player detection
- Shot type classification within points
- Real-time processing capabilities
- Advanced movement analysis (velocity, acceleration)
- Integration with scoring systems
