# Image Data Processing Refactoring Documentation

## Overview

This document describes the refactoring of the image data processing logic in `main.py`. The refactoring improves code organization, maintainability, and testability while preserving all existing functionality.

## Changes Made

### 1. New Classes

#### FrameStatistics (Dataclass)
**Purpose**: Encapsulates statistics tracking for frame fragments.

**Responsibilities**:
- Accumulates statistics across multiple frame fragments
- Calculates compression metrics
- Provides clean reset functionality

**Key Methods**:
- `add_fragment(stats)`: Add statistics for a received fragment
- `reset()`: Clear all accumulated statistics
- `get_compression_summary()`: Calculate and return compression statistics

**Benefits**:
- Eliminates manual list/counter management in main loop
- Provides type safety with dataclass
- Centralizes statistics calculation logic

#### ImageProcessor
**Purpose**: Processes encoded chunks into decoded image frames.

**Responsibilities**:
- Decode video chunks using the provided decoder
- Track codec changes and configuration updates
- Manage frame statistics through FrameStatistics
- Handle decoding errors gracefully

**Key Methods**:
- `process_chunk(chunk, stats)`: Process a single encoded chunk
- `get_statistics()`: Get the current frame statistics tracker

**Benefits**:
- Encapsulates all chunk processing logic in one place
- Maintains codec state internally
- Simplifies error handling and recovery
- Easier to test in isolation

#### ImageDisplay
**Purpose**: Handles image display and statistics reporting.

**Responsibilities**:
- Display decoded frames using OpenCV
- Print transmission and compression statistics
- Automatically reset statistics after display

**Key Methods**:
- `show_frame(image, statistics)`: Display a frame and print its statistics

**Benefits**:
- Separates display logic from processing logic
- Makes it easy to change display format without affecting processing
- Clean interface for displaying frames

### 2. Removed Functions

The following functions were removed as their functionality is now handled by the new classes:

- `process_frame()`: Replaced by `ImageProcessor.process_chunk()`
- `display_frame()`: Replaced by `ImageDisplay.show_frame()`

### 3. Updated Code

#### Main Processing Loop
**Before**:
```python
def _main_processing_loop(frame_queue, decoder, stop_event):
    current_codec = None
    pending_stats = []
    pending_fragments = 0
    
    while not stop_event.is_set():
        chunk, stats = frame_queue.get(timeout=0.2)
        
        decoded_frames, current_codec, pending_fragments = process_frame(
            chunk, stats, decoder, current_codec, pending_stats, pending_fragments
        )
        
        if decoded_frames:
            image = decoded_frames[-1]
            pending_stats, pending_fragments = display_frame(
                image, pending_stats, pending_fragments, WINDOW_TITLE
            )
```

**After**:
```python
def _main_processing_loop(frame_queue, decoder, stop_event):
    processor = ImageProcessor(decoder)
    display = ImageDisplay(WINDOW_TITLE)
    
    while not stop_event.is_set():
        chunk, stats = frame_queue.get(timeout=0.2)
        
        decoded_frames = processor.process_chunk(chunk, stats)
        
        if decoded_frames:
            image = decoded_frames[-1]
            display.show_frame(image, processor.get_statistics())
```

**Benefits**:
- Cleaner, more readable code
- Fewer parameters passed between functions
- State is managed internally by classes
- Easier to understand the flow

## Testing

### Unit Tests
New comprehensive unit tests have been added in `tests/test_image_processor.py`:

- `TestFrameStatistics`: Tests for the FrameStatistics class
  - Initialization
  - Fragment addition
  - Reset functionality
  - Compression summary calculation
  - Edge cases (empty statistics)

- `TestImageProcessor`: Tests for the ImageProcessor class
  - Initialization
  - Successful chunk processing
  - Codec change detection
  - Decoding error handling

- `TestImageDisplay`: Tests for the ImageDisplay class
  - Initialization
  - Frame display with statistics

### Running Tests
```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test module
python -m unittest tests.test_image_processor -v
```

## Backward Compatibility

All existing functionality has been preserved:
- Frame reception and decoding works identically
- Statistics display format is unchanged
- Error handling behavior is the same
- All command-line arguments are supported

## Benefits of Refactoring

1. **Improved Code Organization**: Related functionality is now grouped in classes
2. **Better Separation of Concerns**: Processing, statistics, and display are separate
3. **Enhanced Testability**: Each class can be tested independently with mocks
4. **Easier Maintenance**: Changes to one aspect don't affect others
5. **Type Safety**: Use of dataclasses and type hints improves code quality
6. **Reduced Complexity**: Main loop is simpler and easier to understand
7. **Better Encapsulation**: Internal state is hidden from callers

## Migration Guide

If you have code that depends on the old function signatures:

### Old Code
```python
decoded_frames, current_codec, pending_fragments = process_frame(
    chunk, stats, decoder, current_codec, pending_stats, pending_fragments
)

pending_stats, pending_fragments = display_frame(
    image, pending_stats, pending_fragments, window_title
)
```

### New Code
```python
processor = ImageProcessor(decoder)
display = ImageDisplay(window_title)

# Processing
decoded_frames = processor.process_chunk(chunk, stats)

# Display
display.show_frame(image, processor.get_statistics())
```

## Future Enhancements

The refactored structure makes it easier to add new features:

1. **Alternative Display Backends**: Easy to create new display classes (e.g., for web streaming)
2. **Statistics Logging**: Can subclass FrameStatistics to log to files
3. **Processing Pipelines**: Can chain multiple processors together
4. **Custom Decoders**: Easy to swap decoder implementations
5. **Performance Monitoring**: Can add timing/profiling to individual classes

## Performance Impact

The refactoring has minimal performance impact:
- No additional memory allocations in hot paths
- Method calls have negligible overhead
- Statistics calculation is identical to before
- Frame processing path is unchanged

## Conclusion

This refactoring significantly improves the code quality and maintainability of the image data processing logic while preserving all existing functionality. The new structure is more modular, testable, and easier to extend with new features.
