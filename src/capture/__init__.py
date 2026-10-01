"""Live capture layer (Phase: live perception).

FrameSource is the single interface every input implements:
    LiveCameraSource (OpenCV device index or stream URL)
    VideoFileSource  (MP4/AVI replay - regression testing)
    ImageFileSource  (single still - sanity testing)

Downstream processing must never know which source is active (spec §41:
camera/video/image feed the SAME pipeline). Sources only read and decode;
all motion/quality/screening decisions live in src/live/controller.py.
"""
