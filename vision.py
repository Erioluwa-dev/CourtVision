import cv2


# ---------------------------------------------------
# Video Functions
# ---------------------------------------------------

def load_video(video_path):
    """
    Open a video for processing.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {video_path}"
        )

    return video


def read_frame(video):
    """
    Read the next frame.
    """
    return video.read()


def release_video(video):
    """
    Release an opened video.
    """
    video.release()


# ---------------------------------------------------
# Drawing Functions
# ---------------------------------------------------

def draw_box(
    frame,
    x,
    y,
    width,
    height,
    label,
):
    """
    Draw a labeled bounding box.
    """
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return frame
