import os
import cv2


# ---------------------------------------------------
# Image Functions
# ---------------------------------------------------

def load_image(path):
    """
    Load an image from disk.
    """
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


def save_image(frame, output_path):
    """
    Save an image to disk.
    """
    cv2.imwrite(output_path, frame)
    print(f"Saved {output_path}")


def image_shape(image):
    """
    Return the dimensions of an image.
    """
    return image.shape


def resize_frame(frame, width, height):
    """
    Resize a frame.
    """
    return cv2.resize(frame, (width, height))


def to_rgb(frame):
    """
    Convert a BGR image to RGB.
    """
    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


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


def get_frame_count(video_path):
    """
    Return the number of frames in a video.
    """
    video = load_video(video_path)

    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    release_video(video)

    return frame_count


def extract_frame(video_path, frame_number):
    """
    Extract a single frame from a video.
    """
    video = load_video(video_path)

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number,
    )

    success, frame = read_frame(video)

    release_video(video)

    if not success:
        return None

    return frame


def extract_frames_to_folder(
    video_path,
    output_dir,
    every_n=30,
):
    """
    Save every Nth frame from a video.
    """
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    video = load_video(video_path)

    frame_index = 0
    saved_count = 0

    while True:

        success, frame = read_frame(video)

        if not success:
            break

        if frame_index % every_n == 0:

            output_path = os.path.join(
                output_dir,
                f"frame_{frame_index}.jpg",
            )

            save_image(
                frame,
                output_path,
            )

            saved_count += 1

        frame_index += 1

    release_video(video)

    print(
        f"Extracted {saved_count} frames."
    )

    return saved_count


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


# ---------------------------------------------------
# Demo
# ---------------------------------------------------

if __name__ == "__main__":

    image = load_image("court.jpeg")

    print(
        "Image shape:",
        image_shape(image),
    )

    frame_count = get_frame_count(
        "footage.mp4",
    )

    print(
        "Total frames:",
        frame_count,
    )

    frame = extract_frame(
        "footage.mp4",
        10,
    )

    if frame is not None:

        annotated = draw_box(
            frame,
            100,
            100,
            200,
            300,
            "Player",
        )

        save_image(
            annotated,
            "annotated_frame.jpg",
        )

    extract_frames_to_folder(
        "footage.mp4",
        "frames",
        every_n=30,
    )
