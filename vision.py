import cv2
import os


def load_image(path):
    return cv2.imread(path)

def image_shape(image):
    return image.shape

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    return frame

def draw_box(frame, x, y, w, h, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    return frame

def save_image(frame, output_path):
    cv2.imwrite(output_path, frame)
    print(f"Saved {output_path}")

def extract_frames_to_folder(video_path, output_dir, every_n=30):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % every_n == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
            print(f"  saved frame {frame_index}...")
        frame_index += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}/")
    return saved_count


if __name__ == "__main__":
    img = load_image("court.jpeg")
    print("Image shape:", image_shape(img))

    count = get_frame_count("footage.mp4")
    print("Total frames in video:", count)

    frame = extract_frame("footage.mp4", 10)
    annotated = draw_box(frame, 100, 100, 200, 300, "Player")
    save_image(annotated, "annotated_frame.jpg")

    extract_frames_to_folder("footage.mp4", "frames", every_n=30)
