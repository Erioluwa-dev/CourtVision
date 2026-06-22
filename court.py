import cv2
from detect import box_to_coords
import numpy as np

from vision import extract_frame

frame = extract_frame("/content/CourtVision/footage.mp4", 10)
cv2.imwrite("calibration_frame.jpg", frame)
print("Frame size:", frame.shape)

def build_homography(src_points, dst_points):
    return cv2.getPerspectiveTransform(
        np.float32(src_points),
        np.float32(dst_points),
    )
def get_default_homography():
    src_points = [
        [355,  380],   # halfcourt left sideline
        [1590, 218],   # halfcourt right sideline
        [1590, 655],   # baseline right corner
        [100,  780],   # baseline left corner
    ]
    dst_points = [
        [0,  47],   # halfcourt left
        [50, 47],   # halfcourt right
        [50, 94],   # baseline right
        [0,  94],   # baseline left
    ]
    return build_homography(src_points, dst_points)

def pixel_to_court(pixel_pt, H):
    pt = np.float32([[list(pixel_pt)]])
    result = cv2.perspectiveTransform(pt, H)
    x, y = result[0][0]
    return (round(float(x), 1), round(float(y), 1))


def get_player_foot_position(box):
    x, y, w, h, conf = box_to_coords(box)
    foot_x = x + w // 2
    foot_y = y + h
    return (foot_x, foot_y)

def classify_zone(court_x, court_y):
    in_paint = 17 <= court_x <= 33 and court_y <= 19
    near_basket = court_y <= 5.25
    corner_three = court_y <= 14 and (court_x <= 3 or court_x >= 47)
    arc_three = court_y > 14 and (
        (court_x - 25)**2 + (court_y - 5.25)**2 >= 23.75**2
    )
    past_halfcourt = court_y >= 47

    if past_halfcourt:
        return "backcourt"
    if corner_three or arc_three:
        return "three_point"
    if in_paint:
        return "paint"
    if near_basket:
        return "restricted_area"
    return "mid_range"
def get_player_zone(box, H):
    pixel_pos = get_player_foot_position(box)
    court_pos = pixel_to_court(pixel_pos, H)
    zone = classify_zone(court_pos[0], court_pos[1])
    return court_pos, zone
def classify_zone(court_x, court_y):
    if not (0 <= court_x <= 50 and 47 <= court_y <= 94):
        return "out_of_bounds"

    # distance from basket (at x=25, y=94)
    dist_from_basket = ((court_x - 25)**2 + (court_y - 94)**2) ** 0.5

    # paint: 16ft wide, 19ft deep from baseline
    in_paint = 17 <= court_x <= 33 and court_y >= 75

    # restricted area: 4ft radius around basket
    restricted = dist_from_basket <= 4

    # corner three: sideline within 14ft of baseline
    corner_three = court_y >= 80 and (court_x <= 3 or court_x >= 47)

    # arc three: beyond 23.75ft from basket
    arc_three = dist_from_basket >= 23.75 and not corner_three

    # halfcourt area
    at_halfcourt = court_y <= 52

    if arc_three or corner_three:
        return "three_point"
    if restricted:
        return "restricted_area"
    if in_paint:
        return "paint"
    if at_halfcourt:
        return "halfcourt"
    return "mid_range"
def enrich_player_positions(boxes, H):
    enriched = []
    for box in boxes:
        track_id = int(box.id[0]) if box.id is not None else -1
        court_pos, zone = get_player_zone(box, H)
        enriched.append({
            "player_id": track_id,
            "pixel_pos": get_player_foot_position(box),
            "court_pos": court_pos,
            "zone":      zone,
        })
        print(f"Player {track_id} → {zone} at {court_pos}")

    print(f"Enriched {len(enriched)} players with court positions")
    return enriched
if __name__ == "__main__":
    from detect import load_model
    from tracker import track_players_in_frame
    from vision import extract_frame

    VIDEO_PATH = "/content/CourtVision/footage.mp4"
    H = get_default_homography()

    model = load_model()
    frame = extract_frame(VIDEO_PATH, 10)
    boxes, result = track_players_in_frame(model, frame)

    enriched = enrich_player_positions(boxes, H)
    for p in enriched:
        print(p)
