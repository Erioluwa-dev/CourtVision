"""
Court zone classification (Phase 2 - Spatial Understanding).

Converts court coordinates (metres) into named zones:

    Paint | Left Corner | Right Corner | Left Wing | Right Wing |
    Top of Key | Midrange | Half Court | Backcourt

Coordinate convention:
    x - along the court length (metres)
    y - across the court width (metres), 0..COURT_WIDTH_M

For a HALF_COURT view the visible baseline is x=0, the attack hoop
sits at (0, WIDTH/2), and x grows toward midcourt. For a full court
the attack hoop sits at (COURT_LENGTH_M, WIDTH/2) and x < LENGTH/2
is the backcourt.
"""

import math

import config


def _corner_side(y, half_width):
    """
    Return "Left" / "Right" for a position near a sideline.
    """
    return "Left" if y < half_width else "Right"


def classify_position(x, y, half_court=None, court_length=None, court_width=None):
    """
    Classify a court-space position into a named zone.

    Args:
        x, y: court coordinates in metres.
        half_court: True when only one half of the court is visible.
            Defaults to config.HALF_COURT.
        court_length, court_width: court dimensions in metres.
            Defaults to config values.

    Returns:
        Zone name string, or "Unknown" when the position is outside
        the mapped court area.
    """
    if half_court is None:
        half_court = config.HALF_COURT

    if court_length is None:
        court_length = config.COURT_LENGTH_M

    if court_width is None:
        court_width = config.COURT_WIDTH_M

    if not (0.0 <= y <= court_width):
        return "Unknown"

    half_width = court_width / 2.0
    midcourt = court_length / 2.0

    # Convert to attack-relative local coordinates: local_x measures
    # distance from the attack baseline (0 at the hoop baseline).
    if half_court:
        # Visible half has the attack hoop on the baseline (x=0).
        if x < 0.0 or x > midcourt:
            return "Unknown"
        local_x = x
    else:
        # Attack hoop is at the far end; mirror so the same zone
        # logic applies on the attack half.
        if x < 0.0 or x > court_length:
            return "Unknown"
        local_x = court_length - x

    # Backcourt (full court only): past midcourt away from the hoop.
    if not half_court and local_x > midcourt:
        return "Backcourt"

    # Half court: the band nearest midcourt.
    if half_court and local_x > midcourt - config.HALF_COURT_BAND_M:
        return "Half Court"

    # ---------------------------------------------------------
    # Corner regions: within CORNER_SIDELINE_DISTANCE_M of a
    # sideline and not far from the baseline.
    # ---------------------------------------------------------
    sideline_distance = half_width - config.CORNER_SIDELINE_DISTANCE_M

    if abs(y - half_width) >= sideline_distance and local_x < 6.0:
        side = _corner_side(y, half_width)
        return f"{side} Corner"

    # ---------------------------------------------------------
    # Paint (key): FIBA 4.9m wide, 5.8m deep from the baseline.
    # ---------------------------------------------------------
    if (
        local_x < config.PAINT_DEPTH_M
        and abs(y - half_width) < config.PAINT_WIDTH_M / 2.0
    ):
        return "Paint"

    # ---------------------------------------------------------
    # Top of key: inside the free-throw circle.
    # ---------------------------------------------------------
    ft_centre = (config.PAINT_DEPTH_M, half_width)

    distance_to_ft = math.hypot(
        local_x - ft_centre[0],
        y - ft_centre[1],
    )

    if distance_to_ft <= config.FREE_THROW_CIRCLE_RADIUS_M:
        return "Top of Key"

    # ---------------------------------------------------------
    # Wings: lateral bands beside the key up to ~3m beyond the
    # free-throw line.
    # ---------------------------------------------------------
    wing_end = config.PAINT_DEPTH_M + 3.0

    if (
        config.PAINT_DEPTH_M <= local_x <= wing_end
        and abs(y - half_width) >= config.PAINT_WIDTH_M / 2.0
    ):
        side = _corner_side(y, half_width)
        return f"{side} Wing"

    # ---------------------------------------------------------
    # Everything else inside the visible half.
    # ---------------------------------------------------------
    return "Midrange"


def classify_player(player, half_court=None):
    """
    Classify a tracked player dict that carries court_position.
    """
    court_position = player.get("court_position")

    if court_position is None:
        return "Unknown"

    return classify_position(
        court_position[0],
        court_position[1],
        half_court=half_court,
    )


def classify_ball(ball, half_court=None):
    """
    Classify a tracked ball dict that carries court_position.
    """
    court_position = ball.get("court_position")

    if court_position is None:
        return "Unknown"

    return classify_position(
        court_position[0],
        court_position[1],
        half_court=half_court,
    )
