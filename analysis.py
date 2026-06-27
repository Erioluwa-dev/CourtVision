from math import sqrt


def calculate_distance(point1, point2):
    """
    Calculate the distance between two points.
    """
    x1, y1 = point1
    x2, y2 = point2

    return sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def total_distance(trajectory):
    """
    Calculate the total distance travelled along a trajectory.
    """
    total = 0

    for i in range(len(trajectory) - 1):
        point1 = trajectory[i]
        point2 = trajectory[i + 1]

        total += calculate_distance(point1, point2)

    return total


def average_speed(distance, time_seconds):
    """
    Calculate average speed.
    """
    if time_seconds <= 0:
        return 0

    return distance / time_seconds


def zone_visit_counter(zones):
    """
    Count how many times each court zone was visited.
    """
    counts = {}

    for zone in zones:
        if zone not in counts:
            counts[zone] = 0

        counts[zone] += 1

    return counts


def player_summary(
    player_id,
    trajectory,
    time_seconds,
    zone_history,
):
    """
    Generate a summary of player movement analytics.
    """
    distance = total_distance(trajectory)

    speed = average_speed(
        distance,
        time_seconds,
    )

    zone_counts = zone_visit_counter(
        zone_history,
    )

    most_visited_zone = max(
        zone_counts,
        key=zone_counts.get,
    )

    return {
        "player_id": player_id,
        "distance": round(distance, 2),
        "average_speed": round(speed, 2),
        "most_visited_zone": most_visited_zone,
    }
def position_frequency(trajectory):
    counts = {}

    for position in trajectory:
        counts[position] = counts.get(position, 0) + 1

    return counts
def rank_players(player_summaries):
    return sorted(
        player_summaries,
        key=lambda player: player["distance"],
        reverse=True
    )
def team_total_distance(player_summaries):
    distances = [
        player["distance"]
        for player in player_summaries
    ]

    return sum(distances)
def team_average_speed(player_summaries):
    if not player_summaries:
        return 0

    speeds = [
        player["average_speed"]
        for player in player_summaries
    ]

    return round(
        sum(speeds) / len(speeds),
        2
    )
def fastest_player(player_summaries):
    if not player_summaries:
        return None

    return max(
        player_summaries,
        key=lambda player: player["average_speed"]
    )
def team_summary(player_summaries):
    return {
        "total_distance": team_total_distance(player_summaries),
        "average_speed": team_average_speed(player_summaries),
        "fastest_player": fastest_player(player_summaries),
        "rankings": rank_players(player_summaries),
    }
