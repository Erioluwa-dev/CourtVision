
from tracker import centroid_distance


class PlayerStats:
    """
    Tracks live player statistics throughout a game.
    """

    def __init__(self):
        """
        Initialize player statistics.
        """

        self.players = {}

    def _create_player(
        self,
        player_id,
    ):
        """
        Create a new player record.
        """

        self.players[player_id] = {

            "frames_seen": 0,

            "distance_pixels": 0.0,

            "current_speed": 0.0,

            "max_speed": 0.0,

            "last_position": None,

            "trajectory": [],

        }

    def update(
        self,
        tracked_players,
    ):
        """
        Update every tracked player's statistics.
        """

        for player in tracked_players:

            player_id = player["id"]

            position = player["position"]

            if player_id not in self.players:

                self._create_player(
                    player_id,
                )

            stats = self.players[player_id]

            stats["frames_seen"] += 1

            stats["trajectory"].append(
                position,
            )

            if stats["last_position"] is not None:

                distance = centroid_distance(

                    stats["last_position"],

                    position,

                )

                stats["distance_pixels"] += distance

                stats["current_speed"] = distance

                if distance > stats["max_speed"]:

                    stats["max_speed"] = distance

            stats["last_position"] = position

    def get_player(
        self,
        player_id,
    ):
        """
        Return one player's statistics.
        """

        return self.players.get(
            player_id,
        )

    def get_all_players(
        self,
    ):
        """
        Return all player statistics.
        """

        return self.players

    def total_players(
        self,
    ):
        """
        Return the number of tracked players.
        """

        return len(
            self.players,
        )

    def reset(
        self,
    ):
        """
        Reset every player's statistics.
        """

        self.players.clear()
