import cv2


class TrajectoryTracker:
    """
    Draws player trajectories using the positions that PlayerStats
    already records, instead of keeping a second copy.
    """

    def __init__(
        self,
        player_stats,
    ):
        self._stats = player_stats

    def get_trajectory(
        self,
        player_id,
    ):
        """
        Return one player's trajectory.
        """

        stats = self._stats.get_player(
            player_id,
        )

        if stats is None:
            return []

        return stats["trajectory"]

    def get_all_trajectories(
        self,
    ):
        """
        Return all stored trajectories.
        """

        return {
            player_id: stats["trajectory"]
            for player_id, stats in self._stats.players.items()
        }

    def draw_trajectories(
        self,
        frame,
        max_points=50,
    ):
        # Draw last N trajectory points per player to avoid O(n²) rendering

        for positions in self.get_all_trajectories().values():

            tail = (
                positions[-max_points:]
                if len(positions) > max_points
                else positions
            )

            for position in tail:

                x, y = position

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1,
                )

        return frame
