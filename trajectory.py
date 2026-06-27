"""
trajectory.py

Stores player movement trajectories throughout a game.
"""


class TrajectoryTracker:
    """
    Stores every detected position for each player.
    """

    def __init__(self):
        self.trajectories = {}

    def add_position(
        self,
        player_id,
        x,
        y,
    ):
        """
        Add a new position for a player.
        """

        if player_id not in self.trajectories:
            self.trajectories[player_id] = []

        self.trajectories[player_id].append((x, y))

    def get_trajectory(self, player_id):
        """
        Return one player's trajectory.
        """

        return self.trajectories.get(player_id, [])

    def get_all_trajectories(self):
        """
        Return all stored trajectories.
        """

        return self.trajectories
