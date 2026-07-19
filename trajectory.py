import cv2
class TrajectoryTracker:
    """
    Stores every detected position for each player.
    """

    def __init__(self):
        self.trajectories = {}

    def add_position(
        self,
        player_id,
        position,
    ):
        """
        Add a new position for a player.
        """

        if player_id not in self.trajectories:
            self.trajectories[player_id] = []

        self.trajectories[player_id].append(position)

    def update(
        self,
        tracked_players,
    ):
        """
        Update trajectories for every tracked player
        in the current frame.
        """

        for player in tracked_players:

            self.add_position(
                player["id"],
                player["position"],
            )

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


    def draw_trajectories(
        self,
        frame,
        max_points=50,
    ):
        # Draw last N trajectory points per player to avoid O(n²) rendering

        for positions in self.trajectories.values():

            tail = positions[-max_points:] if len(positions) > max_points else positions

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
