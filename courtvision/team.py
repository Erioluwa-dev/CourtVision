import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Assign players to basketball teams.
    """

    def __init__(self):
        self.teams = {}
        self.colors = {}
        # team name -> cluster centre colour (persisted so team names
        # do not flip when player ids change or colours drift).
        self.team_centers = {}

    def assign_team(
        self,
        player_id,
        team,
    ):
        """
        Store a player's team.
        """

        self.teams[player_id] = team

    def get_team(
        self,
        player_id,
    ):
        """
        Return a player's team.
        """

        return self.teams.get(
            player_id,
            "Unknown",
        )

    def get_all_teams(
        self,
    ):
        """
        Return every assigned player.
        """

        return self.teams

    def extract_jersey(
        self,
        frame,
        bounding_box,
    ):
        """
        Crop the centre band of the player box (the torso), avoiding
        the head, arms and background at the edges.
        """

        x, y, w, h = bounding_box

        x1 = x + int(w * 0.15)
        x2 = x + int(w * 0.85)
        y1 = y + int(h * 0.25)
        y2 = y + int(h * 0.55)

        return frame[y1:y2, x1:x2]

    def average_jersey_color(
        self,
        jersey,
    ):
        """
        Median BGR colour of the saturated jersey pixels, so skin,
        background and shadow pixels do not pollute the team colour.
        """

        if jersey.size == 0:
            return (0, 0, 0)

        hsv = cv2.cvtColor(
            jersey,
            cv2.COLOR_BGR2HSV,
        )

        saturated = hsv[:, :, 1] >= 60

        if saturated.any():

            bgr = jersey[saturated]

            median = np.median(
                bgr.reshape(-1, 3),
                axis=0,
            )

            return (
                int(median[0]),
                int(median[1]),
                int(median[2]),
            )

        average = cv2.mean(jersey)

        return (
            int(average[0]),
            int(average[1]),
            int(average[2]),
        )

    def store_color(
        self,
        player_id,
        color,
    ):
        """
        Store a player's average jersey color.
        """

        self.colors[player_id] = color

    def get_all_colors(
        self,
    ):
        """
        Return every stored jersey color.
        """

        return self.colors

    def classify_teams(
        self,
    ):
        """
        Group players into two teams
        using KMeans clustering.
        """

        if len(self.colors) < 2:
            return

        player_ids = list(
            self.colors.keys()
        )

        colors = np.array(
            list(self.colors.values()),
            dtype=np.float32,
        )

        kmeans = KMeans(
            n_clusters=2,
            random_state=42,
            n_init="auto",
        )

        labels = kmeans.fit_predict(
            colors,
        )

        centers = kmeans.cluster_centers_

        assigned = self._assign_clusters(
            centers,
        )

        for player_id, label in zip(
            player_ids,
            labels,
        ):

            self.assign_team(
                player_id,
                assigned[label],
            )

        self.team_centers = {
            team: centers[label].tolist()
            for label, team in assigned.items()
        }

    def _assign_clusters(
        self,
        centers,
    ):
        """
        Map each cluster index to a stable team name by matching its
        centre colour to the closest previously established team,
        so re-classification never flips Team A / Team B.
        """

        teams = ["Team A", "Team B"]

        used = set()

        assigned = {}

        for label, center in enumerate(centers):

            if not self.team_centers:

                team = teams[label]

            else:

                best_team = None
                best_distance = float("inf")

                for team, old_center in self.team_centers.items():

                    distance = np.linalg.norm(
                        center - np.array(old_center)
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_team = team

                if best_team in used:

                    team = next(
                        (
                            candidate
                            for candidate in teams
                            if candidate not in used
                        ),
                        best_team,
                    )

                else:

                    team = best_team

            assigned[label] = team

            used.add(team)

        return assigned
