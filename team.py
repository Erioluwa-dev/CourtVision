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
        Crop only the player's jersey.
        """

        x, y, w, h = bounding_box

        jersey_top = y + int(h * 0.20)
        jersey_bottom = y + int(h * 0.60)

        jersey = frame[
            jersey_top:jersey_bottom,
            x:x + w,
        ]

        return jersey

    def average_jersey_color(
        self,
        jersey,
    ):
        """
        Calculate the average BGR color
        of a jersey image.
        """

        if jersey.size == 0:
            return (0, 0, 0)

        average = cv2.mean(jersey)

        blue = int(average[0])
        green = int(average[1])
        red = int(average[2])

        return (
            blue,
            green,
            red,
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
            list(self.colors.values())
        )

        kmeans = KMeans(
            n_clusters=2,
            random_state=42,
            n_init="auto",
        )

        labels = kmeans.fit_predict(
            colors,
        )

        for player_id, label in zip(
            player_ids,
            labels,
        ):

            team = (
                "Team A"
                if label == 0
                else "Team B"
            )

            self.assign_team(
                player_id,
                team,
            )
