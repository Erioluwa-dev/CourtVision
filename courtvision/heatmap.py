"""
Heatmap generation (Phase 2 - Spatial Understanding).

Accumulates court-space positions into 2D grids and renders them
as colour-mapped images.

Three heatmaps are produced:

  * Player movement - per tracked player id.
  * Ball - every ball position.
  * Possession - ball positions while a player possesses it.
"""

import math

import cv2
import numpy as np

from . import config


class HeatmapGenerator:
    """
    Accumulates and renders court-space heatmaps.
    """

    def __init__(
        self,
        bin_size_m=None,
        court_length=None,
        court_width=None,
    ):
        if bin_size_m is None:
            bin_size_m = config.HEATMAP_BIN_SIZE_M

        if court_length is None:
            court_length = config.COURT_LENGTH_M

        if court_width is None:
            court_width = config.COURT_WIDTH_M

        self.bin_size = bin_size_m

        self.nx = max(int(math.ceil(court_length / bin_size_m)), 1)
        self.ny = max(int(math.ceil(court_width / bin_size_m)), 1)

        self.player_heatmaps = {}

        self.ball_heatmap = np.zeros((self.ny, self.nx), dtype=np.float32)

        self.possession_heatmap = np.zeros(
            (self.ny, self.nx),
            dtype=np.float32,
        )

        # player_id -> court position for the possession map.
        self.possession_player = None

    def _bin(self, court_position):
        x, y = court_position

        if x < 0 or y < 0:
            return None

        bx = int(x / self.bin_size)
        by = int(y / self.bin_size)

        if bx >= self.nx or by >= self.ny:
            return None

        return by, bx

    def update(
        self,
        tracked_players,
        mapped_ball,
        possession_player=None,
    ):
        """
        Accumulate one frame of court-space positions.
        """

        for player in tracked_players:

            court_position = player.get("court_position")

            if court_position is None:
                continue

            cell = self._bin(court_position)

            if cell is None:
                continue

            player_id = player["id"]

            if player_id not in self.player_heatmaps:

                self.player_heatmaps[player_id] = np.zeros(
                    (self.ny, self.nx),
                    dtype=np.float32,
                )

            self.player_heatmaps[player_id][cell] += 1.0

        self.possession_player = possession_player

        if mapped_ball is None:
            return

        court_position = mapped_ball.get("court_position")

        if court_position is None:
            return

        cell = self._bin(court_position)

        if cell is None:
            return

        self.ball_heatmap[cell] += 1.0

        if possession_player is not None:
            self.possession_heatmap[cell] += 1.0

    def get_player_heatmap(self, player_id):
        """
        Return a player's movement grid (ny, nx) or None.
        """

        return self.player_heatmaps.get(player_id)

    def get_ball_heatmap(self):
        """
        Return the ball position grid (ny, nx).
        """

        return self.ball_heatmap

    def get_possession_heatmap(self):
        """
        Return the possession position grid (ny, nx).
        """

        return self.possession_heatmap

    def render(
        self,
        heatmap,
        canvas_size=(560, 300),
    ):
        """
        Render a heatmap grid as a colour-mapped BGR image.

        Cells with no visits render dark; dense cells render
        toward red (cv2.COLORMAP_JET).
        """

        normalized = np.zeros_like(heatmap)

        maximum = float(heatmap.max()) if heatmap.size else 0.0

        if maximum > 0:
            normalized = (heatmap / maximum * 255.0).astype(np.uint8)
        else:
            normalized = heatmap.astype(np.uint8)

        colored = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        return cv2.resize(
            colored,
            canvas_size,
            interpolation=cv2.INTER_AREA,
        )
