"""
Export the CourtVision game record to disk.

Writes three kinds of output into an output directory:

  * frames.json  - every in-memory frame record (sampled by
                   MatchData.sample_every; the JSONL spool in the
                   pipeline output dir holds the full record).
  * summary.json - shots, passes, possession, player statistics.
  * heatmap_*.jpg - rendered player, ball and possession heatmaps.

Values are sanitised first because court-space positions carry
numpy scalars (homography output) that plain json.dump rejects.
"""

import json
import os

import cv2
import numpy as np


def _json_safe(value):
    """
    Recursively convert numpy scalars / arrays into JSON-safe
    Python types.
    """

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    return value


def build_summary(
    possession_tracker,
    pass_detector,
    shot_detector,
    player_stats,
    team_classifier,
):
    """
    Assemble the analytics summary as a plain dict.
    """

    return {
        "shots": shot_detector.get_shot_chart(),
        "passes": pass_detector.get_all_passes(),
        "pass_map": [
            {"from": pair[0], "to": pair[1], "count": count}
            for pair, count in sorted(
                pass_detector.get_pass_map().items()
            )
        ],
        "passing_network": pass_detector.get_passing_network(),
        "possession": {
            "timeline": possession_tracker.get_timeline(),
            "team_timeline": possession_tracker.get_team_possessions(),
            "team_percentages": possession_tracker.possession_percentage(),
        },
        "players": [
            {
                "id": player_id,
                "team": team_classifier.get_team(player_id),
                "frames_seen": stats["frames_seen"],
                "distance_pixels": round(stats["distance_pixels"], 2),
                "current_speed": round(stats["current_speed"], 2),
                "max_speed": round(stats["max_speed"], 2),
                "trajectory": stats["trajectory"],
            }
            for player_id, stats in sorted(
                player_stats.players.items()
            )
        ],
    }


def export_heatmaps(
    heatmaps,
    output_dir,
):
    """
    Render every heatmap grid to a JPEG and return the file paths.
    """

    os.makedirs(output_dir, exist_ok=True)

    paths = []

    for player_id, grid in heatmaps.get_player_heatmaps().items():

        path = os.path.join(
            output_dir,
            f"heatmap_player_{player_id}.jpg",
        )

        cv2.imwrite(
            path,
            heatmaps.render(grid),
        )

        paths.append(path)

    ball_path = os.path.join(
        output_dir,
        "heatmap_ball.jpg",
    )

    cv2.imwrite(
        ball_path,
        heatmaps.render(heatmaps.get_ball_heatmap()),
    )

    paths.append(ball_path)

    possession_path = os.path.join(
        output_dir,
        "heatmap_possession.jpg",
    )

    cv2.imwrite(
        possession_path,
        heatmaps.render(heatmaps.get_possession_heatmap()),
    )

    paths.append(possession_path)

    return paths


def export_game(
    output_dir,
    match,
    heatmaps,
    possession_tracker,
    pass_detector,
    shot_detector,
    player_stats,
    team_classifier,
):
    """
    Write the full game record (frames + summary + heatmaps) into
    output_dir and return the paths that were written.
    """

    os.makedirs(output_dir, exist_ok=True)

    frames_path = os.path.join(output_dir, "frames.json")

    with open(frames_path, "w") as f:
        json.dump(
            _json_safe(match.get_all_frames()),
            f,
            indent=2,
        )

    summary_path = os.path.join(output_dir, "summary.json")

    summary = build_summary(
        possession_tracker,
        pass_detector,
        shot_detector,
        player_stats,
        team_classifier,
    )

    with open(summary_path, "w") as f:
        json.dump(
            _json_safe(summary),
            f,
            indent=2,
        )

    heatmap_paths = export_heatmaps(
        heatmaps,
        output_dir,
    )

    return {
        "frames": frames_path,
        "summary": summary_path,
        "heatmaps": heatmap_paths,
    }
