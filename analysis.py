def analyze_all_players(
    trajectories,
    game_duration,
    zone_histories,
):
    summaries = []

    for player_id, trajectory in trajectories.items():
        summary = player_summary(
            player_id,
            trajectory,
            game_duration,
            zone_histories.get(player_id, []),
        )

        summaries.append(summary)

    return summaries
def analytics_report(
    trajectories,
    game_duration,
    zone_histories,
):
    player_reports = analyze_all_players(
        trajectories,
        game_duration,
        zone_histories,
    )

    team_report = team_summary(
        player_reports
    )

    return {
        "players": player_reports,
        "team": team_report,
    }
