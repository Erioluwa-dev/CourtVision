from exceptions import InvalidStatError

def get_tier(ppg):
    if ppg < 0:
        raise InvalidStatError(f"PPG cannot be negative: {ppg}")
    if ppg >= 30: return "MVP Candidate"
    elif ppg >= 25: return "Elite Scorer"
    elif ppg >= 20: return "Starter"
    elif ppg >= 15: return "Rotational piece"
    else: return "Role player"

def get_star_label(is_all_star):
    return "[ALL-STAR]" if is_all_star else ""

def get_commentary(player):
    tier = get_tier(player["ppg"])
    star = get_star_label(player["is_all_star"])
    return f"{player['name']} | {tier} | {player['ppg']} PPG | {star}"

def count_allstars(players):
    count = 0
    for player in players:
        if player["is_all_star"]:
            count += 1
    return count
