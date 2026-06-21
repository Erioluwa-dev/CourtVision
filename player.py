class Player:
    def __init__(self, name, team, jersey, ppg, apg, is_all_star):
        self.name = name
        self.team = team
        self.jersey = jersey
        self.ppg = ppg
        self.apg = apg
        self.is_all_star = is_all_star

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            team=data["team"],
            jersey=data["jersey"],
            ppg=data["ppg"],
            apg=data["apg"],
            is_all_star=data["is_all_star"],
        )

    def __repr__(self):
        return f"{self.name} ({self.team}): {self.ppg} PPG"

    def is_elite_scorer(self):
        return self.ppg >= 25

    def combined_production(self):
        return round(self.ppg + self.apg, 1)

    def __lt__(self, other):
        return self.ppg < other.ppg


class AllStar(Player):
    def __repr__(self):
        return f"{self.name} ({self.team}): {self.ppg} PPG [All-Star]"


class Team:
    def __init__(self, name):
        self.name = name
        self.roster = []

    def add_player(self, player):
        self.roster.append(player)

    def top_scorer(self):
        return max(self.roster, key=lambda p: p.ppg)


if __name__ == "__main__":
    p1 = Player(
        "Shai Gilgeous-Alexander", "Oklahoma City Thunder", "2", 31.1, 6.6, True
    )
    p2 = AllStar("Stephen Curry", "Golden State Warriors", "30", 26.6, 4.7, True)
    sample = {
        "name": "Kevin Durant",
        "team": "Houston Rockets",
        "jersey": "7",
        "ppg": 26.0,
        "apg": 4.8,
        "is_all_star": True,
    }
    kd = Player.from_dict(sample)
    print(kd)

    print(p1)
    print("Combined production:", p1.combined_production())
    print("Elite scorer:", p1.is_elite_scorer())
    print(p2)

    thunder = Team("Oklahoma City Thunder")
    thunder.add_player(p1)
    print("Thunder top scorer:", thunder.top_scorer())
