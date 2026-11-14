class PrisonersDilemma:
    def __init__(self, reward=3, temptation=5, punishment=1, sucker=0):
        self.payoffs = {
            ("Cooperate", "Cooperate"): (reward, reward),
            ("Cooperate", "Defect"): (sucker, temptation),
            ("Defect", "Cooperate"): (temptation, sucker),
            ("Defect", "Defect"): (punishment, punishment)
        }

    def compute_payoff(self, action_a, action_b):
        return self.payoffs[(action_a, action_b)]