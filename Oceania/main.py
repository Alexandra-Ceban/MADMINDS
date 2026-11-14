"""Walu Season: starter main.py (the greedy baseline).

This file already speaks the evaluator's protocol. The part to change is the
Strategy class. As shipped it is pure greedy: try every spot once, then keep
going back to whichever spot has the best average so far. It runs end to end
and it does NOT reach the threshold.

Check it locally on the practice reef:

    python evaluation_script.py main.py --dataset train

Rules of the protocol (see task_description.md):
  * read test-input.json from the current directory (bare filename),
  * for every cast print one spot number on stdout, then read one line from
    stdin with the catch in kg,
  * start a fresh Strategy every day (spot numbers are reshuffled daily),
  * anything else you want to print goes to stderr.
"""

import json
import sys
import math



class Strategy:
    def __init__(self, n_spots, n_casts):
        self.n_spots = n_spots
        self.n_casts = n_casts
        self.casts = [0] * n_spots
        self.total = [0.0] * n_spots
        self.total_casts = 0

    def choose(self):
        # 1. Explore every spot at least once
        for spot in range(self.n_spots):
            if self.casts[spot] == 0:
                return spot

        # 2. Calculate UCB score for each spot
        scores = []

        for spot in range(self.n_spots):
            average = self.total[spot] / self.casts[spot]

            exploration = math.sqrt(
                2 * math.log(self.total_casts) / self.casts[spot]
            )

            score = average + exploration
            scores.append(score)

        # 3. Choose the spot with the highest score
        return max(range(self.n_spots), key=lambda s: scores[s])

    def update(self, spot, catch):
        self.casts[spot] += 1
        self.total[spot] += catch
        self.total_casts += 1

        
    

def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
