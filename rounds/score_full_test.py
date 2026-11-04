"""Score a round on the full test: python rounds/score_full_test.py rounds/results/full_test_14T.csv"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1])
d = d[d.in_test]  # colour in stock at least half the season


def score(g):
    e = (g.forecast - g.actual_demand).abs().sum() / g.actual_demand.sum()
    b = (g.forecast - g.actual_demand).sum() / g.actual_demand.sum()
    return f"{e:.0%} ({b:+.0%})"


print("colour seasons", d[d.lead_weeks == 0].shape[0])
print("all  ", score(d))
print("old  ", score(d[~d.new_colour]))
print("new  ", score(d[d.new_colour]))
for lead, g in d.groupby("lead_weeks"):
    print(f"lead {lead:2}", score(g))
for store, g in d.groupby("store"):
    print("store", store, score(g))
