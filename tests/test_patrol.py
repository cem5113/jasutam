from patrol.routing import propose_routes
from patrol.scoring import score_plan

def test_roundrobin():
    routes = propose_routes(["A","B","C","D"], team_count=2, max_stops_per_team=3)
    assert len(routes) == 2
    total = sum(len(v) for v in routes.values())
    assert total <= 6  # limit

def test_scoring():
    routes = {"Alpha":["A","B"], "Bravo":["C"]}
    s = score_plan(routes, risky_geoids=["A","C","X"])
    assert 0.0 <= s["coverage"] <= 1.0
