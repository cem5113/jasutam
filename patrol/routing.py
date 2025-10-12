# patrol/routing.py
from __future__ import annotations
from typing import List, Dict, Optional
import itertools

def _mk_team_names(n: int) -> List[str]:
    base = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    if n <= len(base):
        return base[:n]
    # ek takım adları üret
    return base + [f"Team-{i}" for i in range(1, n - len(base) + 1)]

def roundrobin_assign(
    geoids: List[str],
    teams: List[str],
    max_per_team: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    Basit round-robin atama: GEOID'leri takımlara sırayla dağıtır.
    max_per_team verilirse, o sınırı aşmayacak şekilde kırpar.
    """
    plan: Dict[str, List[str]] = {t: [] for t in teams}
    cycle = itertools.cycle(teams) if teams else []
    for g in geoids:
        if not teams: break
        t = next(cycle)
        if max_per_team is not None and len(plan[t]) >= int(max_per_team):
            # sıradaki takıma at, gerekirse birkaç kez ilerle
            tried = 0
            while tried < len(teams) and len(plan[t]) >= int(max_per_team):
                t = next(cycle); tried += 1
            if tried >= len(teams) and len(plan[t]) >= int(max_per_team):
                break  # tüm takımlar dolu
        plan[t].append(str(g))
    return plan

def propose_routes(
    risky_geoids: List[str],
    team_count: int = 3,
    team_names: Optional[List[str]] = None,
    max_stops_per_team: int = 8,
) -> Dict[str, List[str]]:
    """
    Riskli GEOID listesini (ör. Top-K) alır, takımlara rota önerir (sadece sırayla).
    """
    teams = team_names or _mk_team_names(int(team_count))
    return roundrobin_assign(
        [str(g) for g in risky_geoids],
        teams=teams,
        max_per_team=int(max_stops_per_team),
    )
