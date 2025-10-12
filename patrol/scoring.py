# patrol/scoring.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd

# ------------------------------------------------------------
# 1) Kapsama (iki farklı kullanım)
# ------------------------------------------------------------
def coverage_score_weighted(
    route_geoids: Iterable[str],
    geo_scores: pd.DataFrame,
    score_col: str,
) -> float:
    """
    Ağırlıklı kapsama (%): (rotadaki GEOID'lerin toplam skoru) / (tüm GEOID toplam skoru) * 100
    - geo_scores en az ['GEOID', score_col] içermeli (pencereye göre özetlenmiş)
    - Dönüş: 0..100 (yüzde)
    """
    if geo_scores is None or len(geo_scores) == 0 or "GEOID" not in geo_scores.columns or score_col not in geo_scores.columns:
        return 0.0
    route_set = set(map(str, route_geoids))
    sub = geo_scores[geo_scores["GEOID"].astype(str).isin(route_set)]
    num = float(sub[score_col].sum())
    den = float(geo_scores[score_col].sum()) or 1.0
    return 100.0 * num / den


def coverage_score_binary(
    route_geoids: Iterable[str],
    risky_geoids: Iterable[str],
) -> float:
    """
    İkili kapsama (0..1): planlanan rota GEOID'lerinin riskli seti karşılama oranı.
    - Dönüş: 0..1 (yüksek = iyi)
    """
    R = set(map(str, risky_geoids))
    P = set(map(str, route_geoids))
    if not R:
        return 0.0
    return len(R & P) / float(min(len(R), max(1, len(P))))


# ------------------------------------------------------------
# 2) Tek rota benzerliği/çeşitliliği
# ------------------------------------------------------------
def overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    """
    İki rota arası örtüşme (Jaccard benzeri): |kesişim| / |birleşim|  → 0..1
    """
    A, B = set(map(str, a)), set(map(str, b))
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def diversity_score_route(
    route_geoids: Iterable[str],
    recent_routes: Optional[List[List[str]]] = None,
) -> float:
    """
    Tek rota için çeşitlilik skoru = 1 - (son N rota ile en yüksek örtüşme).
    - Düşük = kötü (çok benzer), yüksek = iyi (farklı).
    - Dönüş: 0..1
    """
    if not recent_routes:
        return 1.0
    max_ov = max(overlap_ratio(route_geoids, r) for r in recent_routes if r)
    return max(0.0, 1.0 - max_ov)


# ------------------------------------------------------------
# 3) Çoklu takım/rota metrikleri
# ------------------------------------------------------------
def overlap_penalty(routes: Dict[str, List[str]]) -> float:
    """
    Aynı GEOID'in birden fazla takımda bulunma oranı (0 iyi, 1 kötü).
    - routes: {"Alpha": [...geoids...], "Bravo": [...], ...}
    """
    seen: Dict[str, int] = {}
    for lst in routes.values():
        for g in lst:
            k = str(g)
            seen[k] = seen.get(k, 0) + 1
    if not seen:
        return 0.0
    dup = sum(1 for _, c in seen.items() if c > 1)
    return dup / float(max(1, len(seen)))


def diversity_score_routes(routes: Dict[str, List[str]]) -> float:
    """
    Takımlar arası çeşitlilik: farklı hücre oranı (1 iyi).
    - Dönüş: 0..1  (toplam içindeki benzersiz oran)
    """
    all_list = [str(g) for lst in routes.values() for g in lst]
    if not all_list:
        return 0.0
    uniq = len(set(all_list))
    return uniq / float(len(all_list))


# ------------------------------------------------------------
# 4) Bileşik plan skoru
# ------------------------------------------------------------
def score_plan(
    routes: Dict[str, List[str]],
    risky_geoids: Iterable[str],
    w_cov: float = 0.6,
    w_div: float = 0.3,
    w_ovp: float = 0.3,
) -> Dict[str, float]:
    """
    Basit bileşik skorlar:
      - coverage (0..1, yüksek iyi)       → riskli set karşılama
      - diversity (0..1, yüksek iyi)      → takımlar arası çeşitlilik
      - overlap_penalty (0..1, düşük iyi) → aynı hücreyi birden çok takımın seçmesi
      - composite = coverage*w_cov + diversity*w_div - overlap*w_ovp (alt sınır 0)
    """
    planned = [g for lst in routes.values() for g in lst]
    cov = coverage_score_binary(planned, risky_geoids)
    div = diversity_score_routes(routes)
    ovp = overlap_penalty(routes)
    comp = max(0.0, cov * w_cov + div * w_div - ovp * w_ovp)
    return {
        "coverage": round(cov, 4),
        "diversity": round(div, 4),
        "overlap_penalty": round(ovp, 4),
        "composite": round(comp, 4),
    }
