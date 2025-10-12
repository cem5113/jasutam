# patrol/suggestions.py
from __future__ import annotations
from typing import List, Dict, Iterable

from .scoring import score_plan

def suggest_route_actions(
    routes: Dict[str, List[str]],
    risky_geoids_topk: Iterable[str],
) -> List[str]:
    """
    Skorlara göre kısa aksiyon önerileri üretir.
    """
    s = score_plan(routes, risky_geoids_topk)
    tips: List[str] = []
    if s["coverage"] < 0.5:
        tips.append("Kapsama düşük görünüyor; riskli ilk hücrelere ek durak eklenebilir.")
    if s["overlap_penalty"] > 0.10:
        tips.append("Takımlar arasında hücre çakışması var; rotaları ayrıştırarak çeşitliliği artırın.")
    if s["diversity"] < 0.7:
        tips.append("Benzer bölgeler tekrar edilmiş; farklı bölgelerle çeşitlendirme önerilir.")
    if not tips:
        tips.append("Plan dengeli görünüyor; mevcut dağılım korunabilir.")
    return tips

def text_summary(routes: Dict[str, List[str]], risky_geoids_topk: Iterable[str]) -> str:
    s = score_plan(routes, risky_geoids_topk)
    lines = [
        f"Kapsama: %{s['coverage']*100:.0f}",
        f"Çeşitlilik: %{s['diversity']*100:.0f}",
        f"Çakışma cezası: %{s['overlap_penalty']*100:.0f}",
        f"Bileşik skor: {s['composite']:.2f}",
    ]
    for t in suggest_route_actions(routes, risky_geoids_topk):
        lines.append(f"- {t}")
    return "\n".join(lines)
