"""Transparent, rule-based match scoring used by the Searcher (0-100, with reasons and concerns).

Weights: subject relevance 40, education level 20, cost and funding 20, location and mode 10,
GPA 10. Scoring is deterministic so rankings are explainable and testable.
"""

from __future__ import annotations

from scholarpath.schemas import Opportunity, StudentProfile
from scholarpath.text import keywords, normalize_name


def _relevance(profile: StudentProfile, opp: Opportunity, reasons: list[str], concerns: list[str]) -> int:
    wanted = keywords(profile.field_of_study, profile.career_goals, *profile.interests, *profile.skills)
    # A field tag counts when all its words appear in the profile ("machine learning", not "machine").
    matched = [f for f in opp.fields if f != "any" and keywords(f) and keywords(f) <= wanted]
    covered = keywords(*matched)
    text_hits = (wanted & keywords(opp.name, opp.description)) - covered
    score = min(40, 14 * len(matched) + 3 * len(text_hits))
    if matched:
        reasons.append("Matches your focus on " + ", ".join(matched[:3]))
    elif text_hits:
        reasons.append("Related to your interest in " + ", ".join(sorted(text_hits)[:3]))
    if "any" in opp.fields and score < 15:
        if not (matched or text_hits):
            reasons.append("Open to applicants from any field")
        score = 15
    if score == 0:
        concerns.append("Weak match with your field and goals")
    return score


def _level(profile: StudentProfile, opp: Opportunity, reasons: list[str], concerns: list[str]) -> int:
    if not opp.levels:
        return 10
    allowed = ", ".join(level.replace("_", " ") for level in opp.levels)
    if profile.education_level is None:
        concerns.append(f"Check that your education level fits ({allowed})")
        return 10
    if profile.education_level in opp.levels:
        reasons.append(f"Open to {profile.education_level.replace('_', ' ')} applicants")
        return 20
    concerns.append(f"Intended for {allowed} applicants")
    return 0


def _cost(profile: StudentProfile, opp: Opportunity, reasons: list[str], concerns: list[str]) -> int:
    cons = profile.constraints
    if opp.funded:
        reasons.append(f"Funded: {opp.cost_or_funding}")
        return 20
    if opp.cost_usd == 0:
        reasons.append("Free to take")
        return 20
    if opp.cost_usd is None:
        return 10
    if cons.needs_funding:
        concerns.append(f"Costs about ${opp.cost_usd:,} and is not funded")
        return 5
    if cons.max_budget_usd is not None:
        if opp.cost_usd <= cons.max_budget_usd:
            reasons.append(f"Within your ${cons.max_budget_usd:,} budget")
            return 18
        concerns.append(f"About ${opp.cost_usd:,}, above your ${cons.max_budget_usd:,} budget")
        return 0
    return 12


def _location(profile: StudentProfile, opp: Opportunity, reasons: list[str], concerns: list[str]) -> int:
    cons = profile.constraints
    if cons.study_mode == "online":
        if opp.mode == "online":
            reasons.append("Fully online")
            return 10
        concerns.append("Requires some in-person attendance")
        return 3 if opp.mode == "hybrid" else 0
    if cons.part_time and opp.mode == "in_person" and opp.type.value == "degree":
        concerns.append("May not fit a part-time schedule")
    if not cons.countries:
        return 7
    if opp.country.lower() in {c.lower() for c in cons.countries}:
        reasons.append(f"Located in {opp.country}, one of your preferred countries")
        return 10
    if opp.mode == "online":
        reasons.append("Online, so location is no barrier")
        return 8
    concerns.append(f"Located in {opp.country}, outside your preferred countries")
    return 2


def _gpa(profile: StudentProfile, opp: Opportunity, reasons: list[str], concerns: list[str]) -> int:
    if opp.min_gpa is None:
        return 10
    if profile.gpa is None:
        concerns.append(f"Minimum GPA {opp.min_gpa}/4.0; add yours to check")
        return 5
    if profile.gpa >= opp.min_gpa:
        reasons.append(f"Your GPA meets the {opp.min_gpa}/4.0 minimum")
        return 10
    concerns.append(f"Your GPA is below the {opp.min_gpa}/4.0 minimum")
    return 0


def score_opportunity(profile: StudentProfile, opp: Opportunity) -> Opportunity:
    reasons: list[str] = []
    concerns: list[str] = []
    total = sum(
        part(profile, opp, reasons, concerns) for part in (_relevance, _level, _cost, _location, _gpa)
    )
    return opp.model_copy(update={"match_score": total, "match_reasons": reasons, "concerns": concerns})


def dedupe(opportunities: list[Opportunity]) -> list[Opportunity]:
    seen: dict[str, Opportunity] = {}
    for opp in opportunities:
        key = f"{normalize_name(opp.name)}|{normalize_name(opp.provider)}"
        seen.setdefault(key, opp)
    return list(seen.values())


def rank(profile: StudentProfile, opportunities: list[Opportunity], top_k: int) -> list[Opportunity]:
    scored = [score_opportunity(profile, opp) for opp in dedupe(opportunities)]
    scored.sort(key=lambda o: o.match_score, reverse=True)
    return scored[:top_k]
