import re
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from tools import query_hotels, load_hotels

load_dotenv()

def _df_to_md(df) -> str:
    headers = [str(h) for h in getattr(df, "columns", [])]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        vals = [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def _query_summary(params: Dict[str, Any], total: int) -> str:
    parts = []
    city = params.get("city")
    country = params.get("country")
    sort_by = (params.get("sort_by") or "star_rating").lower()
    limit = int(params.get("limit", 5) or 5)
    if city:
        parts.append(f"in **{city.title()}**")
    if country:
        parts.append(f"in **{country.title()}**")
    if not (city or country):
        parts.append("globally")
    th = []
    ms = params.get("min_star", 0) or 0
    mc = params.get("min_clean", 0) or 0
    mco = params.get("min_comfort", 0) or 0
    mf = params.get("min_facilities", 0) or 0
    if ms > 0: th.append(f"⭐ ≥ {ms:g}")
    if mc > 0: th.append(f"cleanliness ≥ {mc:g}")
    if mco > 0: th.append(f"comfort ≥ {mco:g}")
    if mf > 0: th.append(f"facilities ≥ {mf:g}")
    if th:
        parts.append("(" + ", ".join(th) + ")")
    sort_map = {"star_rating": "star rating", "cleanliness": "cleanliness", "comfort": "comfort", "facilities": "facilities"}
    sort_nice = sort_map.get(sort_by, sort_by)
    shown = min(limit, total) if total else 0
    where = ", ".join(parts)
    return f"**Showing {shown} of {total} result(s) {where} — sorted by {sort_nice}.**"

def _extract_location_tokens(low: str, cities: List[str], countries: List[str]) -> Tuple[Optional[str], Optional[str]]:
    city_hit = None
    country_hit = None
    for ctry in sorted(countries, key=len, reverse=True):
        if re.search(rf"\b{re.escape(ctry)}\b", low):
            country_hit = ctry
            break
    for cty in sorted(cities, key=len, reverse=True):
        if re.search(rf"\b{re.escape(cty)}\b", low):
            city_hit = cty
            break
    return city_hit, country_hit

def parse_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state["messages"][-1][1] if state.get("messages") else ""
    low = text.lower().strip()
    df = load_hotels()
    ds_cities = df["city"].dropna().unique().tolist() if "city" in df.columns else []
    ds_countries = df["country"].dropna().unique().tolist() if "country" in df.columns else []
    params = {
        "city": None,
        "country": None,
        "min_star": 0.0,
        "min_clean": 0.0,
        "min_comfort": 0.0,
        "min_facilities": 0.0,
        "sort_by": "star_rating",
        "limit": 5,
    }
    m = re.search(r"\b(?:top|list)\s+(\d{1,2})\b", low) or re.search(r"\b(\d{1,2})\b", low)
    if m:
        try:
            params["limit"] = max(1, min(int(m.group(1)), 10))
        except ValueError:
            pass
    if "cleanliness" in low:
        params["sort_by"] = "cleanliness"
    elif "comfort" in low:
        params["sort_by"] = "comfort"
    elif "facilities" in low:
        params["sort_by"] = "facilities"
    elif any(w in low for w in ["best rated", "top rated", "highest rated", "star", "stars", "rating"]):
        params["sort_by"] = "star_rating"
    mstar = re.search(r"star(?:s| rating)?\s*(?:>=|≥|at\s+least)?\s*([0-9]+(?:\.[0-9]+)?)", low)
    if mstar:
        try:
            params["min_star"] = float(mstar.group(1))
        except ValueError:
            pass
    m_city = re.search(r"\bcity\s*[:=]\s*([a-z][a-z\s\-']+)", low)
    if m_city:
        params["city"] = m_city.group(1).strip().lower()
    m_country = re.search(r"\bcountry\s*[:=]\s*([a-z][a-z\s\-']+)", low)
    if m_country:
        params["country"] = m_country.group(1).strip().lower()
    if not params["city"] and not params["country"]:
        m_in = re.search(r"\bin\s+([a-z][a-z\s\-']+)", low)
        if m_in:
            token = m_in.group(1)
            token = re.split(r"\s+(?:with|by|having|where|and)\b|[.,;:!?]", token, maxsplit=1)[0]
            token = token.strip().lower()
            if token in ds_countries:
                params["country"] = token
            elif token in ds_cities:
                params["city"] = token
    if not params["city"] and not params["country"]:
        cty, ctry = _extract_location_tokens(low, ds_cities, ds_countries)
        if ctry:
            params["country"] = ctry
        if cty:
            params["city"] = cty
    state["params"] = params
    return state

def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    p = state.get("params", {})
    result = query_hotels(
        city=p.get("city"),
        country=p.get("country"),
        min_star=p.get("min_star", 0),
        min_clean=p.get("min_clean", 0),
        min_comfort=p.get("min_comfort", 0),
        min_facilities=p.get("min_facilities", 0),
        sort_by=p.get("sort_by", "star_rating"),
        limit=p.get("limit", 5),
    )
    state["tool_result"] = result
    return state

def respond_node(state: Dict[str, Any]) -> Dict[str, Any]:
    result = state.get("tool_result", None)
    params = state.get("params", {}) or {}
    if result is None:
        reply = "No results available."
    else:
        try:
            total = getattr(result, "shape", (0, 0))[0]
            if total == 0:
                summary = _query_summary(params, total)
                reply = summary + "\n\nNo hotels found matching your criteria.\n\nTry loosening filters (e.g., lower star threshold or remove city/country)."
            else:
                summary = _query_summary(params, total)
                table_md = _df_to_md(result)
                reply = summary + "\n\n" + table_md
        except Exception:
            reply = str(result)
    state.setdefault("messages", []).append(("assistant", reply))
    return state

def build_agent():
    graph = StateGraph(dict)
    graph.add_node("parse", parse_node)
    graph.add_node("tool", tool_node)
    graph.add_node("respond", respond_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "tool")
    graph.add_edge("tool", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
