from __future__ import annotations
import os
import json
import re
from typing import Dict, Any
import requests

def _regex_parse_fallback(query: str) -> Dict[str, Any]:
    q = query.strip()
    ql = q.lower()
    params: Dict[str, Any] = {
        "city": None,
        "country": None,
        "min_star": 0.0,
        "min_clean": 0.0,
        "min_comfort": 0.0,
        "min_facilities": 0.0,
        "sort_by": "star_rating",
        "limit": 5,
    }
    m = re.search(r"\btop\s+(\d+)\b", ql)
    if m:
        try:
            params["limit"] = max(1, min(10, int(m.group(1))))
        except: pass
    m = re.search(r"\b(?:at\s+least\s+)?(\d+)\s*[- ]?\s*star", ql)
    if m:
        try:
            params["min_star"] = float(m.group(1))
        except: pass
    m = re.search(r"\bsorted\s+by\s+(star(?:_?rating)?|rating|clean(?:liness)?|comfort|facilit(?:y|ies))\b", ql)
    if m:
        key = m.group(1)
        if key.startswith("clean"):
            params["sort_by"] = "cleanliness"
        elif key.startswith("comfort"):
            params["sort_by"] = "comfort"
        elif key.startswith("facilit"):
            params["sort_by"] = "facilities"
        else:
            params["sort_by"] = "star_rating"
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s\-]+?)(?:\s*,\s*([A-Za-z][A-Za-z\s\-]+))?(?:$|\b)", q)
    if m:
        loc1 = m.group(1).strip()
        loc2 = m.group(2).strip() if m.group(2) else None
        if loc2:
            params["city"] = loc1.title()
            params["country"] = loc2.title()
        else:
            params["city"] = loc1.title()
    return params

def parse_query_with_llama3(query: str) -> Dict[str, Any]:
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{base_url}/api/chat"
    model = os.getenv("OLLAMA_MODEL", "llama3:8b")

    system_msg = (
        "Convert the user's hotel request into STRICT JSON with keys:\n"
        '{ "city": string|null, "country": string|null, '
        '"min_star": number, "min_clean": number, "min_comfort": number, "min_facilities": number, '
        '"sort_by": "star_rating"|"cleanliness"|"comfort"|"facilities", "limit": number }\n'
        "- Defaults: sort_by=star_rating, limit=5\n"
        "- 'top N' => limit=N (1..10)\n"
        "- '4-star' => min_star=4\n"
        "- If mentions cleanliness/comfort/facilities, set sort_by accordingly\n"
        "- Location: 'in City' or 'in City, Country'. If only one token, put it in city; downstream will fallback.\n"
        "Return ONLY the JSON."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9},
    }

    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message", {}) or {}).get("content", "").strip()
        if not content:
            return _regex_parse_fallback(query)

        if content.startswith("```"):
            content = content.strip("` \n")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        params = json.loads(content)

        def _num(val, default):
            try:
                return float(val)
            except:
                return default

        out = {
            "city": params.get("city") or None,
            "country": params.get("country") or None,
            "min_star": _num(params.get("min_star", 0.0), 0.0),
            "min_clean": _num(params.get("min_clean", 0.0), 0.0),
            "min_comfort": _num(params.get("min_comfort", 0.0), 0.0),
            "min_facilities": _num(params.get("min_facilities", 0.0), 0.0),
            "sort_by": (params.get("sort_by") or "star_rating").lower().replace(" ", "_"),
            "limit": int(min(max(int(params.get("limit", 5)), 1), 10)),
        }
        sb = out["sort_by"]
        if sb in {"rating", "star"}:
            sb = "star_rating"
        if sb not in {"star_rating", "cleanliness", "comfort", "facilities"}:
            sb = "star_rating"
        out["sort_by"] = sb
        return out
    except Exception:
        return _regex_parse_fallback(query)

def parse_query_with_groq(query: str) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _regex_parse_fallback(query)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    system_msg = (
        'Return ONLY JSON: {"city":string|null,"country":string|null,"min_star":number,"min_clean":number,'
        '"min_comfort":number,"min_facilities":number,"sort_by":"star_rating"|"cleanliness"|"comfort"|"facilities","limit":number}'
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("` \n")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        params = json.loads(content)

        def _num(val, default):
            try:
                return float(val)
            except:
                return default
        sb = (params.get("sort_by") or "star_rating").lower().replace(" ", "_")
        if sb in {"rating", "star"}: sb = "star_rating"
        if sb not in {"star_rating", "cleanliness", "comfort", "facilities"}:
            sb = "star_rating"

        return {
            "city": params.get("city") or None,
            "country": params.get("country") or None,
            "min_star": _num(params.get("min_star", 0.0), 0.0),
            "min_clean": _num(params.get("min_clean", 0.0), 0.0),
            "min_comfort": _num(params.get("min_comfort", 0.0), 0.0),
            "min_facilities": _num(params.get("min_facilities", 0.0), 0.0),
            "sort_by": sb,
            "limit": int(min(max(int(params.get("limit", 5)), 1), 10)),
        }
    except Exception:
        return _regex_parse_fallback(query)
