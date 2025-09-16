import re
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

_hotels_df: Optional[pd.DataFrame] = None

REQUIRED_COLS = [
    "hotel_id",
    "hotel_name",
    "city",
    "country",
    "star_rating",
    "lat",
    "lon",
    "cleanliness_base",
    "comfort_base",
    "facilities_base",
]

def _load_csv() -> pd.DataFrame:
    p = Path("hotels.csv")
    if not p.exists():
        alt = Path("/mnt/data/hotels.csv")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError("hotels.csv not found next to app.py/tools.py.")
    return pd.read_csv(p)

def load_hotels() -> pd.DataFrame:
    global _hotels_df
    if _hotels_df is not None:
        return _hotels_df
    df = _load_csv()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing) + "\nAvailable: " + ", ".join(df.columns))
    for col in ["city", "country", "hotel_name"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    for col in ["star_rating", "cleanliness_base", "comfort_base", "facilities_base"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["hotel_name", "city", "country"]).copy()
    _hotels_df = df
    return _hotels_df

def _resolve_sort_column(sort_by: str) -> str:
    s = (sort_by or "star_rating").strip().lower()
    return {
        "star_rating": "star_rating",
        "stars": "star_rating",
        "star": "star_rating",
        "rating": "star_rating",
        "cleanliness": "cleanliness_base",
        "comfort": "comfort_base",
        "facilities": "facilities_base",
    }.get(s, "star_rating")

def _wb_mask(series: pd.Series, needle: Optional[str]) -> pd.Series:
    if not needle:
        return pd.Series([True] * len(series), index=series.index)
    pattern = rf"(?i)\b{re.escape(needle.strip())}\b"
    return series.str.contains(pattern, na=False, regex=True)

def query_hotels(
    city: Optional[str] = None,
    country: Optional[str] = None,
    min_star: float = 0.0,
    min_clean: float = 0.0,
    min_comfort: float = 0.0,
    min_facilities: float = 0.0,
    sort_by: str = "star_rating",
    limit: int = 5,
) -> pd.DataFrame:
    df = load_hotels()
    q = df.copy()
    if city:
        q = q[_wb_mask(q["city"], city)]
    if country:
        q = q[_wb_mask(q["country"], country)]
    if float(min_star) > 0:
        q = q[q["star_rating"].fillna(0) >= float(min_star)]
    if float(min_clean) > 0:
        q = q[q["cleanliness_base"].fillna(0) >= float(min_clean)]
    if float(min_comfort) > 0:
        q = q[q["comfort_base"].fillna(0) >= float(min_comfort)]
    if float(min_facilities) > 0:
        q = q[q["facilities_base"].fillna(0) >= float(min_facilities)]
    sort_col = _resolve_sort_column(sort_by)
    if sort_col not in q.columns:
        sort_col = "star_rating"
    q = q.sort_values(by=sort_col, ascending=False)
    try:
        n = int(limit)
    except Exception:
        n = 5
    n = max(1, min(n, 10))
    out = q.head(n).copy()
    out["city"] = out["city"].str.title()
    out["country"] = out["country"].str.title()
    return out[[
        "hotel_name", "city", "country",
        "star_rating", "cleanliness_base", "comfort_base", "facilities_base"
    ]].reset_index(drop=True)

def debug_filter_counts(params: Dict[str, Any]) -> str:
    df = load_hotels()
    lines = [f"- total rows in CSV: **{len(df)}**"]
    q = df.copy()
    if params.get("city"):
        m = _wb_mask(q["city"], params["city"])
        lines.append(f"- after city == `{params['city']}`: **{int(m.sum())}**")
        q = q[m]
    else:
        lines.append("- city filter: *(none)*")
    if params.get("country"):
        m = _wb_mask(q["country"], params["country"])
        lines.append(f"- after country == `{params['country']}`: **{int(m.sum())}**")
        q = q[m]
    else:
        lines.append("- country filter: *(none)*")
    def step(label, col, val):
        nonlocal q
        if float(val) > 0:
            before = len(q)
            q = q[q[col].fillna(0) >= float(val)]
            lines.append(f"- after {label} ≥ {val:g}: **{len(q)}** (was {before})")
        else:
            lines.append(f"- {label} threshold: *(none)*")
    step("star rating", "star_rating", params.get("min_star", 0) or 0)
    step("cleanliness", "cleanliness_base", params.get("min_clean", 0) or 0)
    step("comfort", "comfort_base", params.get("min_comfort", 0) or 0)
    step("facilities", "facilities_base", params.get("min_facilities", 0) or 0)
    return "\n".join(lines)
