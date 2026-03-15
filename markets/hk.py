from __future__ import annotations
import re
from typing import List
from core.hkex_list import fetch_hkex_list

def get_universe(config: dict) -> List[dict]:
    rows = fetch_hkex_list()
    return [{"id": code5, "name": name} for code5, name in rows]

def to_ticker(row: dict) -> str:
    digits = re.sub(r"\D", "", row["id"])
    return f"{digits[-4:].zfill(4)}.HK"