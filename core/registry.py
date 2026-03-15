import importlib

def load_market(code: str):
    return importlib.import_module(f"markets.{code}")