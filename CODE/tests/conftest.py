import sys
from pathlib import Path

# Make `retrieval` and `llm` importable as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
