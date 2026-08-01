# Make the repo root importable so tests can `from courtvision import ...`
# regardless of where pytest is invoked from.

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__)),
)
