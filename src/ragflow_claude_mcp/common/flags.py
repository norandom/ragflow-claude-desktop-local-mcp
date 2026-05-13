"""Module-level flags shared between server and client.

DSPY_AVAILABLE lives here (rather than in server.py) so client.ragflow can read
it at call time without creating an import cycle. Tests can patch this attribute
to simulate a missing DSPy install.
"""

from importlib.util import find_spec

# Probe for DSPy without actually importing it (avoids loading the heavy module
# at startup just to test availability).
DSPY_AVAILABLE = find_spec("dspy") is not None
