import sys
from pathlib import Path

# Make `import model`, `import dataset`, etc. work from the tests without
# packaging the project. Keeping src/ a plain directory (no pip install -e)
# is deliberate simplicity for a learning project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
