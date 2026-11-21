"""Parse every CSV in data/ once and write the fast-loading cache to data/cache/.

uv run python scripts/build_cache.py [--refresh]
"""

import sys

from electric_fish import find_recordings, load_recording

refresh = "--refresh" in sys.argv
for name in find_recordings():
    rec = load_recording(name, refresh=refresh)
    print(rec, f"pose conflicts: {rec.n_pose_conflicts}")
