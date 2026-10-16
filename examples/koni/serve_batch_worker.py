"""Compatibility entrypoint; B4 baseline shares the validated worker."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("serve_length_worker.py")),run_name="__main__")
