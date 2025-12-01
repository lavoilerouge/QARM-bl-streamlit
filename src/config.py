"""
QARM Source Configuration
=========================
Re-exports from main config for convenience.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports BEFORE importing
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import the parent config module explicitly
import importlib.util
_config_path = Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("parent_config", _config_path)
_parent_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent_config)

# Re-export all public names from parent config
from types import ModuleType as _ModuleType
for _name in dir(_parent_config):
    if not _name.startswith('_'):
        _obj = getattr(_parent_config, _name)
        if not isinstance(_obj, _ModuleType):
            globals()[_name] = _obj
