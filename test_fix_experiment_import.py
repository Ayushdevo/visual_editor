import importlib.util
from pathlib import Path
from unittest.mock import patch

def test_import_does_not_launch_experiments():
    path = Path(__file__).with_name('experiments.py')
    spec = importlib.util.spec_from_file_location('experiment_module', path)
    module = importlib.util.module_from_spec(spec)
    with patch('subprocess.run', side_effect=AssertionError('import launched training')):
        spec.loader.exec_module(module)
