import json
from pathlib import Path
from unittest.mock import patch
import experiments

def test_running_from_another_directory_resolves_project_files(tmp_path, monkeypatch):
    script = tmp_path / 'project' / 'experiments.py'
    script.parent.mkdir()
    monkeypatch.setattr(experiments, '__file__', str(script))
    monkeypatch.chdir(tmp_path)
    calls = []
    def train(command, **kwargs):
        assert Path(command[1]).is_absolute()
        assert Path(command[1]).parent == script.parent
        assert Path(kwargs['cwd']) == script.parent
        out = Path(command[command.index('--out') + 1])
        assert out.is_absolute() and out.parent == script.parent / 'runs'
        out.mkdir(parents=True)
        data = {'config': {'lr':.001, 'margin':.3, 'dim':128, 'mining':'hard', 'optimizer':'adam'},
                'batch_size':64, 'val_triplet_accuracy':.8, 'test': {'mAP':.1}}
        (out / 'results.json').write_text(json.dumps(data))
        calls.append(command)
    with patch('subprocess.run', side_effect=train):
        experiments.main()
    assert len(calls) == 6
    assert (script.parent / 'runs' / 'results.csv').exists()
