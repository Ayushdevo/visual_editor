import subprocess
import sys
import pytest
@pytest.mark.parametrize('flags', [['--dim','0'],['--p','1'],['--margin','nan'],['--lr','nan'],['--seed','-1']])
def test_invalid_training_options_have_no_side_effects(tmp_path,flags):
    from pathlib import Path
    run=subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--device','cpu','--out',str(tmp_path/'out'),*flags],capture_output=True,text=True)
    assert run.returncode==2, run.stderr
    assert not (tmp_path/'out').exists()
