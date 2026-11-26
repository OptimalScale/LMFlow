"""Static regression test for undefined names in the RAFT trainer.

``raft_trainer.py`` is adapted from ``transformers.Trainer``. Its ``_one_train`` method
once referenced a ``model`` variable that only exists in the upstream implementation,
which raised ``NameError`` as soon as gradient clipping was used. The file is parsed
with ``symtable`` so the test doesn't need ``transformers``/GPU.
"""

import builtins
import symtable
from pathlib import Path

RAFT_TRAINER = Path(__file__).resolve().parents[2] / "src/lmflow/pipeline/utils/raft_trainer.py"


def _find(table, name):
    for child in table.get_children():
        if child.get_name() == name:
            return child
        found = _find(child, name)
        if found is not None:
            return found
    return None


def test_one_train_has_no_undefined_global_names():
    source = RAFT_TRAINER.read_text()
    module_table = symtable.symtable(source, str(RAFT_TRAINER), "exec")
    module_names = {s.get_name() for s in module_table.get_symbols() if s.is_assigned() or s.is_imported()}

    one_train = _find(module_table, "_one_train")
    assert one_train is not None

    undefined = [
        s.get_name()
        for s in one_train.get_symbols()
        if s.is_global()
        and s.is_referenced()
        and s.get_name() not in module_names
        and not hasattr(builtins, s.get_name())
    ]
    assert undefined == []
