from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for module_name in tuple(sys.modules):
    if module_name == "rvc_realtime" or module_name.startswith("rvc_realtime."):
        del sys.modules[module_name]

from rvc_realtime import dml_train_hook as hook


class ApplyOfficialTrainHookTests(unittest.TestCase):
    def _official_script(self, *, already_hooked: bool = False) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory()
        script = Path(temp.name) / "train.py"
        parts = [
            hook.HOOK_BLOCK if already_hooked else hook.IMPORT_OLD,
            hook.NETS_NEW if already_hooked else hook.NETS_OLD,
            hook.CACHE_NEW if already_hooked else hook.CACHE_OLD,
            hook.STEP_NEW if already_hooked else hook.STEP_OLD,
            hook.PIN_NEW if already_hooked else hook.PIN_OLD,
            hook.WORKERS_OLD,
            hook.MULTIPROCESS_OLD if already_hooked else hook.STOCK_MULTIPROCESS_OLD,
        ]
        script.write_text("\n".join(parts), encoding="utf-8")
        return temp, script

    def test_patches_cpu_and_directml_paths_and_is_idempotent(self) -> None:
        temp, script = self._official_script()
        self.addCleanup(temp.cleanup)
        with patch.object(hook, "official_train_script", return_value=script):
            first = hook.apply_official_train_hook()
            patched = script.read_text(encoding="utf-8")
            second = hook.apply_official_train_hook()

        self.assertIn("DirectML/CPU 单进程", first)
        self.assertIn(hook.MARKER, patched)
        self.assertIn(hook.MULTIPROCESS_NEW, patched)
        self.assertIn(hook.WORKERS_NEW, patched)
        self.assertEqual(patched, script.read_text(encoding="utf-8"))
        self.assertIn("DirectML/CPU 单进程", second)

    def test_upgrades_existing_directml_hook(self) -> None:
        temp, script = self._official_script(already_hooked=True)
        self.addCleanup(temp.cleanup)
        with patch.object(hook, "official_train_script", return_value=script):
            hook.apply_official_train_hook()
        patched = script.read_text(encoding="utf-8")
        self.assertIn(hook.MULTIPROCESS_NEW, patched)
        self.assertIn(hook.WORKERS_NEW, patched)
        self.assertNotIn(hook.MULTIPROCESS_OLD, patched)


if __name__ == "__main__":
    unittest.main()
