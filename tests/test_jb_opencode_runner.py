# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A rollout stopped by the time limit after writing must not read as finished."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2.benchmarks import jb_opencode_runner as jb


def _settings(root: Path) -> jb.Settings:
    tasks_base = root / "tasks"
    tasks_base.mkdir(parents=True, exist_ok=True)
    return jb.Settings(
        task_ids=("t1",),
        label="lbl",
        render_dir=root / "render",
        tasks_base=tasks_base,
        opencode_dir=root / "oc",
        model_id="m",
        agent_name="",
        variant="",
        agent_steps="",
        timeout_per_task="2",
        max_retries=2,
        log_dir=root / "logs",
    )


class JbRunnerExitStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = _settings(self.root)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        task_dir = self.cfg.tasks_base / "t1" / "task_folder"
        task_dir.mkdir(parents=True)
        (task_dir / "TASK_INSTRUCTIONS.txt").write_text("hi", encoding="utf-8")

    def _run(self, codes: list[int]) -> tuple[bool, int]:
        calls = []

        def fake_attempt(cfg, workspace, traj_file, log_file, task_name, attempt):
            calls.append(attempt)
            (workspace / "output").mkdir(parents=True, exist_ok=True)
            (workspace / "output" / "draft.txt").write_text("partial", encoding="utf-8")
            return codes[min(attempt - 1, len(codes) - 1)]

        with (
            patch.object(jb, "TEMP_BASE", self.root / "ws"),
            patch.object(jb, "_stage", lambda *a, **k: None),
            patch.object(jb, "_attempt", side_effect=fake_attempt),
        ):
            result = jb.run_one(self.cfg, "t1")
        return result, len(calls)

    def test_killed_rollout_is_not_finished(self):
        """Exit 124 (timeout) after writing must not mark the rollout delivered."""
        delivered, attempts = self._run([124, 124])
        self.assertFalse(delivered)
        self.assertEqual(attempts, 2)  # retried, never accepted as clean

    def test_sigkill_rollout_is_not_finished(self):
        """A negative returncode (SIGKILL) after writing must not be accepted."""
        delivered, attempts = self._run([-9, -9])
        self.assertFalse(delivered)
        self.assertEqual(attempts, 2)

    def test_clean_finish_is_delivered(self):
        delivered, attempts = self._run([0])
        self.assertTrue(delivered)
        self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()
