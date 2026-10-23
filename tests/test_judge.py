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

"""judge_one must treat a null all_pass / n_passed as absent, not as a bad score."""

import unittest
from pathlib import Path

from harness2.core.judge import judge_one


class _PayloadAdapter:
    def __init__(self, payload):
        self.payload = payload

    def score(self, run_dir, task_id):
        return self.payload


def _judge(payload):
    return judge_one(_PayloadAdapter(payload), "ledger-1", "agg", 1, "agg1", "r1", Path("/tmp/x"))


class JudgeOneNullFieldsTests(unittest.TestCase):
    def test_all_pass_none_falls_back_to_derived(self):
        # 4 of 4 criteria passed, but the LAB adapter forwards all_pass=None.
        row = _judge({"n_passed": 4, "n_criteria": 4, "all_pass": None})
        self.assertTrue(row.graded)
        self.assertTrue(row.all_pass, "null all_pass must fall back to n_passed == n_criteria")
        self.assertEqual(row.frac, 1.0)

    def test_explicit_false_is_preserved(self):
        row = _judge({"n_passed": 2, "n_criteria": 4, "all_pass": False})
        self.assertFalse(row.all_pass, "an explicit all_pass=False must stay False")
        self.assertEqual(row.frac, 0.5)

    def test_n_passed_none_does_not_crash_frac(self):
        # A partial payload: n_criteria present but n_passed missing -> forwarded None.
        row = _judge({"n_passed": None, "n_criteria": 4, "all_pass": None})
        self.assertTrue(row.graded)
        self.assertIsNone(row.frac, "a missing n_passed must yield a None fraction, not raise")


if __name__ == "__main__":
    unittest.main()
