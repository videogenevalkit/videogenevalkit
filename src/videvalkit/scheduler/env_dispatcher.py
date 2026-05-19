"""CondaEnvDispatcher — run a benchmark adapter in its own conda env.

Communication: subprocess with JSON over stdin/stdout. The child process
boots `python -m videvalkit.benchmarks.entry`, which loads the named
adapter and dispatches the requested method.

This is the toolkit's answer to upstream dependency conflicts: each
benchmark lives in `videvalkit-<bench>` and we never have to make their
requirements coexist.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from typing import Any


class CondaEnvDispatcherError(RuntimeError):
    pass


class CondaEnvDispatcher:
    def __init__(self, conda_root: str | None = None) -> None:
        self.conda_exe = self._find_conda(conda_root)

    @staticmethod
    def _find_conda(conda_root: str | None) -> str:
        if conda_root:
            cand = os.path.join(conda_root, "bin", "conda")
            if os.path.exists(cand):
                return cand
        which = shutil.which("conda")
        if which:
            return which
        raise CondaEnvDispatcherError("conda not found on PATH and no conda_root supplied")

    def run(
        self,
        env_name: str,
        benchmark: str,
        method: str,
        payload: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Invoke entry.py inside `env_name`. Returns parsed JSON result.

        Convention: child writes a single JSON object to stdout on success;
        on error it exits non-zero with the traceback on stderr.
        """
        cmd = [
            self.conda_exe, "run", "-n", env_name, "--no-capture-output",
            "python", "-m", "videvalkit.benchmarks.entry",
            "--benchmark", benchmark,
            "--method", method,
        ]
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise CondaEnvDispatcherError(
                f"{benchmark}.{method} in env={env_name} failed (exit={proc.returncode})\n"
                f"cmd: {shlex.join(cmd)}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise CondaEnvDispatcherError(
                f"adapter returned non-JSON stdout (first 2KB):\n{proc.stdout[:2000]}"
            ) from e
