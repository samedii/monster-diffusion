import subprocess
from pathlib import Path
from functools import wraps
from pydantic import BaseModel
from typing import Optional


class GuildRun(BaseModel):
    """
    Usage:
        guild_run("464170e3").path

        with guild_run("464170e3") as model_path:
            ...

        with guild_run(4, "model") as model_path:
            ...

        @guild_run(1, "model")
        def test_model(...):
            ...
    """

    path: Path
    symlink_path: Optional[Path]

    def __init__(self, selection: str, symlink_path: Path = None):
        super().__init__(
            path=GuildRun.run_path(selection),
            symlink_path=symlink_path,
        )

    def __enter__(self):
        if self.symlink_path is None:
            return self.path
        else:
            if self.symlink_path.exists():
                raise FileExistsError(f"{self.symlink_path} already exists")
            self.symlink_path.symlink_to(self.path)
            return self.symlink_path

    def __exit__(self, type, value, traceback):
        if self.symlink_path is not None:
            self.symlink_path.unlink()

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with GuildRun(self.selection, self.symlink_path):
                return fn(*args, **kwargs)

        return wrapper

    @staticmethod
    def run_path(selection):
        try:
            return Path(subprocess.check_output(
                f"guild select {selection} --path", shell=True
                # f"guild open {selection} -c echo", shell=True
            ).decode("utf-8").strip())
        except subprocess.CalledProcessError:
            raise ValueError(
                f"Failed to get guild run path, the selection {selection} "
                "probably didn't match any run"
            )


guild_run = GuildRun
