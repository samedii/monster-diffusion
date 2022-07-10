"""
Add a debug config for a python module in vscode like:

{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Module",
            "type": "python",
            "request": "launch",
            "module": "operations.debug.train",
            "justMyCode": true
        }
    ]
}

Also set a shortcut like "Shift+Enter" for "Evaluate in debug console" to interactively
run code you are trying to fix while debugging.
"""
from guild.commands.run import run


if __name__ == "__main__":
    run(
        [
            "train",
            "-y",
            "n_workers=0",
            "n_batches_per_epoch=2",
            "batch_size=4",
            "eval_batch_size=4",
            "--debug-sourcecode=.",
            # "--label debug",
        ]
    )
