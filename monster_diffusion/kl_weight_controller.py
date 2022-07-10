from typing import List
from pydantic import BaseModel
import numpy as np
from simple_pid import PID


class KLWeightController(BaseModel):
    weights: np.ndarray
    targets: np.ndarray
    pids: List[PID]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, weights, targets):
        super().__init__(
            weights=np.array(weights, dtype=np.float32),
            targets=np.array(targets, dtype=np.float32),
            pids=KLWeightController.new_pids(weights, targets),
        )

    @staticmethod
    def new_pids(weights, targets):
        pids = [
            PID(
                -0.0,
                -0.001,
                -0.0,
                setpoint=np.log10(target),
                auto_mode=False,
            )
            for target in targets
        ]
        for pid, weight in zip(pids, weights):
            pid.set_auto_mode(True, last_output=np.log10(weight))
        return pids

    def update_(self, kl_losses):
        if len(self.pids) != len(kl_losses):
            raise ValueError("Expected same number of kl as pid controllers")

        for index, (pid, kl) in enumerate(zip(self.pids, kl_losses)):
            self.weights[index] = 10 ** pid(np.log10(kl.item()), dt=1)
        return self.weights

    def state_dict(self):
        return dict(weights=self.weights)

    def load_state_dict(self, state_dict):
        self.weights = state_dict["weights"]
        self.pids = KLWeightController.new_pids(self.weights, self.targets)

    def map_(self, fn):
        self.weights = fn(self.weights)
        self.pids = KLWeightController.new_pids(self.weights, self.targets)

    def targets_(self, targets):
        for pid, target in zip(self.pids, targets):
            pid.setpoint = np.log10(target)
