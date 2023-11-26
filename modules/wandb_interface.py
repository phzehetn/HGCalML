from typing import Any
import wandb

class WandBWrapper:
    def __init__(self):
        self._wandb = None
        self.active = True #for future use

    def init(self, *args, **kwargs):
        if not self.active:
            return
        if self._wandb is not None:
            raise RuntimeError("wandb is already initialised")
        self._wandb = wandb.init(*args, **kwargs)

    def _dummy(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if not self.active:
            return self._dummy
        if self._wandb is not None:
            return getattr(self._wandb, name)
                
        raise AttributeError(f"WandB instance not initialized and set to active at the same time. Attribute '{name}' is not available.")


wandb_wrapper = WandBWrapper()