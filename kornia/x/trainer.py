# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from typing import Any, Callable, Dict, Optional

# the accelerator library is a requirement for the Trainer
# but it is optional for grousnd base user of kornia.
import torch
from accelerate import Accelerator
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from kornia.core import Module, Tensor
from kornia.x.utils import Configuration, TrainerState

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

from kornia.metrics import AverageMeter

from .utils import Configuration, StatsTracker, TrainerState

callbacks_whitelist = [
    "preprocess",
    "augmentations",
    "evaluate",
    "fit",
    "fit_epoch",
    "on_epoch_start",
    "on_before_model",
    "on_after_model",
    "on_checkpoint",
    "on_epoch_end",
]


class Trainer:
    """Base class to train the different models in kornia.

    .. warning::
        The API is experimental and subject to be modified based on the needs of kornia models.

    Args:
        model: the nn.Module to be optimized.
        train_dataloader: the data loader used in the training loop.
        valid_dataloader: the data loader used in the validation loop.
        criterion: the nn.Module with the function that computes the loss.
        optimizer: the torch optimizer object to be used during the optimization.
        scheduler: the torch scheduler object with defiing the scheduling strategy.
        accelerator: the Accelerator object to distribute the training.
        config: a TrainerConfiguration structure containing the experiment hyper parameters.
        callbacks: a dictionary containing the pointers to the functions to overrides. The
          main supported hooks are ``evaluate``, ``preprocess``, ``augmentations`` and ``fit``.

    .. important::
        The API heavily relies on `accelerate <https://github.com/huggingface/accelerate/>`_.
        In order to use it, you must: ``pip install kornia[x]``

    .. seealso::
        Learn how to use the API in our documentation
        `here <https://kornia.readthedocs.io/en/latest/get-started/training.html>`_.

    """

    def __init__(
        self,
        model: Module,
        train_dataloader: DataLoader[Any],
        valid_dataloader: DataLoader[Any],
        criterion: Optional[Module],
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        config: Configuration,
        callbacks: Optional[Dict[str, Callable[..., None]]] = None,
    ) -> None:
        # Fast path: define frequently used vars locally
        Accelerator_ = Accelerator
        if Accelerator_ is None:
            raise ModuleNotFoundError('accelerate library is not installed: pip install "kornia[x]"')

        accelerator = Accelerator_()
        self.accelerator = accelerator

        # Prepare all objects in fewer lines
        model = accelerator.prepare(model)
        train_dataloader = accelerator.prepare(train_dataloader)
        valid_dataloader = accelerator.prepare(valid_dataloader)
        optimizer = accelerator.prepare(optimizer)
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Avoid attribute lookup in conditional
        device = getattr(accelerator, "device", None)
        if criterion is not None and device is not None:
            criterion = criterion.to(device)
        self.criterion = criterion

        if callbacks is None:
            callbacks = {}
        setattr_ = setattr
        TrainerType = type(self)
        for fn_name, fn in callbacks.items():
            if fn_name not in _callbacks_whitelist_set:
                raise ValueError(f"Not supported: {fn_name}.")
            setattr_(TrainerType, fn_name, fn)

        self.num_epochs = config.num_epochs
        self.state = TrainerState.STARTING

        # Use optimized logger get for speed-up (local var binding)
        self._logger = logging.getLogger("train")

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    def backward(self, loss: Tensor) -> None:
        self.accelerator.backward(loss)

    def fit_epoch(self, epoch: int) -> None:
        # train loop
        self.model.train()
        losses = AverageMeter()
        for sample_id, sample in enumerate(self.train_dataloader):
            sample = {"input": sample[0], "target": sample[1]}  # new dataset api will come like this
            self.optimizer.zero_grad()
            # perform the preprocess and augmentations in batch
            sample = self.preprocess(sample)
            sample = self.augmentations(sample)
            sample = self.on_before_model(sample)
            # make the actual inference
            output = self.on_model(self.model, sample)
            self.on_after_model(output, sample)  # for debugging purposes
            loss = self.compute_loss(output, sample["target"])
            self.backward(loss)
            self.optimizer.step()

            losses.update(loss.item(), len(sample["input"]))

            if sample_id % 50 == 0:
                self._logger.info(
                    f"Train: {epoch + 1}/{self.num_epochs}  "
                    f"Sample: {sample_id + 1}/{len(self.train_dataloader)} "
                    f"Loss: {losses.val:.3f} {losses.avg:.3f}"
                )

    def fit(self) -> None:
        # execute the main loop
        # NOTE: Do not change and keep this structure clear for readability.
        for epoch in range(self.num_epochs):
            # call internally the training loop
            # NOTE: override to customize your evaluation routine
            self.state = TrainerState.TRAINING
            self.fit_epoch(epoch)

            # call internally the evaluation loop
            # NOTE: override to customize your evaluation routine
            self.state = TrainerState.VALIDATE
            valid_stats = self.evaluate()

            self.on_checkpoint(self.model, epoch, valid_stats)

            self.on_epoch_end()
            if self.state == TrainerState.TERMINATE:
                break

            # END OF THE EPOCH
            self.scheduler.step()

        ...

    # events stubs

    @torch.no_grad()
    def evaluate(self) -> Dict[str, AverageMeter]:
        self.model.eval()
        stats = StatsTracker()
        for sample_id, sample in enumerate(self.valid_dataloader):
            sample = {"input": sample[0], "target": sample[1]}  # new dataset api will come like this
            # perform the preprocess and augmentations in batch
            sample = self.preprocess(sample)
            sample = self.on_before_model(sample)
            # Forward
            out = self.on_model(self.model, sample)
            self.on_after_model(out, sample)

            batch_size: int = len(sample["input"])
            # measure accuracy and record loss
            # Loss computation
            if self.criterion is not None:
                val_loss = self.compute_loss(out, sample["target"])
                stats.update("losses", val_loss.item(), batch_size)
            stats.update_from_dict(self.compute_metrics(out, sample["target"]), batch_size)

            if sample_id % 10 == 0:
                self._logger.info(f"Test: {sample_id}/{len(self.valid_dataloader)} {stats}")

        return stats.as_dict()

    def on_epoch_start(self, *args: Any, **kwargs: Any) -> None: ...

    def preprocess(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def augmentations(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def compute_metrics(self, *args: Any) -> Dict[str, float]:
        """Compute metrics during the evaluation."""
        return {}

    def compute_loss(self, *args: Tensor) -> Tensor:
        if self.criterion is None:
            raise RuntimeError("`criterion` should not be None.")
        return self.criterion(*args)

    def on_before_model(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def on_model(self, model: Module, sample: Dict[str, Tensor]) -> Tensor:
        return model(sample["input"])

    def on_after_model(self, output: Tensor, sample: Dict[str, Tensor]) -> None: ...

    def on_checkpoint(self, *args: Any, **kwargs: Dict[str, Any]) -> None: ...

    def on_epoch_end(self, *args: Any, **kwargs: Dict[str, Any]) -> None: ...


_callbacks_whitelist_set = set(callbacks_whitelist)
