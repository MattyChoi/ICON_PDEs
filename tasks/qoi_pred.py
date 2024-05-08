import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import hydra
from omegaconf import DictConfig
import lightning as L
import random
import torch
import torchmetrics

from utils.visualizations import plot_figure


class QOIPredModule(L.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(hparams.model)
        self.loss = hydra.utils.instantiate(hparams.loss)
        
        # save the validation conditions, qois, and labels
        self.val_outs = []
        self.test_outs = []


    def training_step(
        self, batch, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        prompt, labels = batch
        
        # run it through the model to get the logits and loss
        qois = self.model(prompt)

        # calculate the loss
        loss = self.loss(qois[:, ::2, :], labels)

        # log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        prompt, labels = batch
        
        # run it through the model to get the logits and loss
        qois = self.model(prompt)

        # calculate the loss on only the query vector
        loss = self.loss(qois[:, -1], labels[:, -1])

        # log the loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.val_outs.append((qois[:, ::2, :], labels))


    def on_validation_epoch_end(self) -> None:
        batch_num = random.randint(0, len(self.val_outs)-1)
        
        # get the test conditions, qois, and labels
        qois, labels = self.test_outs[batch_num]
        dim = qois.size(-1)
        conditions = torch.linspace(0, 1, dim)
        
        batch_inds = random.sample(range(len(qois)), 9)
        
        fig = plot_figure(
            conditions=conditions, 
            qois=qois[:, -1], 
            labels=labels[:, -1],
            inds=batch_inds,
            show=False
        )
        self.logger.experiment.add_figure(f"Validation Epoch {self.current_epoch}", fig)
        self.val_outs.clear()


    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get all the data
        prompt, labels = batch
        
        # run it through the model to get the logits and loss
        qois = self.model(prompt)

        # calculate the loss on only the query vector
        loss = self.loss(qois[:, -1], labels[:, -1])

        # log the loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.test_outs.append((qois[:, ::2, :], labels))


    def on_test_epoch_end(self) -> None:
        batch_num = random.randint(0, len(self.test_outs)-1)
        
        # get the test conditions, qois, and labels
        qois, labels = self.test_outs[batch_num]
        dim = qois.size(-1)
        conditions = torch.linspace(0, 1, dim)
        
        batch_inds = random.sample(range(len(qois)), 9)
        
        fig = plot_figure(
            conditions=conditions, 
            qois=qois[:, -1], 
            labels=labels[:, -1],
            inds=batch_inds,
            show=False
        )
        self.logger.experiment.add_figure(f"Test Epoch {self.current_epoch}", fig)
            
        self.test_outs.clear()


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
