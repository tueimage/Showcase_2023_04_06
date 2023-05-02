#**************************************************************************************************#
#                                        Pytorch Lightning                                         #
#**************************************************************************************************#
#                                                                                                  #
# "A framework to train and deploy Pytorch models".                                                #
#                                                                                                  #
#  Benefits:                                                                                       #
#    - More compact code (80% less)                                                                #
#    - Structured approach                                                                         #
#    - Flexible with standard Pytorch                                                              #
#                                                                                                  #
#  Three main Modules:                                                                             #
#    - Datamodule                                                                                  #
#    - Lightning Module                                                                            #
#    - Trainer                                                                                     #
#                                                                                                  #
#**************************************************************************************************#



#**************************************************************************************************#
#                                             Imports                                              #
#**************************************************************************************************#
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#**************************************************************************************************#
#                                          Class Datamodule                                        #
#**************************************************************************************************#
#                                                                                                  #
# The data module for training, validation and testing data.                                       #
#                                                                                                  #
#**************************************************************************************************#
class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):

        if stage == "fit"/"test"/"predict":
            ...

        self.train_dataset = ...
        self.valid_dataset = ...
        self.test_dataset = ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.config.batch_size,
                          shuffle=self.config.shuffle,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)



#**************************************************************************************************#
#                                          Class Framework                                         #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test data-driven models.                            #
#                                                                                                  #
#**************************************************************************************************#
class Framework(pl.LightningModule):

    # Initialize framework that take model as input
    def __init__(self, model, loss, opt=torch.optim.Adam, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss = loss
        self.opt = opt
        self.lr = lr

        self.identifier = ''

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, ppm = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, ppm[0])
        self.log("train_loss", loss)
        return loss # This is passed to the optimizer for training
    
    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = self.opt(params, lr=self.lr)
        return optimizer
    
    # Multiple optimizers/schedulers
    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(...)
        optimizer2 = torch.optim.SGD(...)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, ...)
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, ...)
        return (
            {"optimizer": optimizer1,
                "lr_scheduler": {"scheduler": scheduler1, "monitor": "metric_to_track"},
            },
            {"optimizer": optimizer2, "lr_scheduler": scheduler2},
        )


    def validation_step(self, batch, batch_idx):
        x, y, ppm = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, ppm[0])
        self.log("val_loss", loss, prog_bar=True)

    

    # Implement other steps
    def test_step(self, batch, batch_idx):
        return ...  
    def predict_step(self, batch, batch_idx):
        return ...
    
    # Implement own code at specific steps if needed
    def on_fit_end(self):
        return ...
    def on_before_optimizer_step(self):
        return ...



#**************************************************************************************************#
#                                           Callbacks                                              #
#**************************************************************************************************#
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

checkpoint_callback = ModelCheckpoint(monitor='val_loss')
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=10)

# Many more callbacks available and ability to create your own! 


#**************************************************************************************************#
#                                     Define model + loss                                          #
#**************************************************************************************************#
net = ...
loss = ...


#**************************************************************************************************#
#                                     Define Framework                                             #
#**************************************************************************************************#
lightning_model = Framework(net, loss, opt=torch.optim.Adam, lr=1e-3)


#**************************************************************************************************#
#                                           Trainer                                                #
#**************************************************************************************************#

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",  # set to "auto" or "gpu" to use GPUs if available
    devices="auto",  # Uses all available GPUs if applicable
    logger = ...,
    log_every_n_steps=...,
    callbacks= [checkpoint_callback, early_stop_callback],
    check_val_every_n_epoch=...,
)


#**************************************************************************************************#
#                                           Fitting                                                #
#**************************************************************************************************#

# With Datamodule
trainer.fit(lightning_model, datamodule=DataModule(config=...))

# Or with individual dataloaders
trainer.fit(lightning_model, train_dataloaders=..., val_dataloaders=...)