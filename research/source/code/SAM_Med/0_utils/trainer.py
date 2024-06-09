# Imports
import torch
import monai
from tqdm import tqdm
from torch import nn

class MedSAMTrainer(object):
    def __init__(self, loggingHandler, dataloaderHandler, checkpointHandler, device, *args, **kwargs):
        self.loggingHandler = loggingHandler
        self.dataloaderHandler = dataloaderHandler
        self.checkpointHandler = checkpointHandler
        self.device = device

        self.epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 100
        self.batches_per_epoch = -1 if 'batches_per_epoch' not in kwargs.keys() or kwargs['batches_per_epoch'] is None else kwargs['batches_per_epoch']
        self.resume = kwargs['resume'] if 'resume' in kwargs.keys() else False

        self.use_boxes = kwargs['use_boxes'] if 'use_boxes' in kwargs.keys() and kwargs['use_boxes'] else False
        self.use_positive_points = kwargs['use_positive_points'] if 'use_positive_points' in kwargs.keys() and kwargs['use_positive_points'] else False

        assert self.use_boxes or self.use_positive_points, 'Please set either use_boxes or use_positive_points to True'

        self.dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.epochs):
            self.on_epoch_start(epoch)

            pbar = tqdm(enumerate(self.dataloaderHandler.train_loader), total=self.batches_per_epoch)
            for batch_id, batch in pbar:
                self.loggingHandler.log('Getting batch {}'.format(batch_id))
                # batch = next(iter(self.dataloaderHandler.train_loader))
                dice_loss, ce_loss = self.train_step(batch_id, batch)            
                
                pbar.set_description(f"Epoch {epoch}, loss: {dice_loss + ce_loss:.4f}")

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_len = min(len(self.dataloaderHandler.val_loader), self.batches_per_epoch) # in debugging batches might be small, so we go with this.
                pbar = tqdm(enumerate(self.dataloaderHandler.val_loader), total=val_len)
                for batch_id, batch in pbar:
                    
                    dice_loss, ce_loss = self.validation_step(batch_id, batch)
                    
                    pbar.set_description(f"Validating epoch {epoch}, loss: {dice_loss + ce_loss:.4f}")

                self.on_validation_epoch_end()

            self.on_epoch_end(epoch)

    def on_train_start(self):
        self.loggingHandler.setup_logger()

        torch.cuda.empty_cache()

        self.loggingHandler.log('Setting up dataloaders')
        self.dataloaderHandler.try_setup_data_split_from_save_with_fallback()
        self.dataloaderHandler.setup_dataloaders()

        self.batches_per_epoch = len(self.dataloaderHandler.train_loader) if self.batches_per_epoch == -1 else self.batches_per_epoch

        self.loggingHandler.log('Setting up models')
        self.model, self.optimizer, self.current_epoch, self.best_loss = None, None, 0, 1e10
        if self.checkpointHandler.final_checkpoint_exists() and self.resume:
            self.loggingHandler.log('We have already trained this model to completion')
            exit(1)
        elif self.checkpointHandler.checkpoint_exists() and self.resume:
            self.loggingHandler.log('Resume is true and a checkpoint exists, we resume')
            self.model, self.optimizer, self.current_epoch, self.best_loss = self.checkpointHandler.load_checkpoint()
            self.current_epoch += 1
            self.loggingHandler.log(f'Resuming at epoch: {self.current_epoch}')
        else:
            self.loggingHandler.log('Setting up a fresh start model')
            self.model, self.optimizer = self.checkpointHandler.load_base_checkpoint()

    def on_epoch_start(self, epoch):
        self.loggingHandler.log('=====================================')
        self.loggingHandler.log('Setting up a new epoch for the logger')
        self.loggingHandler.start_new_epoch(epoch)
        self.current_epoch = epoch

    def forward_pass(self, batch):
        # Get data
        image = batch["image"].to(self.device)
        gt2D = batch["gt2D"].to(self.device)

        point_prompt = None
        if self.use_positive_points:
            coords_torch = batch["coords"].squeeze().to(self.device) # ([B, Ps, 2])
            labels_torch = torch.ones(coords_torch.shape[0], coords_torch.shape[1]).long() # (B, Ps)
            coords_torch, labels_torch = coords_torch.to(self.device), labels_torch.to(self.device)
            point_prompt = (coords_torch, labels_torch)
        
        boxes_torch = None
        if self.use_boxes:
            boxes_torch = batch["boxes"].squeeze().to(self.device) # ([B, Ps, 4])

        medsam_lite_pred = self.model(image, point_prompt, boxes_torch)

        dice_loss = self.dice_loss_fn(medsam_lite_pred, gt2D)
        ce_loss = self.ce_loss_fn(medsam_lite_pred, gt2D.float())

        return dice_loss, ce_loss


    def train_step(self, step, batch):
        self.loggingHandler.log(f'Starting epoch {self.current_epoch} and step {step} out of {self.batches_per_epoch}')

        dice_loss, ce_loss = self.forward_pass(batch)

        loss = dice_loss + ce_loss
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.loggingHandler.log(f'[TRAINING]:   Received dice loss: {dice_loss.item()} with cross entropy loss: {ce_loss.item()}')
        self.loggingHandler.log_metric('dice_loss', dice_loss.item(), self.current_epoch)
        self.loggingHandler.log_metric('ce_loss', ce_loss.item(), self.current_epoch)

        return dice_loss.item(), ce_loss.item()

    def on_epoch_end(self, epoch):
        self.loggingHandler.end_current_epoch(epoch)

        # Get reduced loss
        dice_loss = self.loggingHandler.curr_epoch_stats['dice_loss']
        ce_loss = self.loggingHandler.curr_epoch_stats['ce_loss']
        epoch_loss_reduced = (sum(dice_loss) + sum(ce_loss)) / (len(dice_loss) + len(ce_loss))

        if epoch_loss_reduced < self.best_loss:
            self.best_loss = epoch_loss_reduced
        
        self.checkpointHandler.save_checkpoint(self.model, self.optimizer, epoch, epoch_loss_reduced, self.best_loss, final=False)

        self.loggingHandler.plot_stats()

        self.loggingHandler.log('=====================================')

    def on_validation_epoch_start(self):
        self.loggingHandler.log('Starting validation epoch')

    def validation_step(self, batch_id, batch):
        self.loggingHandler.log(f'Validation step {batch_id} out of {min(len(self.dataloaderHandler.val_loader), self.batches_per_epoch)}')

        dice_loss, ce_loss = self.forward_pass(batch)

        self.loggingHandler.log(f'[VALIDATION]: Received dice loss: {dice_loss.item()} with cross entropy loss: {ce_loss.item()}')
        self.loggingHandler.log_metric('val_dice_loss', dice_loss.item(), self.current_epoch)
        self.loggingHandler.log_metric('val_ce_loss', ce_loss.item(), self.current_epoch)

        return dice_loss.item(), ce_loss.item()

    def on_validation_epoch_end(self):
        
        # Get reduced loss
        dice_loss = self.loggingHandler.curr_epoch_stats['val_dice_loss']
        ce_loss = self.loggingHandler.curr_epoch_stats['val_ce_loss']
        epoch_loss_reduced = (sum(dice_loss) + sum(ce_loss)) / (len(dice_loss) + len(ce_loss))

        self.loggingHandler.log('Validation epoch ended')
        self.loggingHandler.log(f'Average loss: {epoch_loss_reduced}')