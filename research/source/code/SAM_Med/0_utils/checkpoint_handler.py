# Imports
import os
import torch
import torch.optim as optim
from segment_anything import sam_model_registry
from medsam_model import MedSAM

class CheckpointHandler():
    def __init__(
            self,
            save_dir: str,
            checkpoint_path: str,
            device,
            *args,
            **kwargs
    ):
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']

    def save_checkpoint(self, model, optimizer, epoch, epoch_loss, best_loss, final):
        """
        Will be guaranteed to save the checkpoint in the save_dir location. If the model
        is at peak performance, it saves it under 'checkpoint_best' otherwise, by default
        it is 'checkpoint_latest'. If specified, the checkpoint is saved under its final
        form and thus replaces 'checkpoint_latest' -> 'checkpoint_final'.
        """

        checkpoint = {
            "model": model.state_dict(),
            "epochs": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss,
            "best_loss": best_loss
        }

        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_best.pth'))

        if final:
            torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_final.pth'))
            os.remove(os.path.join(self.checkpoint_path, 'checkpoint_latest.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_latest.pth'))

        return best_loss

    def load_checkpoint(self):
        """
        Loads a checkpoint from the save_dir directory. Assumes this function will be called in the context of continuing training that hasn't finished yet.
        """
        model, optimizer = self.load_base_checkpoint()
        assert self.checkpoint_exists(), "did you check that checkpoint_latest exists?"
        checkpoint = torch.load(os.path.join(self.save_dir, 'checkpoint_latest.pth'))

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epochs']
        best_loss = checkpoint['best_loss']

        return model, optimizer, epoch, best_loss

    def load_base_checkpoint(self):
        sam_model = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path)

        medsam_model = MedSAM(
            image_encoder = sam_model.image_encoder,
            mask_decoder = sam_model.mask_decoder,
            prompt_encoder = sam_model.prompt_encoder,
            freeze_image_encoder = True
        )
        medsam_model = medsam_model.to(self.device)

        optimizer = optim.AdamW(
            medsam_model.mask_decoder.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay
        )

        return medsam_model, optimizer

    def checkpoint_exists(self):
        return os.path.exists(os.path.join(self.save_dir, 'checkpoint_latest.pth'))

    def final_checkpoint_exists(self):
        return os.path.exists(os.path.join(self.save_dir, 'checkpoint_final.pth')) 