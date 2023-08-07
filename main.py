import os
import torch

import lightning.pytorch as pl
from transformers import AutoImageProcessor
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint

from arguments import args
from utils import create_preprocessor_config, get_timesformer_model, load_model_from_ckpt
from model import VideoClassificationLightningModule
from data_module import FlyDataModule

def main():

    # DATASET INFO
    class_labels = sorted([i for i in os.listdir(args.train_data_path) if i != '.DS_Store'])
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # GET MODEL
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = get_timesformer_model(ckpt="facebook/timesformer-base-finetuned-k400",
                                  label2id=label2id,
                                  id2label=id2label,
                                  num_frames=args.num_frames)
    
    # Freeze the timesformer model to finetune classification head    
    if args.finetune_head:
        for param in model.timesformer.parameters():
            param.requires_grad = False


    # Create Arguments
    model_args = create_preprocessor_config(model, 
                                            image_processor, 
                                            sample_rate=args.sample_rate, 
                                            fps=args.fps)
    for key, value in model_args.items():
        setattr(args, key, value)


    if args.ckpt_file_path != '':
        #saved_ckpt = "tb_logs/timesformer_logs_s16_noES_b16_lr1e3/version_0/checkpoints/epoch=24-step=1000.ckpt"
        model = load_model_from_ckpt(model, args.ckpt_file_path)
        classification_module = VideoClassificationLightningModule(model, args)
    else:
        classification_module = VideoClassificationLightningModule(model, args)
    data_module = FlyDataModule(args)

    # Lightning Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="max",
        dirpath="./wandb_checkpoints",
        filename=f"timesformer_b{args.batch_size}_lr{args.lr}_sr{args.sample_rate}",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=None,
        callbacks=[TQDMProgressBar(refresh_rate=args.batch_size), checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )

    if args.mode is None:
        print('Please enter a mode!')
    elif args.mode == 'train':
        trainer.fit(classification_module, data_module)
    elif args.mode == 'test':
        trainer.test(classification_module, data_module)
    elif args.mode == 'predict':
        data_module.setup()
        classification_module.dataloader_length = len(data_module.val_dataloader())
        print(classification_module.dataloader_length)
        trainer.predict(classification_module, data_module.val_dataloader())


if __name__ == '__main__':
    main()