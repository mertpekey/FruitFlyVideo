import itertools
import torch
import lightning.pytorch as pl
import pytorchvideo.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

class FlyDataModule(pl.LightningDataModule):

    def __init__(self, args):
        self.args = args
        super().__init__()

    
    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            train_transform = self._make_transforms(mode="train")
            val_transform = self._make_transforms(mode="val")

            self.train_dataset = LimitDataset(
                pytorchvideo.data.labeled_video_dataset(
                    data_path=self.args.train_data_path,
                    clip_sampler=pytorchvideo.data.make_clip_sampler('uniform', self.args.clip_duration),
                    transform=train_transform,
                    video_path_prefix=self.args.video_path_prefix, # could be '' I think
                    decode_audio=False
                )
            )

            self.val_dataset = LimitDataset(
                pytorchvideo.data.labeled_video_dataset(
                    data_path=self.args.val_data_path,
                    clip_sampler=pytorchvideo.data.make_clip_sampler('uniform', self.args.clip_duration),
                    transform=val_transform,
                    video_path_prefix=self.args.video_path_prefix, # could be '' I think
                    decode_audio=False
                )
            )
        elif stage == 'inference':
            val_transform = self._make_transforms(mode="val")
            self.inference_dataset = pytorchvideo.data.labeled_video_dataset(
                    data_path=self.args.inference_data_path,
                    clip_sampler=pytorchvideo.data.make_clip_sampler('uniform', self.args.clip_duration), # Experiment olarak random da denenebilir
                    transform=val_transform,
                    video_path_prefix=self.args.video_path_prefix, # could be '' I think
                    decode_audio=False
                )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8
        )
    
    def _inference_dataloader(self):
        return torch.utils.data.DataLoader(
            self.inference_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )

    def _make_transforms(self, mode: str):
        return Compose([self._video_transform(mode)])


    def _video_transform(self, mode: str):
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.args.num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.args.video_means, self.args.video_stds),
                ]
                + (
                    [
                        RandomShortSideScale(
                            min_size=self.args.video_min_short_side_scale,
                            max_size=self.args.video_max_short_side_scale,
                        ),
                        RandomCrop(self.args.crop_size),
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(self.args.video_min_short_side_scale),
                        CenterCrop(self.args.crop_size),
                    ]
                )
            ),
        )
    

class LimitDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos