import torch
from transformers import AutoImageProcessor,TimesformerForVideoClassification

def create_preprocessor_config(model, image_processor, sample_rate=8, fps=30):

    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]

    crop_size = (height, width)

    num_frames_to_sample = model.config.num_frames # 16 for VideoMAE
    clip_duration = num_frames_to_sample * sample_rate / fps
    print('Clip Duration:', clip_duration, 'seconds')

    return {
        "video_means" : mean,
        "video_stds" : std,
        "crop_size" : crop_size,
        "num_frames_to_sample" : num_frames_to_sample,
        "clip_duration": clip_duration
    }

def get_timesformer_model(ckpt, label2id, id2label, num_frames):
    return TimesformerForVideoClassification.from_pretrained(
        ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        num_frames = num_frames
    )

def load_model_from_ckpt(model, ckpt):
    state_dict_model = torch.load(ckpt)['state_dict']
    for key in list(state_dict_model.keys()):
        state_dict_model[key.replace('model.', '')] = state_dict_model.pop(key)
    model.load_state_dict(state_dict_model)
    return model