import os
import time
import json
import argparse
import torch

from transformers import AutoImageProcessor
from arguments import args
from utils import create_preprocessor_config, get_timesformer_model, load_model_from_ckpt
from data_module import FlyDataModule

## Arguments

INFERENCE_PATH = 'Prediction_Data'
parser = argparse.ArgumentParser(description="Enter Arguments for Video Fly")
parser.add_argument("--inference_data_path", type=str, default=INFERENCE_PATH, help="Path to inference data folder")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--video_path_prefix", type=str, default="", help="Prefix for video paths")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--load_ckpt", type=bool, default=True, help="Load finetuned model or not")
args = parser.parse_args()

device = torch.device(args.device)

# Dataset Info
class_labels = ['Feeding', 'Grooming', 'Pumping']
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}


# Get model and create model configs
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = get_timesformer_model(ckpt="facebook/timesformer-base-finetuned-k400",
                              label2id=label2id,
                              id2label=id2label,
                              num_frames=8)

model_args = create_preprocessor_config(model,
                                        image_processor, 
                                        sample_rate=16, 
                                        fps=30)
model_args['video_min_short_side_scale'] = 256
model_args['video_max_short_side_scale'] = 320
for key, value in model_args.items():
    setattr(args, key, value)

# Load model checkpoints
if args.load_ckpt:
    saved_ckpt = "tb_logs/timesformer_logs_s16_noES_b16_lr1e3/version_0/checkpoints/epoch=24-step=1000.ckpt"
    model = load_model_from_ckpt(model, saved_ckpt)
    print('Model Loaded!')

model.to(device).eval()

# Create dataset and dataloader
data_module = FlyDataModule(args)
data_module.setup(stage='inference')
dataloader = data_module._inference_dataloader()
print('Dataset Created')

print('Prediction Starts...')
inf_start = time.time()
# Forward Pass
output_preds = {}

for batch in dataloader:
    video = batch['video'].to(device)

    with torch.no_grad():
        output = model(video.permute(0, 2, 1, 3, 4))
    probs = torch.softmax(output.logits, dim=1)
    predictions = torch.argmax(output.logits, dim=1)

    for i in range(len(batch['video_name'])):
        if batch['video_name'][i] in output_preds:
            output_preds[batch['video_name'][i]]['clip_index'].append(batch['clip_index'][i].item())
            output_preds[batch['video_name'][i]]['prediction'].append(id2label[predictions[i].item()])
            output_preds[batch['video_name'][i]]['probs'].append(probs[i].tolist())
        else:
            output_preds[batch['video_name'][i]] = {'clip_index' : [batch['clip_index'][i].item()],'prediction': [id2label[predictions[i].item()]], 'probs' : [probs[i].tolist()]}

inf_end = time.time()
print(f'Prediction of {len(output_preds)} videos took {inf_end - inf_start} seconds')

with open('predictions.json', 'w') as jsonfile:
    json.dump(output_preds, jsonfile, indent=4)
print('JSON file created')


