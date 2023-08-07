import os
import argparse

PROJ_DIR = os.getcwd()
TRAIN_DATA_PATH = os.path.join(PROJ_DIR, 'FlyTrainingData', 'Train')
VAL_DATA_PATH = os.path.join(PROJ_DIR, 'FlyTrainingData', 'Validation')

parser = argparse.ArgumentParser(description="Enter Arguments for Video Fly")

# Add the arguments to the parser
parser.add_argument("--mode", type=str, default='train', help="Train or Test the model ('train' or 'test')")
parser.add_argument("--finetune_head", type=bool, default=True, help="Finetune only classification head or whole model")
parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH, help="Path to training data")
parser.add_argument("--val_data_path", type=str, default=VAL_DATA_PATH, help="Path to validation data")
parser.add_argument("--ckpt_file_path", type=str, default='', help="Checkpoint file path to load model")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--max_epochs", type=int, default=25, help="Maximum number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--video_path_prefix", type=str, default="", help="Prefix for video paths")
parser.add_argument("--video_min_short_side_scale", type=int, default=256, help="Minimum short side scale for videos")
parser.add_argument("--video_max_short_side_scale", type=int, default=320, help="Maximum short side scale for videos")
parser.add_argument("--sample_rate", type=int, default=16, help="Sample rate")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--num_frames", type=int, default=8, help="Number of frames")
# Parse the arguments and store them in a variable
args = parser.parse_args()
