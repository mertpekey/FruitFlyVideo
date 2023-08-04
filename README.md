# Fruit Fly Video Classification

From the fruit fly video clips, the model predicts 3 fly behaviours; Feeding, Pumping and Grooming.

## Model Overview

The model used in this project is Timesformer. It's designed for video classification tasks. The training parameters include:
- `sample_rate`: 16
- `learning_rate`: 1e-3
- `num_frames`: 8
- Image size: 224x224
- Model input shape: `(batch_size, num_frames, channel, height, width)`

## Installation

You can install the required libraries using the following pip commands:

```bash
pip install lightning torch transformers torchvision wandb pytorchvideo
```

## Running Inference

To run inference using the provided `inference.py` file, follow these steps:

1. Clone the project repository:

    ```bash
    git clone https://github.com/mertpekey/FruitFlyVideo.git
    ```

2. Create a directory named `Prediction_Data`, and inside that directory, create a `Test` folder. Place the images you want to make inferences on inside the `Test` folder. You can modify the folder structure and image names if needed.

3. Create a directory named `Pretrained_Model`, and download the pretrained model inside that folder. Change the name of the file to `pretrained_model.ckpt`. Or you can define it using `--model_name` argument.

4. Run the inference script with the following command:

    ```bash
    python inference.py --inference_data_path 'Prediction_Data' --batch_size 1 --device cuda --load_ckpt True
    ```

   - `--inference_data_path`: Path to the `Prediction_Data` folder.
   - `--batch_size`: (Optional) Set the batch size for inference (default: 1).
   - `--device`: (Optional) Set the device to run inference on (e.g., `cuda` or `cpu`).
   - `--load_ckpt`: (Optional) Use this flag to indicate using a finetuned model (default: True).
   - `--model_name`: (Optional) Use this to determine which pretrained checkpoints will be used (default: pretrained_model.ckpt).

4. The inference script will generate a `prediction.json` output file in the project directory, containing the inference results.

Please make sure to adjust paths and arguments as needed for your project.

---