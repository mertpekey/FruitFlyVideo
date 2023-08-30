# Peak Detection

Peak Detection of Time Series data using Isolation Forest model.

## Installation

You can install the required libraries using the following pip commands:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Data

The project uses `bouts_dict.pkl` and `true_annotations.pkl` files to get the time series informations. If you need to add new data, you should modify these data or create new data in similar format.

### Arguments

#### Common Arguments

   - `--features`: Features used to predict peaks
   - `--contamination`: Contamination value
   - `--grouped_range`: Grouped range for peaks. 60 means -/+ 30
   - `--add_golay`: Flag to add golay filters for `pose.thor_post_x, pose.thor_post_y` features.
   - `--add_mv_stats`: Flag to add moving avg and std for `distance.origin-prob` feature

### Predicting Plots

1. In a txt file, you should add fly name and experiment id as the following format:

```
Fly05182022_5d -- 0
Fly06072022_5d -- 16
```

2. Run the following script to generate the plots in folder called `peak_plots_output`. The `fly_experiments_fn` argument is only for plot_peaks.py

```bash
    python plot_peaks.py --fly_experiments_fn your_filename.txt
```

### Evaluating Model Predictions

1. To evaluate the model predictions on the data in `bouts_dict.pkl` file (Only the ones which has true annotations in `true_annotations.pkl`), run the following script

```bash
    python evaluate_predictions.py --features distance.origin-prob distance.head-prob pose.prob_x pose.prob_y --contamination 0.04 --grouped_range 60 --add_golay --add_mv_stats
```

---

# Peak Detection

This project focuses on detecting peaks in time series data using the Isolation Forest model.

## Installation

Install the required libraries using the following pip commands:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Data

The project relies on two essential files, `bouts_dict.pkl` and `true_annotations.pkl`, to extract time series information. If you wish to introduce new data, consider modifying these existing files or creating new data with a similar format.

### Usage

#### Common Arguments

- `--features`: Specify the features used for peak prediction.
- `--contamination`: Set the contamination value.
- `--grouped_range`: Define the range for peak grouping (e.g., 60 corresponds to a range of -30 to +30).
- `--add_golay`: Include Golay filters for `pose.thor_post_x` and `pose.thor_post_y` features.
- `--add_mv_stats`: Add moving averages and standard deviations for the `distance.origin-prob` feature.

### Predicting Plots

1. Prepare a text file containing fly names and corresponding experiment IDs in the format:

```
Fly05182022_5d -- 0
Fly06072022_5d -- 16
```

2. Generate peak prediction plots using the following command, saving them in the `peak_plots_output` folder. Note that the `fly_experiments_fn` argument is specific to `plot_peaks.py`.

```bash
python plot_peaks.py --fly_experiments_fn your_filename.txt
```

### Evaluating Model Predictions

1. Evaluate model predictions on the data in the `bouts_dict.pkl` file, considering only those instances with true annotations in `true_annotations.pkl`. Use the following command:

```bash
python evaluate_predictions.py --features distance.origin-prob distance.head-prob pose.prob_x pose.prob_y --contamination 0.04 --grouped_range 60 --add_golay --add_mv_stats
```

2. I recommend avoiding the addition of Golay filters and moving statistics, as these tend to decrease the overall performance of the model.
