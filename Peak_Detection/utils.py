import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


######## Reading & Creating Data ########

def read_pickle(filename):
    with open(filename, 'rb') as f: 
        loaded_file = pickle.load(f)
    return loaded_file


def create_input_data(fly, experiment, features, bouts_dict):
    input_data = bouts_dict[fly][features[0]][experiment].reshape(-1,1)
    for i in range(1, len(features)):
        input_data = np.concatenate((input_data,
                                    bouts_dict[fly][features[i]][experiment].reshape(-1,1)),
                                    axis=1)
    return input_data


def create_fly_database(bouts_dict, true_peak_annotations_df=None):
    from FlyInfo import FlyDatabase, FlyInfo
    
    fly_db = FlyDatabase()
    all_fly_names = list(bouts_dict.keys())
    fly_names_annot = true_peak_annotations_df['name'].unique().tolist()

    for name in all_fly_names:
        true_idx = []
        if name in fly_names_annot:
            true_idx = true_peak_annotations_df[true_peak_annotations_df['name'] == name]['trial_id'].unique().astype(int).tolist()

        for idx in range(len(bouts_dict[name])):
            if idx in true_idx:
                peak_index = true_peak_annotations_df[(true_peak_annotations_df['name'] == name) & (true_peak_annotations_df['trial_id'] == str(idx))]['peak_index'].values
                peak_values = true_peak_annotations_df[(true_peak_annotations_df['name'] == name) & (true_peak_annotations_df['trial_id'] == str(idx))]['value'].values
                fly_db.add_fly(FlyInfo(name, idx, peak_index, peak_values))
            else:
                fly_db.add_fly(FlyInfo(name, idx, None, None))
    return fly_db


######## Feature Engineering ########

def add_mv_stats(df):
    df['moving_avg'] = df['distance.origin-prob'].rolling(window=10).mean()
    df['moving_std'] = df['distance.origin-prob'].rolling(window=10).std()
    
    # fill NaN values with its corresponding row value
    df['moving_avg'].fillna(df['distance.origin-prob'], inplace=True)
    df['moving_std'].fillna(df['distance.origin-prob'], inplace=True)
    return df


######## Model ########

def create_config(features = ['distance.origin-prob','distance.head-prob', 'pose.prob_x','pose.prob_y'],
                  contamination = 0.04,
                  grouped_range = 60,
                  add_golay = False,
                  add_mv_stats = False,
                  show_true_annotations = True):
    config = {}
    config['features'] = features
    config['contamination'] = contamination
    config['grouped_range'] = grouped_range
    config['add_golay'] = add_golay
    config['add_mv_stats'] = add_mv_stats
    config['show_true_annotations'] = show_true_annotations
    return config


def get_model_prediction(fly, config, bouts_dict):
    input_data = create_input_data(fly = fly.name,
                                   experiment = int(fly.trial_id),
                                   features = config['features'],
                                   bouts_dict = bouts_dict)

        
    info_df = pd.DataFrame(input_data, columns = config['features'])

    if config['add_mv_stats']:
        info_df = add_mv_stats(info_df)
    
    if config['add_golay']:
        from scipy.signal import savgol_filter
        golay_cols = ['pose.thor_post_x', 'pose.thor_post_y']
        for col in golay_cols:
            feat = bouts_dict[fly.name][col][int(fly.trial_id)]
            info_df[f'gol_{col}'] = savgol_filter(x = feat, window_length=len(feat), polyorder=5, axis=0)
        
    input_data = info_df.to_numpy()

    scaled_data = StandardScaler().fit_transform(input_data)
    model = IsolationForest(n_estimators = 100,
                            contamination=config['contamination'],
                            max_samples='auto',
                            max_features=1.0,
                            bootstrap=False)
    info_df['predictions'] = model.fit_predict(scaled_data)
    return info_df


######## Evaluation ########

def calculate_grouped_recall(predicted_peaks, true_peaks, matching_range):
    recall_predictions = []
    for true_idx in true_peaks:
        found_true_pred = False
        for pred_idx in predicted_peaks:
            if abs(pred_idx - true_idx) <= matching_range:
                found_true_pred = True
                break
        recall_predictions.append(found_true_pred)
    recall = np.sum(recall_predictions) / len(recall_predictions)
    return recall


def calculate_grouped_precision(predicted_peaks, true_peaks, matching_range):
    precision_predictions = []
    for pred_idx in predicted_peaks:
        found_pred = False
        for true_idx in true_peaks:
            if abs(pred_idx - true_idx) <= matching_range:
                found_pred = True
                break
        precision_predictions.append(found_pred)
    precision = np.sum(precision_predictions) / len(precision_predictions)
    return precision


def calculate_f1_score(precision, recall):
    if precision + recall > 0.0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0.0


def evaluate_results(all_results, matching_range = 30):
    avg_recall, avg_precision, avg_f1_score = 0, 0, 0
    all_recall, all_precision, all_f1_score, all_true_pred_ratio = [], [], [], []

    for res in all_results:
        predicted_peaks = res['predicted_index']
        true_peaks = res['true_index']
        
        recall = calculate_grouped_recall(predicted_peaks, true_peaks, matching_range)
        precision = calculate_grouped_precision(predicted_peaks, true_peaks, matching_range)
        f1_score = calculate_f1_score(precision, recall)
        
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1_score.append(f1_score)
        all_true_pred_ratio.append(len(predicted_peaks) / len(true_peaks))

        avg_recall += recall
        avg_precision += precision
        avg_f1_score += f1_score

    avg_recall /= len(all_results)
    avg_precision /= len(all_results)
    avg_f1_score /= len(all_results)
    
    return {
        'avg_metrics' : {
            'avg_recall' : avg_recall,
            'avg_precision' : avg_precision,
            'avg_f1_score' : avg_f1_score
            },
        'all_metrics' : {
            'all_recall' : all_recall,
            'all_precision' : all_precision,
            'all_f1_score' : all_f1_score
            },
        'true_pred_amount' : {
            'true_amounts' : [len(res['true_index']) for res in all_results],
            'pred_amounts' : [len(res['predicted_index']) for res in all_results]
            }
        }


def evaluate_model(fly_db, config, bouts_dict):
    all_results, all_results_group = [], []

    for fly in fly_db.fly_data:
        if fly.peak_index is not None:
            info_df = get_model_prediction(fly, config, bouts_dict)
            
            anomalies = info_df.loc[info_df['predictions'] == -1, ['distance.origin-prob']]
            anomalies_idx = list(anomalies.index)

            group_pred_idx, group_pred_val = filter_prediction(anomalies, grouped_range=config['grouped_range'])
            
            all_results.append({'true_index': fly.peak_index, 'predicted_index': anomalies_idx})
            all_results_group.append({'true_index': fly.peak_index, 'predicted_index': group_pred_idx})

    results = evaluate_results(all_results)
    results_g = evaluate_results(all_results_group)
    return results, results_g


######## Post Processing ########

def filter_prediction(anomalies, grouped_range=60):
    idx_group = []
    all_groups = []
    for idx in list(anomalies.index):
        if idx in idx_group:
            continue
        idx_group = []
        for i in range(idx, idx+grouped_range):
            if i in list(anomalies.index):
                idx_group.append(i)
        all_groups.append(idx_group)
    
    group_pred_idx = []
    group_pred_val = []
    for group in all_groups:
        group_vals = anomalies.loc[group]['distance.origin-prob']
        if len(group_vals) > 1 and group_vals.iloc[0] < group_vals.iloc[1]:
            pred = group_vals[group_vals == group_vals.max()]
        else:
            pred = group_vals[group_vals == group_vals.min()]
        group_pred_idx.append(pred.index.values[0])
        group_pred_val.append(pred.values[0])
    return group_pred_idx, group_pred_val


######## Analysis ########

def plot_peak_predictions(fly_db, fly_names, experiments, config, bouts_dict):

    if len(fly_names) != len(experiments):
        raise ValueError("The lengths of fly_names and experiments must match.")

    if not os.path.exists('peak_plots_output'):
        os.makedirs('peak_plots_output')

    for fly_nm, exp_id in zip(fly_names, experiments):
        fly = fly_db.get_fly(fly_nm, int(exp_id))
        if fly is None:
            print(f'Fly Name: {fly_nm}, Experiment ID: {exp_id} Not Found!!!')
            continue
        
        info_df = get_model_prediction(fly, config, bouts_dict)

        anomalies = info_df.loc[info_df['predictions'] == -1, ['distance.origin-prob']]
        anomalies_idx = list(anomalies.index)
        group_pred_idx, group_pred_val = filter_prediction(anomalies, grouped_range=60)

        plot_and_save_data(info_df.index, info_df['distance.origin-prob'], anomalies.index, anomalies['distance.origin-prob'], fly, config['show_true_annotations'], False)
        plot_and_save_data(info_df.index, info_df['distance.origin-prob'], group_pred_idx, group_pred_val, fly, config['show_true_annotations'], True)


def plot_and_save_data(data_x, data_y, peak_x, peak_y, fly, show_annot, is_filtered):
    plt.plot(data_x, data_y, color='black', label = 'Normal')
    plt.scatter(peak_x, peak_y, color='red', label = 'Anomaly')
    if show_annot and fly.peak_index is not None:
        for true_peaks in fly.peak_index:
            min_range, max_range = true_peaks - 30, true_peaks + 30
            plt.axvspan(min_range, max_range, color='yellow', alpha=0.3)
    plt.title("Peak Prediction")
    plt.legend(fontsize='small')
    plot_filename_all = os.path.join('peak_plots_output', f"{fly.name}_{fly.trial_id}_{'filtered' if is_filtered else 'all'}.png")
    plt.savefig(plot_filename_all)
    plt.close()