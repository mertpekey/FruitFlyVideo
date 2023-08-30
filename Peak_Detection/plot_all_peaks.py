import os
import utils
import matplotlib.pyplot as plt

def plot_all_peaks(config):
    base_dir = 'all_peak_output'

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    bouts_dict = utils.read_pickle('bouts_dict.pkl')
    true_peak_annotations_df = utils.read_pickle('true_annotations.pkl')
    fly_db = utils.create_fly_database(bouts_dict, true_peak_annotations_df=true_peak_annotations_df)

    for fly in fly_db.fly_data:
        info_df = utils.get_model_prediction(fly, config, bouts_dict)
        
        anomalies = info_df.loc[info_df['predictions'] == -1, ['distance.origin-prob']]
        group_pred_idx, group_pred_val = utils.filter_prediction(anomalies, grouped_range=config['grouped_range'])

        plt.plot(info_df.index, info_df['distance.origin-prob'], color='black', label = 'Normal')
        plt.scatter(group_pred_idx, group_pred_val, color='red', label = 'Anomaly')
        plt.title("Predicted Peaks")
        plt.legend(fontsize='small')
        
        if not os.path.exists(os.path.join(base_dir, fly.name)):
            os.makedirs(os.path.join(base_dir, fly.name))

        plot_filename_all = os.path.join(base_dir, fly.name, f"{fly.name}_{fly.trial_id}.png")
        plt.savefig(plot_filename_all)
        plt.close()

if __name__ == '__main__':
    
    config = utils.create_config(features = ['distance.origin-prob','distance.head-prob', 'pose.prob_x','pose.prob_y'],
                                contamination = 0.04,
                                grouped_range = 60,
                                add_golay = False,
                                add_mv_stats = False,
                                show_true_annotations = False)
    plot_all_peaks(config)