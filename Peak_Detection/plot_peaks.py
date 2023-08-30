import argparse
import utils

def main(fly_data, args):
    bouts_dict = utils.read_pickle('bouts_dict.pkl')
    true_peak_annotations_df = utils.read_pickle('true_annotations.pkl')

    fly_db = utils.create_fly_database(bouts_dict, true_peak_annotations_df)
    config = utils.create_config(features=args.features,
                                 contamination=args.contamination,
                                 grouped_range=args.grouped_range,
                                 add_golay=args.add_golay,
                                 add_mv_stats=args.add_mv_stats)
    
    fly_names, experiments = [], []
    for line in fly_data:
        parts = line.strip().split(' -- ')
        if len(parts) == 2:
            fly_names.append(parts[0])
            experiments.append(parts[1])

    utils.plot_peak_predictions(fly_db, fly_names, experiments, config, bouts_dict)
    print('Plots created')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments for Plotting Predictions')

    parser.add_argument('--fly_experiments_fn', required=True,
                        help='Filename containing fly names and experiment IDs')
    parser.add_argument('--features', nargs='+', default=['distance.origin-prob', 'distance.head-prob', 'pose.prob_x', 'pose.prob_y'],
                        help='List of feature names')
    parser.add_argument('--contamination', type=float, default=0.04,
                        help='Contamination value')
    parser.add_argument('--grouped_range', type=int, default=60,
                        help='Grouped range value')
    parser.add_argument('--add_golay', action='store_true',
                        help='Flag to add Golay filtering')
    parser.add_argument('--add_mv_stats', action='store_true',
                        help='Flag to add moving statistics')
    
    args = parser.parse_args()

    with open(args.fly_experiments_fn, 'r') as f:
        fly_data = f.readlines()

    main(fly_data, args)
