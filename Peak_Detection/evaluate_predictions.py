import argparse
import utils

def main(args):
    bouts_dict = utils.read_pickle('bouts_dict.pkl')
    true_peak_annotations_df = utils.read_pickle('true_annotations.pkl')

    fly_db = utils.create_fly_database(bouts_dict, true_peak_annotations_df)
    config = utils.create_config(features=args.features,
                                 contamination=args.contamination,
                                 grouped_range=args.grouped_range,
                                 add_golay=args.add_golay,
                                 add_mv_stats=args.add_mv_stats)
    results, results_g = utils.evaluate_model(fly_db, config, bouts_dict)

    print('Results - All Predictions')
    print(results['avg_metrics'])
    print()
    print('Results - Filtered Predictions')
    print(results_g['avg_metrics'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Isolation Forest models')
    
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
    main(args)
