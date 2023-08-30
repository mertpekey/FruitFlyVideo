class FlyInfo:
    def __init__(self, name, trial_id, peak_index, peak_values):
        self.name = name
        self.trial_id = trial_id
        self.peak_index = peak_index
        self.peak_values = peak_values

class FlyDatabase:
    def __init__(self):
        self.fly_data = []

    def add_fly(self, fly_info):
        self.fly_data.append(fly_info)

    def get_fly(self, name, trial_id):
        for fly_info in self.fly_data:
            if fly_info.name == name and fly_info.trial_id == trial_id:
                return fly_info
        return None
    
    def write_fly_info(self, name, trial_id):
        for fly_info in self.fly_data:
            if fly_info.name == name and fly_info.trial_id == trial_id:
                print('Name:', fly_info.name)
                print('Trial Id:', fly_info.trial_id)
                print('Peak Index:', fly_info.peak_index)
                print('Peak Values:', fly_info.peak_values)
                return None
        print('Fly not found!!!')
        return None
    
    def calculate_peak_ratio(self, bouts_dict):
        import numpy as np
        peak_ratios = []
        for fly in self.fly_data:
            data_length = len(bouts_dict[fly.name].loc[int(fly.trial_id)]['distance.origin-prob'])
            peak_amount = len(fly.peak_index)
            peak_ratios.append(peak_amount / data_length)

        peak_ratio = np.mean(peak_ratios)
        max_ratio = np.max(peak_ratios)
        min_ratio = np.min(peak_ratios)

        print(f'Ratio: {peak_ratio}, Min: {min_ratio}, Max: {max_ratio}')