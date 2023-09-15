import pandas as pd
import numpy as np
from datetime import datetime
        

class HKRaceData:
    def __init__(self, runs_data_path, races_data_path, history_length,
                 current_run_features, past_run_features, race_outcome_features, 
                 race_sequence_features, normalize=True, subsample=None, max_horses_per_race=None, 
                 only_running_horse_past=False, skip_multi_wins=True, debug=False):
        self._runs_data = pd.read_csv(runs_data_path)
        self._races_data = pd.read_csv(races_data_path)
        
        self._only_running_horse_past = only_running_horse_past
        self._normalize = normalize
        self._max_horses_per_race = max_horses_per_race
        self._skip_multi_wins = skip_multi_wins
        self._subsample = subsample
        
        self._debug = debug
        
        self.past_run_features = past_run_features
        self.current_run_features = current_run_features
        self.race_outcome_features = race_outcome_features
        self.race_sequence_features = race_sequence_features
        
        self.history_length = history_length
        self.horses_per_race = None
        self.race_ids = None
        
        self._fix_data()
        if self._normalize:
            self._normalize_data()
        self._set_race_ids()
        self._set_past_race_ids()
        
    def _check(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if np.any([res != res for res in result]):
                raise Exception(f"Found false value in {result}")
            
            return result
        
        return wrapper
    
    def _fix_data(self):
        lengths_behind = self._runs_data["lengths_behind"]
        lengths_behind = lengths_behind.mask(self._runs_data["lengths_behind"] < 0, 0)
    
    def _normalize_data(self):
        distance = self._races_data['distance']
        distance = (distance - distance.min()) \
            / (distance.max() - distance.min())
        self._races_data['distance'] = distance
        
        win_odds = self._runs_data["win_odds"]
        win_odds = (win_odds - win_odds.min()) \
            / (win_odds.max() - win_odds.min())
        self._runs_data["win_odds"] = win_odds
        
        finish_time = self._runs_data["finish_time"]
        finish_time = (finish_time - finish_time.min()) \
            / (finish_time.max() - finish_time.min())
        self._runs_data["finish_time"] = finish_time
        
        lengths_behind = self._runs_data["lengths_behind"]
        lengths_behind = (lengths_behind - lengths_behind.min()) \
            / (lengths_behind.max() - lengths_behind.min())
        self._runs_data["lengths_behind"] = lengths_behind
        
        
    def _set_race_ids(self):
        race_ids = self._races_data['race_id'].values
        dates = self._races_data['date'].values
        
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        arg = sorted(range(len(dates)), key=dates.__getitem__)
        
        race_ids = np.array(race_ids)[arg]
        
        if self._subsample:
            race_ids = race_ids[self._subsample]
            
        max_horses = 0
        valid_race_ids = []
        
        for race_id in race_ids:
            runs = self._get_race_runs_data(race_id)
            n_horses = len(runs.index)
            
            if runs["won"].sum() != 1 and self._skip_multi_wins:
                continue
            
            if self._max_horses_per_race and n_horses > self._max_horses_per_race:
                continue
            
            valid_race_ids.append(race_id)
            
            if n_horses > max_horses:
                max_horses = n_horses
        
        self.horses_per_race = max_horses
        self.race_ids = list(valid_race_ids)
        
    def _set_past_race_ids(self):
        past_race_ids = {}
        
        for race_id in self.race_ids:
            index = self.race_ids.index(race_id)
                
            runs = self._get_race_runs_data(race_id)
            race_horse_ids = runs['horse_id'].values
            
            value = {horse_id: [] for horse_id in race_horse_ids}
            
            for other_race_id in self.race_ids[:index][::-1]:
                if np.all([len(v) >= self.history_length for v in value.values()]):
                    break
                
                runs = self._get_race_runs_data(other_race_id)
                other_race_horse_ids = runs['horse_id'].values
                
                for horse_id in race_horse_ids:
                    if horse_id in other_race_horse_ids and len(value[horse_id]) < self.history_length:
                        value[horse_id].append(other_race_id)
                        
            past_race_ids[race_id] = value
            
        self.past_race_ids = past_race_ids
    
    def _get_run_data(self, race_id, horse_id):
        runs = self._get_race_runs_data(race_id)
        
        run = runs[runs['horse_id'] == horse_id]
        
        return run
    
    def get_race_data(self, race_id):
        race = self._races_data[self._races_data["race_id"] == race_id]
        
        return race
    
    def _get_race_runs_data(self, race_id):
        runs = self._runs_data[self._runs_data["race_id"] == race_id]
        
        return runs
    
    @_check 
    def _get_run_features(self, race_id, horse_id, features):
        run = self._get_run_data(race_id, horse_id)
        race = self.get_race_data(race_id)
        
        values = []
        
        for feature in features:
            if feature == "horse_id":
                values.append(horse_id)
                
            elif feature == "race_id":
                values.append(race_id)
                
            elif feature == "won":
                values.append(run["won"].item())
            
            else:
                raise Exception(f"Unknown feature {feature}")
                
        return values
    
    def _get_race_date(self, race_id):
        race = self.get_race_data(race_id)
        
        date = race['date'].item()
        date = datetime.strptime(date, '%Y-%m-%d').date()
        
        return date
    
    def _get_delta_race_date(self, race_id, compare_time):
        date = self._get_race_date(race_id)
        date_delta = (compare_time - date).days
        
        return date_delta
    
    def _get_race_sequence_properties(self, race_id, compare_time, horse_index, race_index,
                                      other_horse_index):                
        values = []
        
        for feature in self.race_sequence_features:
            if feature == "time":
                values.append(self._get_delta_race_date(race_id, compare_time))
            elif feature == "horse_index":
                values.append(horse_index)
            elif feature == "race_index":
                values.append(race_index)
            elif feature == "other_horse_index":
                values.append(other_horse_index)
            else:
                raise Exception(f"Unknown feature {feature}")
                
        return values
    
    def _get_past_run_run(self, race_id, horse_id, compare_time, horse_index, race_index, 
                          other_horse_index):
        run_features = self._get_run_features(race_id, horse_id, self.past_run_features)
        sequence_properties = self._get_race_sequence_properties(race_id, compare_time, 
                                                                 horse_index, race_index, 
                                                                 other_horse_index)
        
        return run_features, sequence_properties
    
    def _get_current_run(self, race_id, horse_id, compare_time, horse_index):
        run_features = self._get_run_features(race_id, horse_id, self.current_run_features)
        sequence_properties = self._get_race_sequence_properties(race_id, compare_time, 
                                                                 horse_index, 0, 0)

        return run_features, sequence_properties
        
    def _get_past_run_runs(self, race_id, horse_id, compare_time, horse_index, race_index):
        runs = []
        
        if self._only_running_horse_past:
            horse_ids = [horse_id]
        else:
            horse_ids = self._get_race_horse_ids(race_id)
            
            # make sure that the important horse of this race is always at position 0
            horse_ids.insert(0, horse_ids.pop(horse_ids.index(horse_id)))
                        
        for other_horse_index, other_horse_id in enumerate(horse_ids):
            if self._debug:
                print(f"Horse {horse_id} (id), {horse_index} (idx) - "
                      f"Past Race {race_index} (idx), {race_id} (id) -" 
                      f"Past Race Horse {other_horse_id} (id), {other_horse_index} (idx)")
                
            runs.append(self._get_past_run_run(race_id, other_horse_id, compare_time, horse_index, 
                                               race_index, other_horse_index))
        
        return runs
    
    def _get_horse_past_runs(self, race_id, horse_id, compare_time, horse_index):
        past_race_ids = self.past_race_ids[race_id][horse_id]
        past_runs = []
                
        for race_index, past_race_id in enumerate(past_race_ids):
            if self._debug:
                print(f"Race ID {race_id} - Horse {horse_id} (id), {horse_index} (idx) "
                      f"- Past Race {race_index + 1} (idx), "
                      f"{past_race_id} (id)")
                
            past_runs += self._get_past_run_runs(past_race_id, horse_id, compare_time, 
                                                 horse_index, race_index + 1)
            
        return past_runs
    
    def _get_race_horse_ids(self, race_id):
        runs = self._get_race_runs_data(race_id)
        
        return list(runs['horse_id'].values)
    
    def _get_race_outcome(self, race_id, horse_id):
        outcome = self._get_run_features(race_id, horse_id, self.race_outcome_features)
        
        return outcome
    
    def _get_past_runs(self, race_id, horse_ids, date):
        if self._debug:
            print(f"Race ID {race_id} past runs")
        
        past_runs = []
        
        for horse_index, horse_id in enumerate(horse_ids):
            if self._debug:
                print(f"Race ID {race_id} - Horse {horse_id} (id), {horse_index} (idx)")
                
            past_runs += self._get_horse_past_runs(race_id, horse_id, date, horse_index)
            
        return past_runs
    
    def _get_current_race(self, race_id, horse_ids, date):
        if self._debug:
            print(f"Race ID {race_id} current runs")
                    
        current_runs = []
        outcome = []
        
        for horse_index, horse_id in enumerate(horse_ids):
            if self._debug:
                print(f"Race ID {race_id} - Horse {horse_id} (id), {horse_index} (idx)")
                
            current_runs.append(self._get_current_run(race_id, horse_id, date, horse_index))
            outcome.append(self._get_race_outcome(race_id, horse_id))
            
        return current_runs, outcome
        
    
    def __call__(self, race_id):
        if self._debug:
            print(f"Race ID {race_id}")
            
        date = self._get_race_date(race_id)
        horse_ids = self._get_race_horse_ids(race_id)
            
        past_runs = self._get_past_runs(race_id, horse_ids, date)
        current_runs, outcome = self._get_current_race(race_id, horse_ids, date)
        
        return current_runs, past_runs, outcome
