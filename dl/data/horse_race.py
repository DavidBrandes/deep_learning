import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


VALID_RUN_FEATURES = ["win_odds", "finish_time", "lengths_behind", "draw", "declared_weight",
                      "actual_weight", "horse_age", "horse_rating", "place_odds", "won", "place",
                      "result"]
VALID_RACE_FEATURES = ["distance", "race_class", "prize", "race_no", "surface", "venue", "going"]
VALID_OTHER_FEATURES = ["race_id", "horse_id", "past_wins", "past_place", "past_races"]
VALID_SEQUENCE_FEATURES = ["days_past", "horse_index", "race_index", "other_race_horse_index", 
                           "days_past_unnormalized", "race_index_unnormalized"]


# TODO:
#   add in trainer and jockey (number of past races, wins)
#   if two horses appear in the same past race, identify the
        

class HKRaceData:
    def __init__(self, runs_data_path, races_data_path, horse_past_races,
                 current_run_features, past_run_features, race_outcome_features, 
                 race_sequence_features, normalize=True, subsample=None, max_horses_per_race=None, 
                 min_horse_past_races=0, min_past_races=1, only_running_horse_past=False, 
                 only_distinct_outcomes=True, only_proper_place_races=True,
                 allow_past_race_correspondence=True, max_race_date_difference=float("inf"), debug=False):
        self._runs_data = pd.read_csv(runs_data_path)
        self._races_data = pd.read_csv(races_data_path)
        
        self._only_running_horse_past = only_running_horse_past
        self._normalize = normalize
        self._max_horses_per_race = max_horses_per_race
        self._only_distinct_outcomes = only_distinct_outcomes
        self._subsample = subsample
        self._min_horse_past_races = min_horse_past_races
        self._min_past_races = min_past_races
        self._horse_past_races = horse_past_races
        self._allow_past_race_correspondence = allow_past_race_correspondence
        self._only_proper_place_races = only_proper_place_races
        self._max_race_date_difference = max_race_date_difference
        
        self._debug = debug
        
        self.past_run_features = past_run_features
        self.current_run_features = current_run_features
        self.race_outcome_features = race_outcome_features
        self.race_sequence_features = race_sequence_features
        
        self._fix_data()
        self._set_race_ids()
        if self._normalize:
            self._normalize_data()
        
    def _check(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if np.any([res != res for res in result]):
                raise Exception(f"Found false value in {result}")
            
            return result
        
        return wrapper
    
    def _fix_data(self):
        # length behind
        lengths_behind = self._runs_data["lengths_behind"]
        lengths_behind = lengths_behind.mask(lengths_behind < 0, 0)
        self._runs_data["lengths_behind"] = lengths_behind
        
        # place odds
        win_odds = self._runs_data["win_odds"].to_numpy()
        place_odds = self._runs_data["place_odds"].to_numpy()
        
        arg = np.isfinite(place_odds)
        frac = np.mean(win_odds[arg] / place_odds[arg])
        place_odds[~arg] =  win_odds[~arg] / frac
        
        self._runs_data["place_odds"] = place_odds
        
        # prize
        prize = self._races_data["prize"]
        prize = prize.fillna(prize.mean())
        self._races_data["prize"] = prize
        
        # venue
        venue = self._races_data["venue"]
        venue = venue.replace("ST", 0)
        venue = venue.replace("HV", 1)
        self._races_data["venue"] = venue
        
        # going
        going = self._races_data["going"]
        
        going = going.replace("GOOD TO FIRM", 1)
        going = going.replace("GOOD", 2)
        going = going.replace("GOOD TO YIELDING", 3)
        going = going.replace('YIELDING', 4)
        going = going.replace('YIELDING TO SOFT', 5)
        going = going.replace("SOFT", 6)
        
        going = going.replace("FAST", 1)
        going = going.replace("WET FAST", 2.66)
        going = going.replace("WET SLOW", 4.33)
        going = going.replace("SLOW", 6)
        
        self._races_data["going"] = going
        
        # place
        place = (self._runs_data["result"].to_numpy() <= 3).astype(int)
        self._runs_data["place"] = place
        
    def _normalize_data(self):        
        for feature in VALID_RACE_FEATURES:
            value = self._races_data[feature]
            self._races_data[feature] = (value - value.min()) / (value.max() - value.min())

        for feature in VALID_RUN_FEATURES:
            value = self._runs_data[feature]
            self._runs_data[feature] = (value - value.min()) / (value.max() - value.min())
            
    def _get_valid_race_ids(self):
        race_ids = self._races_data['race_id'].values
        dates = self._races_data['date'].values
        
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        arg = sorted(range(len(dates)), key=dates.__getitem__)
        
        race_ids = np.array(race_ids)[arg]
        
        if self._subsample:
            race_ids = race_ids[self._subsample]
            
        valid_race_ids = []
        
        max_horses_per_race = 0
        
        if self._only_proper_place_races:
            data = self._races_data[self._races_data["place_combination3"].isna()]
            invalid_race_ids = data["race_id"].to_list()
        else:
            invalid_race_ids = []
        
        for race_id in race_ids:
            if self._only_distinct_outcomes:
                runs = self._get_race_runs_data(race_id)
                n_runs = len(runs.index)
                results = runs["result"].to_numpy()
                
                if np.sum(results == 1) != 1 or np.sum(results == 2) != 1 or np.sum(results == 3) != 1:
                    continue
                
            if race_id in invalid_race_ids:
                continue
            
            if self._max_horses_per_race and n_runs > self._max_horses_per_race:
                continue
            
            if n_runs > max_horses_per_race:
                max_horses_per_race = n_runs
            
            valid_race_ids.append(race_id)
            
        return valid_race_ids, max_horses_per_race
    
    def _set_race_ids(self):
        race_ids, horses_per_race = self._get_valid_race_ids()
        
        self.horses_per_race = horses_per_race
                
        past_race_ids = {}
        horse_races = defaultdict(list)
        
        race_horse_features = {}
        horse_wins = defaultdict(lambda: 0)
        horse_place = defaultdict(lambda: 0)
        
        max_date_difference = 1
        max_horse_past_races = 1
        
        for race_id in race_ids:
            runs_data = self._get_race_runs_data(race_id)
            
            race_horse_ids = runs_data['horse_id'].values
            race_date = self._get_race_date(race_id)
            
            value = {}
            
            race_features = {}
            
            for race_horse_id in race_horse_ids:
                # append features
                race_features[race_horse_id] = {
                    'past_wins': horse_wins[race_horse_id],
                    'past_place': horse_place[race_horse_id],
                    'past_races': len(horse_races[race_horse_id]),
                }
                
                # update old features
                race_horse_data = runs_data[runs_data['horse_id'] == race_horse_id]
                
                if race_horse_data["result"].item() <= 1:
                    horse_wins[race_horse_id] += 1
                 
                if race_horse_data["result"].item() <= 3:
                    horse_place[race_horse_id] += 1
                                
                race_horse_past_races = []
                
                max_race_past_difference = 0
                
                for past_race_id in horse_races[race_horse_id][::-1]:
                    date_difference = self._get_delta_race_date(past_race_id, race_date)
                    if len(race_horse_past_races) >= self._horse_past_races or \
                        date_difference > self._max_race_date_difference:
                           
                        break
                    
                    if date_difference > max_race_past_difference:
                        max_race_past_difference = date_difference
                
                    race_horse_past_races.append(past_race_id)
                
                value[race_horse_id] = race_horse_past_races
                
                horse_races[race_horse_id].append(race_id)
                
            race_horse_features[race_id] = race_features
                
            if sum([len(v) for v in value.values()]) >= self._min_past_races and \
                np.all([len(v) >= self._min_horse_past_races for v in value.values()]):
                    
                past_race_ids[race_id] = value
                
                max_race_horse_past_races = max([len(v) for v in value.values()])
                if max_race_horse_past_races > max_horse_past_races:
                    max_horse_past_races = max_race_horse_past_races
                
                if max_race_past_difference > max_date_difference:
                    max_date_difference = max_race_past_difference
                    
        # TODO normalize past wins
                
        self._past_race_ids = past_race_ids
        self._race_horse_features = race_horse_features
        self.race_ids = list(past_race_ids.keys())
        
        self._max_date_difference = max_date_difference
        self._max_horse_past_races = max_horse_past_races
        self._max_past_wins = max(horse_wins.values())
        self._max_past_place = max(horse_place.values())
        self._max_past_races = max([len(races) for races in horse_races.values()])
    
    def _get_run_data(self, race_id, horse_id):
        runs = self._get_race_runs_data(race_id)
        
        run = runs[runs['horse_id'] == horse_id]
        
        return run
    
    def _get_race_data(self, race_id):
        race = self._races_data[self._races_data["race_id"] == race_id]
        
        return race
    
    def _get_race_runs_data(self, race_id):
        runs = self._runs_data[self._runs_data["race_id"] == race_id]
        
        return runs
    
    @_check 
    def _get_run_features(self, race_id, horse_id, features):
        run = self._get_run_data(race_id, horse_id)
        race = self._get_race_data(race_id)
        
        values = []
        
        for feature in features:
            if feature in VALID_RACE_FEATURES:
                values.append(race[feature].item())
                
            elif feature in VALID_RUN_FEATURES:
                values.append(run[feature].item())
                
            elif feature in VALID_OTHER_FEATURES:
                if feature == "horse_id":
                    values.append(horse_id)
                    
                elif feature == "race_id":
                    values.append(race_id)
                    
                elif feature == "past_wins":
                    past_wins = self._race_horse_features[race_id][horse_id]['past_wins']
                    if self._normalize:
                        past_wins = past_wins / self._max_past_wins
                    values.append(past_wins)
                    
                elif feature == "past_races":
                    past_races = self._race_horse_features[race_id][horse_id]['past_races']
                    if self._normalize:
                        past_races = past_races / self._max_past_races
                    values.append(past_races)
                    
                elif feature == "past_place":
                    past_place = self._race_horse_features[race_id][horse_id]['past_place']
                    if self._normalize:
                        past_place = past_place / self._max_past_place
                    values.append(past_place)
                    
            else:
                raise Exception(f"Unknown feature {feature}")
                
        return values
    
    def _get_race_date(self, race_id):
        race = self._get_race_data(race_id)
        
        date = race['date'].item()
        date = datetime.strptime(date, '%Y-%m-%d').date()
        
        return date
    
    def _get_delta_race_date(self, race_id, compare_time):
        date = self._get_race_date(race_id)
        date_delta = (compare_time - date).days
        
        return date_delta
    
    def _get_race_sequence_properties(self, other_race_id, other_race_horse_index, 
                                      base_race_index, base_race_date, base_horse_index):                
        values = []
                
        for feature in self.race_sequence_features:
            if feature not in VALID_SEQUENCE_FEATURES:
                raise Exception(f"Unknown feature {feature}")
                
            if feature == "days_past":
                days_past = self._get_delta_race_date(other_race_id, base_race_date)
                if self._normalize:
                    days_past = days_past / self._max_date_difference
                values.append(days_past)
                
            elif feature == "days_past_unnormalized":
                values.append(self._get_delta_race_date(other_race_id, base_race_date))
                
            elif feature == "horse_index":
                values.append(base_horse_index)
                
            elif feature == "race_index":
                index = base_race_index
                if self._normalize:
                    index = index / self._max_horse_past_races
                values.append(index)
                
            elif feature == "race_index_unnormalized":
                values.append(base_race_index)
                
            elif feature == "other_race_horse_index":
                values.append(other_race_horse_index)
                
        return values
        
    def _get_past_run_run(self, past_race_id, past_race_horse_id, past_race_horse_index, 
                          race_index, race_date, horse_index):
        run_features = self._get_run_features(past_race_id, past_race_horse_id, 
                                              self.past_run_features)
        sequence_properties = self._get_race_sequence_properties(past_race_id, past_race_horse_index, 
                                                                 race_index, race_date, horse_index)
        if self._allow_past_race_correspondence:
            label = (race_index, horse_index)
        else:
            label = (race_index, horse_index, past_race_horse_index)
        
        return run_features, sequence_properties, label
    
    def _get_current_run(self, race_id, race_date, horse_id, horse_index):
        run_features = self._get_run_features(race_id, horse_id, self.current_run_features)
        sequence_properties = self._get_race_sequence_properties(race_id, 0, 0, 
                                                                 race_date, horse_index)

        return run_features, sequence_properties
        
    def _get_past_run_runs(self, past_race_id, past_race_horse_ids, race_index, race_date, 
                           horse_index):
        runs = []
                        
        for past_race_horse_index, past_race_horse_id in enumerate(past_race_horse_ids):
            if self._debug:
                print(f"Horse {horse_index} (idx) - "
                      f"Past Race {race_index} (idx), {past_race_id} (id) -" 
                      f"Past Race Horse {past_race_horse_id} (id), {past_race_horse_index} (idx)")
                
            runs.append(self._get_past_run_run(past_race_id, past_race_horse_id, past_race_horse_index, 
                                               race_index, race_date, horse_index))

        return runs
    
    def _get_horse_past_runs(self, race_id, race_date, horse_id, horse_index):
        past_race_ids = self._past_race_ids[race_id][horse_id]
        past_runs = []
                
        for race_index, past_race_id in enumerate(past_race_ids):
            if self._debug:
                print(f"Race ID {race_id} - Horse {horse_id} (id), {horse_index} (idx) "
                      f"- Past Race {race_index + 1} (idx), "
                      f"{past_race_id} (id)")
                
            past_race_horse_ids = self._get_race_runs_data(past_race_id)['horse_id'].values
            
            if self._only_running_horse_past:
                past_race_horse_ids = [horse_id]
            else:
                past_race_horse_ids = list(past_race_horse_ids)
            
                # make sure that the important horse of this race is always at position 0
                past_race_horse_ids.insert(0, past_race_horse_ids.pop(past_race_horse_ids.index(horse_id)))
                
            past_runs += self._get_past_run_runs(past_race_id, past_race_horse_ids, 
                                                 race_index + 1, race_date, horse_index)
            
        return past_runs
    
    def _get_race_horse_ids(self, race_id):
        runs = self._get_race_runs_data(race_id)
        
        return list(runs['horse_id'].values)
    
    def _get_race_outcome(self, race_id, horse_id):
        outcome = self._get_run_features(race_id, horse_id, self.race_outcome_features)
        
        return outcome
    
    def _get_past_runs(self, race_id, horse_ids, race_date):
        if self._debug:
            print(f"Race ID {race_id} past runs")
        
        past_runs = []
        
        for horse_index, horse_id in enumerate(horse_ids):
            if self._debug:
                print(f"Race ID {race_id} - Horse {horse_id} (id), {horse_index} (idx)")
                
            past_runs += self._get_horse_past_runs(race_id, race_date, horse_id, horse_index)
            
        return past_runs
    
    def _get_current_race(self, race_id, horse_ids, race_date):
        if self._debug:
            print(f"Race ID {race_id} current runs")
                    
        current_runs = []
        outcome = []
        
        for horse_index, horse_id in enumerate(horse_ids):
            if self._debug:
                print(f"Race ID {race_id} - Horse {horse_id} (id), {horse_index} (idx)")
                
            current_runs.append(self._get_current_run(race_id, race_date, horse_id, horse_index))
            outcome.append(self._get_race_outcome(race_id, horse_id))
            
        return current_runs, outcome
        
    
    def __call__(self, race_id):
        if self._debug:
            print(f"Race ID {race_id}")
            
        race_date = self._get_race_date(race_id)
        horse_ids = self._get_race_horse_ids(race_id)
            
        past_runs = self._get_past_runs(race_id, horse_ids, race_date)
        current_runs, outcome = self._get_current_race(race_id, horse_ids, race_date)
        
        return current_runs, past_runs, outcome
