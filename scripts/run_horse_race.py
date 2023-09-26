from dl.data.horse_race import HKRaceData
from dl.model.transformer.horse_race import RaceModel, CrossEntropyLoss
from dl.optimization.model import ModelOptimizer
from dl.dataloader.horse_race import RaceDataset
from dl.evaluation.horse_race import evaluate

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch

torch.manual_seed(2)


data_subsample = slice(0, 100)
horse_past_races = 1
only_running_horse_past = False
allow_past_race_correspondence = True
past_run_features = ["venue", "lengths_behind", "race_class", "horse_rating"]
current_run_features = ["won", "win_odds", "prize", "finish_time"]
race_outcome_features = ["won"]
race_sequence_features = ["days_past", "horse_index", "race_index", "other_race_horse_index"]
split = (0.8, 0.1, 0.1)
batch_size = 3
optimizer_kwargs = {"lr": 0.1}
epochs = 5

model = RaceModel()
criterion = CrossEntropyLoss()
optimizer = optim.Adam
writer = SummaryWriter("/Users/david/Downloads/hk_horse_race_data/logs")

data = HKRaceData("/Users/david/Downloads/hk_horse_race_data/runs.csv", 
                  "/Users/david/Downloads/hk_horse_race_data/races.csv", horse_past_races,
                  current_run_features, past_run_features, race_outcome_features,
                  race_sequence_features, subsample=data_subsample, 
                  only_running_horse_past=only_running_horse_past,
                  allow_past_race_correspondence=allow_past_race_correspondence)
train_loader, val_loader, test_loader = RaceDataset.get_dataloaders(data, split, batch_size)

model_optimizer = ModelOptimizer(model, train_loader, val_loader, criterion, optimizer, 
                                 writer=writer, optimizer_kwargs=optimizer_kwargs, epochs=epochs)
model_optimizer()

eval_data = model_optimizer.evaluate(test_loader)
evaluate(eval_data, race_outcome_features)
