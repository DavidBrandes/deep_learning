from dl.data.horse_race import HKRaceData
from dl.model.transformer.horse_race import RaceModel, CrossEntropyLoss
from dl.optimization.model import ModelOptimizer
from dl.dataloader.horse_race import RaceDataset
from dl.evaluation.horse_race import evaluate

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch

torch.manual_seed(2)

# PYTHONPATH="/Users/david/Documents/code/deep_learning:$PYTHONPATH"
# export PYTHONPATH


horse_past_races = 5
past_run_features = ["lengths_behind", "finish_time", "draw", "horse_rating", 
                     "past_wins", "past_place", "past_races", "distance"]
current_run_features = ["draw", "horse_rating", "past_wins", "past_place", "past_races",  "distance"]
race_outcome_features = ["won"]
race_sequence_features = ["horse_index", "days_past", "race_index"]

subsample = slice(0, 250)

split = (0.8, 0.1, 0.1)
batch_size = 16
optimizer_kwargs = {"lr": 0.1}
epochs = 5

model = RaceModel()
criterion = CrossEntropyLoss()
optimizer = optim.Adam
writer = SummaryWriter("/Users/david/Downloads/hk_horse_race_data/logs")

data = HKRaceData("/Users/david/Downloads/hk_horse_race_data/runs.csv", 
                  "/Users/david/Downloads/hk_horse_race_data/races.csv", horse_past_races,
                  current_run_features, past_run_features, race_outcome_features,
                  race_sequence_features, subsample=subsample)

train_loader, val_loader, test_loader = RaceDataset.get_dataloaders(data, split, batch_size)

model_optimizer = ModelOptimizer(criterion, optimizer, writer=writer, 
                                 optimizer_kwargs=optimizer_kwargs, epochs=epochs)
model_optimizer(model, train_loader, val_loader)

eval_data = model_optimizer.evaluate(val_loader)
evaluate(eval_data, race_outcome_features)
