from dl.data.horse_race import HKRaceData
from dl.model.transformer.horse_race import RaceModel, CrossEntropyLoss
from dl.optimization.model import ModelOptimizer
from dl.dataloader.horse_race import RaceDataset

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch

torch.manual_seed(2)


data_subsample = slice(0, 200)
history_length = 1
only_running_horse_past = True
past_run_features = ["race_id", "horse_id", "won"]
current_run_features = ["race_id", "horse_id", "won"]
race_outcome_features = ["won"]
race_sequence_features = ["time", "horse_index", "race_index", "other_horse_index"]
eval_every_nth_node = 5
batch_size = 3

model = RaceModel()
criterion = CrossEntropyLoss()
optimizer = optim.Adam
writer = SummaryWriter("/Users/david/Downloads/logs")

data = HKRaceData("/Users/david/Downloads/hk_horse_race_data/runs.csv", 
                  "/Users/david/Downloads/hk_horse_race_data/races.csv", history_length,
                  current_run_features, past_run_features, race_outcome_features,
                  race_sequence_features, subsample=data_subsample, 
                  only_running_horse_past=only_running_horse_past)
train_loader, eval_loader = RaceDataset.get_dataloaders(data, eval_every_nth_node, batch_size)

model_optimizer = ModelOptimizer(model, train_loader, eval_loader, criterion, optimizer, writer=writer)
quit()
model_optimizer()
