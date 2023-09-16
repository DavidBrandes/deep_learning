from torch.utils.data import Dataset, DataLoader
import numpy as np

from dl.utils import tensor as tensor_utils


class RaceDataset(Dataset):
    def __init__(self, data, eval_every_nth_race, train): 
        self._data = data
        self._train = train
                
        race_ids = np.array(data.race_ids)
        arg = np.arange(len(race_ids))
        
        if train:
            arg = arg % eval_every_nth_race != 0
        else:
            arg = arg % eval_every_nth_race == 0
        
        self._race_ids = list(race_ids[arg])
        
    @staticmethod
    def _collate_fn(data_list):
        current_src_list, current_trg_list, current_seq_list, past_src_list, past_seq_list = zip(*data_list)
                
        batch_size = len(current_src_list)
        max_current_shape = max([len(current_src) for current_src in current_src_list])
        max_past_shape = max([len(past_src) for past_src in past_src_list])
        
        current_src = np.zeros((batch_size, max_current_shape, current_src_list[0].shape[1]))
        current_seq = np.zeros((batch_size, max_current_shape, current_seq_list[0].shape[1]))
        current_trg = np.zeros((batch_size, max_current_shape, current_trg_list[0].shape[1]))
        current_mask = np.zeros((batch_size, max_current_shape), dtype=bool)
        
        past_src = np.zeros((batch_size, max_past_shape, past_src_list[0].shape[1]))
        past_seq = np.zeros((batch_size, max_past_shape, past_seq_list[0].shape[1]))
        past_mask = np.zeros((batch_size, max_past_shape), dtype=bool)
        
        for index, (current_src_item, current_trg_item, current_seq_item, 
                    past_src_item, past_seq_item) in enumerate(data_list):
            n_current = current_src_item.shape[0]
            n_past = past_src_item.shape[0]
            
            current_src[index, :n_current, :] = current_src_item
            current_trg[index, :n_current, :] = current_trg_item
            current_seq[index, :n_current, :] = current_seq_item
            current_mask[index, :n_current] = True
            
            past_src[index, :n_past, :] = past_src_item
            past_seq[index, :n_past, :] = past_seq_item
            past_mask[index, :n_past] = True
            
        current_src = tensor_utils.tensor(current_src)
        current_trg = tensor_utils.tensor(current_trg)
        current_seq = tensor_utils.tensor(current_seq)
        current_mask = tensor_utils.tensor(current_mask)
        past_src = tensor_utils.tensor(past_src)
        past_seq = tensor_utils.tensor(past_seq)
        past_mask = tensor_utils.tensor(past_mask)
        
        inpt = (current_src, current_seq, current_mask, past_src, past_seq, past_mask)
        trg = (current_trg, current_mask.clone())
    
        return inpt, trg
            
    def __len__(self):
        return len(self._race_ids)
    
    def __getitem__(self, index):
        current_runs, past_runs, outcome = self._data(self._race_ids[index])
        
        current_src, current_seq = zip(*current_runs)
        current_src, current_seq = np.array(current_src), np.array(current_seq)
        
        if len(past_runs):
            past_src, past_seq = zip(*past_runs)
            past_src, past_seq = np.array(past_src), np.array(past_seq)
            
        else:
            past_src = np.zeros((0, len(self._data.past_run_features)))
            past_seq = np.zeros((0, len(self._data.race_sequence_features)))
            
        current_trg = outcome
        current_trg = np.array(current_trg)
                            
        return current_src, current_trg, current_seq, past_src, past_seq
    
    @classmethod
    def get_dataloaders(cls, data, eval_every_nth_race, batch_size, seed=1001):
        train_loader = DataLoader(cls(data, eval_every_nth_race, True), batch_size,
                                  shuffle=True, collate_fn=cls._collate_fn)
        eval_loader = DataLoader(cls(data, eval_every_nth_race, False), batch_size,
                                 shuffle=False, collate_fn=cls._collate_fn)
        
        return train_loader, eval_loader