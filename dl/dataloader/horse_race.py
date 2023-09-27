from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

from dl.utils import tensor as tensor_utils, linalg as linalg_utils, combinatorics as combinatorics_utils


class RaceDataset(Dataset):
    def __init__(self, data, race_id_arg=None): 
        self._data = data
        
        self.race_ids = np.array(data.race_ids)
        if race_id_arg is not None:
            self.race_ids = self.race_ids[race_id_arg]
        
    @staticmethod
    def _collate_fn(data_list):
        current_src_list, current_trg_list, current_seq_list, past_src_list, past_seq_list, past_label_list = zip(*data_list)
                
        batch_size = len(current_src_list)
        max_current_shape = max([len(current_src) for current_src in current_src_list])
        max_past_shape = max([len(past_src) for past_src in past_src_list])
        
        current_src = np.zeros((batch_size, max_current_shape, current_src_list[0].shape[1]))
        current_seq = np.zeros((batch_size, max_current_shape, current_seq_list[0].shape[1]))
        current_trg = np.zeros((batch_size, max_current_shape, current_trg_list[0].shape[1]))
        current_padding_mask = np.zeros((batch_size, max_current_shape), dtype=bool)
        
        past_src = np.zeros((batch_size, max_past_shape, past_src_list[0].shape[1]))
        past_seq = np.zeros((batch_size, max_past_shape, past_seq_list[0].shape[1]))
        past_padding_mask = np.zeros((batch_size, max_past_shape), dtype=bool)
        past_mask = np.ones((batch_size, max_past_shape, max_past_shape), dtype=bool)
        
        for index, (current_src_item, current_trg_item, current_seq_item, 
                    past_src_item, past_seq_item, past_label_list) in enumerate(data_list):
            n_current = current_src_item.shape[0]
            n_past = past_src_item.shape[0]
            
            current_src[index, :n_current, :] = current_src_item
            current_trg[index, :n_current, :] = current_trg_item
            current_seq[index, :n_current, :] = current_seq_item
            current_padding_mask[index, :n_current] = True
            
            past_src[index, :n_past, :] = past_src_item
            past_seq[index, :n_past, :] = past_seq_item
            past_padding_mask[index, :n_past] = True
            past_mask[index, :n_past, :n_past] = linalg_utils.make_block_diag_from_labels(past_label_list)
                        
        current_src = tensor_utils.tensor(current_src)
        current_trg = tensor_utils.tensor(current_trg)
        current_seq = tensor_utils.tensor(current_seq)
        current_padding_mask = tensor_utils.tensor(current_padding_mask, dtype=torch.bool)
        past_src = tensor_utils.tensor(past_src)
        past_seq = tensor_utils.tensor(past_seq)
        past_padding_mask = tensor_utils.tensor(past_padding_mask, dtype=torch.bool)
        past_mask = tensor_utils.tensor(past_mask, dtype=torch.bool)
                
        inpt = (current_src, current_seq, current_padding_mask, past_src, past_seq, past_mask, past_padding_mask)
        trg = (current_trg, current_padding_mask.clone())
    
        return inpt, trg
            
    def __len__(self):
        return len(self.race_ids)
    
    def __getitem__(self, index):
        return self.get_by_index(index)
    
    def get_by_index(self, index):
        race_id = self.race_ids[index]
        
        return self.get_by_race_id(race_id)
        
    def get_by_race_id(self, race_id):
        current_runs, past_runs, outcome = self._data(race_id)
        
        current_src, current_seq = zip(*current_runs)
        current_src, current_seq = np.array(current_src), np.array(current_seq)
        
        if len(past_runs):
            past_src, past_seq, past_label = zip(*past_runs)
            past_src, past_seq = np.array(past_src), np.array(past_seq)
            _, past_label = np.unique(past_label, axis=0, return_inverse=True)
            
        else:
            past_src = np.zeros((0, len(self._data.past_run_features)))
            past_seq = np.zeros((0, len(self._data.race_sequence_features)))
            past_label = np.zeros((0), dtype=int)
            
        current_trg = outcome
        current_trg = np.array(current_trg)
                            
        return current_src, current_trg, current_seq, past_src, past_seq, past_label
    
    @classmethod
    def get_dataloaders(cls, data, splits, batch_size):
        train_arg, val_arg, test_arg = combinatorics_utils.get_split_args(splits, len(data.race_ids))
        
        train_loader = DataLoader(cls(data, train_arg), batch_size,
                                  shuffle=True, collate_fn=cls._collate_fn)
        val_loader = DataLoader(cls(data, val_arg), batch_size,
                                shuffle=False, collate_fn=cls._collate_fn)
        test_loader = DataLoader(cls(data, test_arg), batch_size,
                                 shuffle=False, collate_fn=cls._collate_fn)
        
        return train_loader, val_loader, test_loader