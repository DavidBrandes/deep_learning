import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._loss = nn.CrossEntropyLoss()
        
    def forward(self, src, trg):
        # trg: shape (batch, current_race_runs, trg_features); The target for each race runs to predict
        # mask: shape (batch, current_race_runs); The mask to the above runs, True where we have data
        trg, mask = trg
        
        # we only keep the index of the (first, in the case of multiple) winners of the race
        trg = (trg == 1).argmax(axis=1).type(int)
        
        loss = self._loss(src, trg)
        n_items = src.shape[0]
        unreduced_loss = loss * n_items
        
        return loss, unreduced_loss, n_items


class RaceModel(nn.Module):
    def __init__(self):
        super().__init__()
                
    def forward(self, inpt):
        # current_src: shape (batch, current_race_runs, current_race_features); the race runs to predict
        # current_seq: shape (batch, current_race_runs, seq_features); the seq to the above race runs
        # current_mask: shape (batch, current_race_runs); the mask to the above race runs, True where we actaully have data
        # past_src shape: (batch, past_race_runs, past_race_features); the past race runs
        # past_seq shape: (batch, past_race_runs, seq_features); the features to the above runs
        # past_mask shape: (batch, past_race_runs); the mask to the above runs, True where we have data
        
        current_src, current_seq, current_mask, past_src, past_seq, past_mask = inpt
        
        pass
        