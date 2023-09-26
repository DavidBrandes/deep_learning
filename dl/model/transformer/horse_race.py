import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._loss = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, src, trg):
        # target: shape (batch, current_race_runs, trg_features); The target for each race runs to predict
        # padding_mask: shape (batch, current_race_runs); The mask to the above runs, True where we have data
        target, padding_mask = trg
        n_items = src.shape[0]
        
        loss = self._loss(src, target)
        loss = loss[padding_mask]
        
        unreduced_loss = torch.sum(loss)
        loss = torch.mean(loss)
        
        return loss, unreduced_loss, n_items
    
    
class BasicRaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        d_model = 4
        nhead = 2
        feedforward_expansion = 4
        num_encoder_layers = 1
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * feedforward_expansion, 
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
    def forward(self, inpt):
        current_src, current_seq, current_padding_mask, past_src, past_seq, past_mask, past_padding_mask = inpt
        
        x = current_src
        x = self.encoder(x, src_key_padding_mask=current_padding_mask)
        
        return x

class RaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        d_model = 4
        nhead = 2
        
        # d_model % nhead == 0 required
        
        self.nhead = nhead
        
        self.transformer = nn.Transformer(d_model, nhead, 1, 1, 2 * d_model,
                                          dropout=0, batch_first=True)
        self.linear = nn.Linear(d_model, 1)
        
    def _expand_mask(self, mask):
        # repeat the mask by nhead times to fit the desired input
        bsz, tgt_len, src_len = mask.shape
        
        mask = mask.unsqueeze(1)
        mask = mask.expand(bsz, self.nhead, tgt_len, src_len)
        mask = mask.reshape(bsz * self.nhead, tgt_len, src_len)
        
        return mask
        
                
    def forward(self, inpt):
        # current_src: shape (batch, current_race_runs, current_race_features); the race runs to predict
        # current_seq: shape (batch, current_race_runs, seq_features); the seq to the above race runs
        # current_padding_mask: shape (batch, current_race_runs); the mask to the above race runs, True where we actaully have data
        # past_src shape: (batch, past_race_runs, past_race_features); the past race runs
        # past_seq shape: (batch, past_race_runs, seq_features); the features to the above runs
        # past_mask shape: (batch, past_race_runs, past_race_runs); The mask that assures past races may only attend themselves
        # past_padding_mask shape: (batch, past_race_runs); the mask to the above runs, True where we have data
        
        current_src, current_seq, current_padding_mask, past_src, past_seq, past_mask, past_padding_mask = inpt
        past_mask = self._expand_mask(past_mask)
                
        x = self.transformer(past_src, current_src,
                             src_mask=~past_mask,
                             src_key_padding_mask=~past_padding_mask,
                             tgt_key_padding_mask=~current_padding_mask)
        x = self.linear(x)
        
        return x
        
        pass
        