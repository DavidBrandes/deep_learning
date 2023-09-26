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
    
    
class SequenceEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.horses_per_race = 14
        
    def forward(self, seq):
        arg = seq[..., 0] != 0

        x = torch.zeros(seq.shape[:2] + (self.horses_per_race,), dtype=seq.dtype)
        x[arg, seq[arg][..., 0].type(torch.long)] = 1
        
        x = torch.cat([x, seq[..., 1:]], dim=-1)
        
        return x
    

class FeatureEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sequence_embedding = SequenceEmbedding()
        
    def forward(self, src, seq):
        seq = self.sequence_embedding(seq)
        print(seq)
        x = torch.cat([src, seq], dim=-1)
        
        return x
    

class RaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        d_current = 6
        d_past = 8
        d_target = 1
        d_sequence = 14 + 2
        d_model = d_past + d_sequence
        nhead = 4
        
        num_encoder_layers = 2
        num_decoder_layers = 4
        
        # d_model % nhead == 0 required
        
        self.nhead = nhead
        
        self.feature_embedding = FeatureEmbedding()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, 
                                          4 * d_model, batch_first=True)
        self.input_linear = nn.Linear(d_current + d_sequence, d_model)
        self.output_linear = nn.Linear(d_model, d_target)
        
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
        
        current = self.feature_embedding(current_src, current_seq)
        current = self.input_linear(current)
        past = self.feature_embedding(past_src, past_seq)
                
        x = self.transformer(past, current,
                             src_mask=~past_mask,
                             src_key_padding_mask=~past_padding_mask,
                             tgt_key_padding_mask=~current_padding_mask)
        x = self.output_linear(x)
        
        return x