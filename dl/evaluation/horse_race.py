import torch


def evaluate(data, features):
    n_races, n_horses = 0, 0
    
    outcome = {}
    if "won" in features:
        outcome["won"] = (0, 0)
    
    for outp, trg in data:
        target, padding_mask = trg
        
        n_races += target.shape[0]
        n_horses += torch.sum(padding_mask).item()
                
        outp = torch.sigmoid(outp)
                
        for feature_index, feature in enumerate(features):
            if feature == "won":
                arg_target = torch.argmax(target[..., feature_index], dim=-1)
                arg_outp = torch.argmax(outp[..., feature_index], dim=-1)
                                
                correct, prob = outcome["won"]
                correct += torch.sum(arg_outp == arg_target).item()
                prob += torch.sum(outp[torch.arange(outp.shape[0]), arg_target, feature_index]).item()
                outcome["won"] = (correct, prob)
             
    if len(outcome):
        print(f"From {n_races} races with {(n_horses / n_races):.2f} horses on average:")
    if "won" in outcome:
        correct, prob = outcome["won"]
        print(f"  Predicted {correct} wins with average precision of {(prob / n_races):.3f}")