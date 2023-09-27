from copy import deepcopy
from tqdm import tqdm
import torch

from dl.optimization.context import RunContext


class ModelOptimizer:
    def __init__(self, criterion, optimizer, optimizer_kwargs={},
                 epochs=250, callback=None, writer=None, device="cpu"):
        self._optimizer = optimizer
        self._criterion = criterion

        self._optimizer_kwargs = optimizer_kwargs
        self._device = device
        self._epochs = epochs
        self._callback = callback
        self._device = device
        self._writer = writer
        
    def evaluate(self, model, loader):
        result = []
        
        print("Evaluating")
        
        model = model.to(self.device).eval()
        
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            inpt, trg = self._unpack(batch)
            outp = model(inpt)
            
            result.append((outp.to("cpu"), tuple(el.to('cpu') for el in trg)))
            
        model = model.to("cpu").train()
        
        return result
        
    def _detach(self, model):
        return deepcopy(model).to("cpu").eval()
    
    def _unpack(self, x):
        inpt, trg = x
        
        inpt = tuple(el.to(self._device) for el in inpt)
        trg = tuple(el.to(self._device) for el in trg)
        
        return inpt, trg
    
    def __call__(self, model, train_loader, val_loader):
        model = model.to(self._device).train()
        optimizer = self._optimizer(model.parameters(), **self._optimizer_kwargs)
        
        best_model = self._detach(model)
        best_eval_loss = last_eval_loss = float("inf")
        last_train_loss = float("inf")
        epoch = 0
                
        with RunContext():
            print("Optimizing")
            
            while epoch < self._epochs:
                epoch += 1
                print(f"  Epoch {epoch}/{self._epochs}:")
                
                accum_loss, accum_weight = [], []
                
                for index, batch in enumerate(pgbar := tqdm(train_loader, desc="    Training",
                                                            leave=False)):
                    inpt, trg = self._unpack(batch)
                    
                    optimizer.zero_grad()
                    
                    outp = model(inpt)
                    loss, unweighted_loss, weight = self._criterion(outp, trg)
                    loss.backward()
                    
                    optimizer.step()
                        
                    pgbar.set_postfix({"Loss": f"{(unweighted_loss.item()/weight):.4e}"})
                    accum_loss.append(unweighted_loss.item())
                    accum_weight.append(weight)
                    if self._writer:
                        self._writer.add_scalar('Loss/train-individual', loss.item(), 
                                                index + epoch * len(pgbar))
                    
                loss_train = sum(accum_loss) / sum(accum_weight)
                diff = last_train_loss - loss_train
                print(f"    Train Loss {loss_train:.4e}, Step Size {diff:.4e}")
                last_train_loss = loss_train
                
                model.eval()
                
                with torch.no_grad():
                    accum_loss, accum_weight = [], []
                    
                    for index, batch in enumerate(pgbar := tqdm(val_loader, desc="    Validating",
                                                                leave=False)):
                        inpt, trg = self._unpack(batch)
                        
                        outp = model(inpt)
                        loss, unweighted_loss, weight = self._criterion(outp, trg)
                        
                        pgbar.set_postfix({"Loss": f"{(unweighted_loss.item()/weight):.4e}"})
                        accum_loss.append(unweighted_loss.item())
                        accum_weight.append(weight)
                        if self._writer:
                            self._writer.add_scalar('Loss/eval-individual', loss.item(), 
                                                    index + epoch * len(pgbar))
                        
                    loss_eval = sum(accum_loss) / sum(accum_weight)
                    diff = last_eval_loss - loss_eval
                    print(f"    Val Loss {loss_eval:.4e}, Step Size {diff:.4e}")
                    last_eval_loss = loss_eval
                    if self._writer:
                        self._writer.add_scalars('Training', 
                                                 {'train': loss_train, 'val': loss_eval}, epoch)
                    
                    if loss < best_eval_loss:
                        best_eval_loss = loss
                        best_model = self._detach(model)
                        
                        if self._callback:
                            self._callback(epoch, loss, best_model)
                            
                model.train()
                            
        print(f"Final Loss {last_eval_loss:.4e}")

        return best_model
                    
                