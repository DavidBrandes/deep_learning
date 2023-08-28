import torch

from dl.optimization.context import RunContext


class Optimizer:
    def __init__(self, model, optimizer, optimizer_kwargs={}, parameterization=None,
                 transformation=None, normalization=None, epochs=250, callback=None, 
                 device="cpu", leave=True):
        self._model = model.to(device)
        self._optimizer = optimizer

        self._parameterization = parameterization
        self._transformation = transformation
        self._normalization = normalization

        self._optimizer_kwargs = optimizer_kwargs
        self._device = device
        self._epochs = epochs
        self._callback = callback
        
        self._leave = leave

    def _detach(self, x):
        x = x.detach().clone().cpu()
        if self._parameterization:
            x = self._parameterization(x)

        return x

    def __call__(self, x):
        if self._normalization:
            x = self._normalization.parameterize(x)
        if self._parameterization:
            x = self._parameterization.parameterize(x)

        x = x.clone().to(self._device)

        optimizer = self._optimizer([x.requires_grad_()], **self._optimizer_kwargs)

        best_x = self._detach(x)
        best_loss = last_loss = float("inf")
        epoch = 0

        with RunContext():
            print("Optimizing")
            while epoch < self._epochs:
                epoch += 1
                
                optimizer.zero_grad()
                
                x_ = x
                if self._parameterization:
                    x_ = self._parameterization(x_)
                if self._transformation:
                    x_ = self._transformation(x_)
                if self._normalization:
                    x_ = self._normalization(x_)
                
                loss = self._model(x_)
                
                with torch.no_grad():
                    this_loss = loss.item()
                    step_size = last_loss - this_loss
                    last_loss = this_loss
                    
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best_x = self._detach(x)

                        if self._callback:
                            self._callback(epoch, this_loss, best_x)

                    print(f"  Epoch ({epoch}/{self._epochs}): ",
                          f"Loss {this_loss:.4e} ({best_loss:.4e})", 
                          f", Step Size {step_size:.4e}",
                          "    " if self._leave else "",
                          end="\r" if self._leave else "\n")

                loss.backward()
                optimizer.step()
        
        if self._leave:
            print("                                                                     ", end="\r")
        print(f"Final Loss {best_loss:.4e}")

        return best_x