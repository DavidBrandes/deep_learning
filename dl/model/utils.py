import torch



def get_model_info(model, input_shape=(1, 3, 256, 256)):    
    for name, layer in model.named_children():
        def wrapper(layer_name):
            def hook(module, input, output):
                print(f"{layer_name}: Input {input[0].shape} -> Output {output[0].shape}")
                
            return hook
            
        layer.register_forward_hook(wrapper(name))
        
    x = torch.randn(input_shape)
    
    model(x)
     
     
    
    