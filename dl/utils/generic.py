

def convert_to_shape(shape):
    if type(shape) is int:
        shape = (shape, shape)
    elif type(shape) is tuple and len(shape) == 1:
        shape = shape + shape
    
    return shape