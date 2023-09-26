import numpy as np


def get_split_args(splits, length):
    """Get arguments partitioning an array.
    
    Get disjoint boolean arrays of given length having tre values according to their fraction
    and as spread out as possible.

    Parameters
    ----------
    splits : tuple
        The fractions according to which length should be split.
    length : int
        The total length to be split.

    Raises
    ------
    ValueError
        splits must sum up to one.

    Returns
    -------
    tuple
        len(splits) disjoint boolean lists of specified length.

    """
    if np.sum(splits) != 1:
        raise ValueError("Splits must sum up to one")
        
    n_splits = len(splits)
    
    arg = np.argsort(splits)
    fraction = 1 / np.array(splits)[arg]
    
    args = [[] for _ in range(n_splits)]
    next_index = [0 for _ in range(n_splits)]
    
    for index in range(length):
        appended = False
        
        for split_index in range(n_splits):
            arg_index = arg[split_index]
            
            if appended:
                args[arg_index].append(False)
                
            else:
                if index >= next_index[split_index]:
                    args[arg_index].append(True)
                    next_index[split_index] += fraction[split_index]
                    appended = True
                    
                elif split_index == n_splits - 1:
                    args[arg_index].append(True)
                    next_index[split_index] = index + fraction[split_index]
                    appended = True
                
                else:
                    args[arg_index].append(False)
                                
    return tuple(args)
