import torch
from torch import nn

#update, this is going to be text classification model to tell apart sentences with correct grammar
#and incorrect grammar

#this will be the bert model, which I will code up in the next commit
class TextClassification(nn.Module):
    def __init__(self, input_shape, hidden_layers, output_shape) -> None:
        
        pass


class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        #this represents an individual attention mechanism which can be run in parallel with more 
        #attention mechanisms
        pass


class MultiHeadedAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        pass
        #this does the parallel thing in the self attentino above to make the training process faster
        #cus boy do i aint got a lotta time, only a few hours at night at most. why did i do this to
        #myself

