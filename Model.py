import torch
from torch import nn

from typing import Dict

#update, this is going to be text classification model to tell apart sentences with correct grammar
#and incorrect grammar

#this will be the bert model, which I will code up in the next commit, or at least part of it.
#goal is to code all the components I need, then after looking after the architecture I'll combine
#them in a way that can make the bert done done. it gonna be fun and painful but it gonna be fun
class TextClassification(nn.Module):
    def __init__(self, input_shape, hidden_layers, output_shape) -> None:
        
        pass

    def forward(self) -> torch.Tensor:
        pass

class SelfAttention(nn.Module):
        #this represents an individual attention mechanism which can be run in parallel with more 
        #attention mechanisms
        
#after passing data through the posotionan encoders (first step) the data will usually be in dimensions:
#1, 4, 512 with 1 word has 4 vectors which each has 512 values each

    #in a paper, I saw that you can instead pass in a dictionary instead of individual values
    def init(self, config: Dict): #the config dictionary should have embed_size and head values
        super(SelfAttention, self).init()

        self.embed_size = config.embed_size
        self.heads = config.heads
        self.head_dim = self.embed_size // self.heads #returns the integer value of the quotient (floor division: dumps the decimal values)

        assert (self.head_dim * self.heads == self.embed_size) # raises an error if embed size is not able to be divided by heads

        self.values = nn.Linear(in_features=self.head_dim,
                                out_features=self.head_dim,
                                bias=False)
        self.keys = nn.Linear(in_features=self.head_dim,
                                out_features=self.head_dim,
                                bias=False)
        self.query = nn.Linear(in_features=self.head_dim,
                                out_features=self.head_dim,
                                bias=False)

        self.fully_connected_out = nn.Linear(in_features=self.embed_size, #this should be the number of heads times the head dimension, which should be equal to the embed_size
                                             out_features=self.embed_size)

    def forward(self, 
                values: torch.Tensor, 
                keys: torch.Tensor,     
                query: torch.Tensor, 
                mask: bool) -> torch.Tensor:

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] #this value will correspond to the input/source sentence length and the target sentence length

        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        query = query.reshape(N, key_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)

        torch.einsum("") #i gotta do some linear math here with some stuff



class MultiHeadedAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        pass
        #this does the parallel thing in the self attentino above to make the training process faster
        #cus boy do i aint got a lotta time, only a few hours at night at most. why did i do this to
        #myself

    def forward(self) -> torch.Tensor:
        pass


