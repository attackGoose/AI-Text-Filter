import torch
from torch import nn
import math
from typing import Dict
from random import Random
#update, this is going to be text classification model to tell apart sentences with correct grammar
#and incorrect grammar and to make sure the language outputted isn't offensive
#it will also test if the model's output sentence makes sense given the context of the previous sentence

"""
resource: 

https://arxiv.org/pdf/1810.04805, bert architecture paper
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf


"""

#this will be the bert model, which I will code up in the next commit, or at least part of it.
#goal is to code all the components I need, then after looking after the architecture I'll combine
#them in a way that can make the bert done done. it gonna be fun and painful but it gonna be fun
class TextClassification(nn.Module):
    def __init__(self, input_shape, hidden_layers, output_shape) -> None:
        
        pass

    def forward(self) -> torch.Tensor:
        pass

class PositionalEmbedding(nn.Module):
    def __init__(self, model_dim, max_len=128):

        self.pos_encode = torch.zeros(max_len, model_dim).float() 
        #position encoding, no more information needed 
        self.pos_encode.requires_grad = False

        #for every dimension in the value: assign it an encoding, formula is sin(position/(10000^(2i/dimension)))
        for key_pos in range(max_len): 
            #key as in each row value, not related to key dimension in later areas of the model
            for value_pos in range(0, model_dim, 2): 
                #value as in encoded value
                self.pos_encode[key_pos, value_pos] = math.sin(key_pos / (10000 ** ((2 * value_pos)/model_dim)))
                self.pos_encode[key_pos, value_pos + 1] = math.cos(key_pos / (10000 ** ((2 * (value_pos + 1))/model_dim)))

        #give an extra dimension for the batch size (to train faster)
        self.pos_encode.unsqueeze()

    def forward(self) -> torch.Tensor:
        return self.pos_encode
    
#bert model has its another embedding to further better understand the text
#required: positional embedding, 
class BERTEmbedding(nn.Module): 
    """
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
    sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len, dropout):
        
        #how many embeddings to add/embedding size of token embedding
        self.embed_size = embed_size

        #makes the token embedding
        self.token_embed = nn.Embedding(
            vocab_size, 
            embed_size, 
            padding_idx=0)
        #makes the segment embedding
        self.seg_embed = nn.Embedding(
            5, #random number, I'm still trying to find the actual number from the research paper
            embedding_dim=embed_size, 
            padding_idx=0)
        #sets a random percentage of the inputted values to 0 during training by preventing co-adaptation of neurons
        self.dropout = nn.Dropout(p=dropout) 



class MultiheadedAttention(nn.Module): #thi is the self-attention mechanism, of which also includes position encoding of data but this is also important
        #this represents an individual attention mechanism which can be run in parallel with more 
        #attention mechanisms
        
#after passing data through the posotionan encoders (first step) the data will usually be in dimensions:
#1, 4, 512 with 1 word has 4 vectors which each has 512 values each

    #in a paper, I saw that you can instead pass in a dictionary instead of individual values
    def init(self, config: Dict): #the config dictionary should have embed_size and head values

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



