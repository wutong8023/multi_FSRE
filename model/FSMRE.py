"""
The main model of FSMRE

Author: Tong
Time: 12-04-2021
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class FSMRE(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """
    
    def __init__(self, encoder=None, aggregator=None, hidden_dim=100, proto_dim=200, support_size=25, query_size=10,
                 max_length=50) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(FSMRE, self).__init__()
        # the number of support instances inside a task
        self.support_size = support_size
        # the number of query instances inside a task
        self.query_size = query_size
        # the max length of sentences
        self.max_length = max_length
        
        self.hidden_dim = hidden_dim
        self.proto_dim = proto_dim
        
        # default: Bert encoder
        self.encoder = encoder
        # default: BiLSTM, simplified: average
        self.aggregator = aggregator
    
    def forward(self, support_set, query_set):
        """
        generate prototype embedding from support set, and conduct prediction for query_set
        Args:
            support_set (tuple):
                [0]: instances, torch.Tensor, sentence_num * max_length
                [1]: mask, torch.Tensor, sentence_num * max_length
                [2]: entities, [torch.Tensor], sentence_num * entity_num * entity_mask
                [3]: context, [torch.Tensor], sentence_num * entity_num * entity_num * context_mask
                [4]: label, [torch.Tensor], sentence_num * entity_num * entity_num
            query_set (tuple):
                [0]: instances, torch.Tensor, sentence_num * max_length
                [1]: mask, torch.Tensor, sentence_num * max_length
                [2]: entities, [torch.Tensor], sentence_num * entity_num * entity_mask
                [3]: context, [torch.Tensor], sentence_num * entity_num * entity_num * context_mask
                [4]: label, [torch.Tensor], sentence_num * entity_num * entity_num
        Returns:
            prediction: [torch.Tensor], sentence_num * entity_num * entity_num * class_size
            query_set[4]: [torch.Tensor], sentence_num * entity_num * entity_num
        """
        # get prototype embedding for each class
        # size: class_size * self.prototype_size
        prototype = self._process_support(support_set)
        prediction = self._process_query(prototype, query_set)
        return prediction
    
    def _process_support(self, support_set):
        """
        generate prototype embedding for each class
        Args:
            support_set (tuple):
                [0]: instances, torch.Tensor, sentence_num * max_length
                [1]: mask, torch.Tensor, sentence_num * max_length
                [2]: entities, [torch.Tensor], sentence_num * entity_num * entity_mask
                [3]: context, [torch.Tensor], sentence_num * entity_num * entity_num * context_mask
                [4]: label, [torch.Tensor], sentence_num * entity_num * entity_num
        Returns:
            support_set
        """
        # out of encoder:
        # 0. the last hidden state (batch_size, sequence_length, hidden_size)
        # 1. the pooler_output of the classification token (batch_size, hidden_size)
        # 2. the hidden_states of the outputs of the model at each layer and the initial embedding outputs
        #    (batch_size, sequence_length, hidden_size)

        # -1 for the last layer presentation -> size: sentence_num * max_length * h_dim(768)
        encodings = self.encoder(support_set[0], support_set[1])[2][-1]
        
        # sequencial_processing
        sent_entities = []
        
        
        
        
        
        
        prototype = []
        return prototype
    
    def _process_query(self, prototype, query_set):
        prediction = []
        return prediction
    
    def _aggregate_entity(self, sentence_encodings, entity_mask):
        
        return self.aggregator()
    
    def _aggregate_context(self):
        pass
