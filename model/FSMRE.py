"""
The main model of FSMRE

Author: Tong
Time: 11-04-2021
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class FSMRE(nn.Module):
    """
    few-shot multi-relation extraction
    """
    
    def __init__(self, encoder=None, aggregator=None, propagator=None, hidden_dim=100, proto_dim=200, support_size=25,
                 query_size=10, max_length=50) -> None:
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
        
        self.propagator = propagator
        
        # attention_layer
        self.rel_aware_att_layer = nn.Sequential(
            nn.Linear(self.hidden_dim + self.proto_dim, self.hidden_dim),
            nn.Sigmoid()
        )
    
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
        
        '''Step 0 & 1: encoding and propagation'''
        batch_entities, batch_context = self._encode_aggregation(support_set)
        
        '''Step 2: general propagation '''
        batch_entities, batch_context = self.propagator(batch_entities, batch_context)
        
        '''Step 3: obtain prototype embedding'''
        # todo: get prototype from batch_entities
        prototype = None
        
        return prototype
    
    def _process_query(self, prototype, query_set):
        """
        generate predictions for query instances
        Args:
            prototype (torch.Tensor):
            query_set (tuple): refer to support_set
        Returns:
            predictions
        """
        '''Step 0 & 1: encoding and propagation'''
        batch_entities, batch_context = self._encode_aggregation(query_set)
        
        '''Step 2: general propagation '''
        batch_entities, batch_context = self.propagator(batch_entities, batch_context)
        
        '''Step 3: relation-aware propagation'''
        rel_att = self._relation_aware_attention(prototype, batch_context)
        batch_context = rel_att * batch_context
        batch_entities, batch_context = self.propagator(batch_entities, batch_context)
        
        '''Step 4: prototype-based classification'''
        # todo: get prototype from batch_entities
        prediction = None
        
        return prediction
    
    def _encode_aggregation(self, input_set):
        """
        general processing of support_set or query_set
        Args:
            input_set (tuple): support_set or query_set

        Returns:
        batch_entities, batch_contexts
        """
        # out of encoder:
        # - 0. the last hidden state (batch_size, sequence_length, hidden_size)
        # - 1. the pooler_output of the classification token (batch_size, hidden_size)
        # - 2. the hidden_states of the outputs of the model at each layer and the initial embedding outputs
        #    (batch_size, sequence_length, hidden_size)
        
        '''Step 0: encoding '''
        # [-1] for the last layer presentation -> size: sentence_num * max_length * h_dim(768)
        encodings = self.encoder(input_set[0], input_set[1])[2][-1]
        
        '''Step 1 - 1: entity aggregation'''
        # sequencial_processing: process entity
        # todo: parallelization in entity aggregation
        batch_entities = []
        for i, entities_list in enumerate(input_set[2]):
            s_encodings = encodings[i]
            s_ent_list = []
            for ent_mask in entities_list:
                s_ent_list.append(self._aggregate_entity(s_encodings, ent_mask))
            pass
            # append(Tensor_size: num_entities * self.hidden_size)
            batch_entities.append(torch.cat(s_ent_list))
        pass
        
        '''Step 1 - 2: context aggregation'''
        batch_context = []
        # todo: parallelization in context aggregation
        for i, context_matrix in enumerate(input_set[3]):
            # context_matrix size: num_ent * num_ent * max_length
            s_encodings = encodings[i]  # size: max_length * encoding_dim
            
            # todo: masked_context = context_matrix * s_encodings # num_ent * num_ent * max_length * encoding_dim
            batch_context.append(self._aggregate_context(s_encodings, context_matrix))  # todo: debug
            pass
        pass
        
        return batch_entities, batch_context
    
    def _aggregate_entity(self, sentence_encodings, entity_mask):
        """
        generate entity encoding from sentence encodings.
        Args:
            sentence_encodings (torch.Tensor): sentence encodings
            entity_mask (torch.Tensor): context_mask

        Returns:
            node-weight (entity embedding) for relation graph
        """
        return self.aggregator(sentence_encodings, entity_mask)
    
    def _aggregate_context(self, sentence_encodings, context_mask):
        """
        generate pair-wise context encodings from sentence encodings.
        Args:
            sentence_encodings (torch.Tensor): sentence encodings
            context_mask (torch.Tensor): context_mask

        Returns:
            edge-weight (context embedding) for relation graph
        """
        return self.aggregator(sentence_encodings, context_mask)
    
    def _relation_aware_attention(self, prototype, weight):
        """
        calculate attention weight for relation-aware propagation
        Args:
            prototype (torch.Tensor): prototype embedding
            weight (torch.Tensor): edge-weight (context embedding) for relation graph

        Returns:
            attention weight
        """
        # todo: expand prototype
        return self.rel_aware_att_layer()
