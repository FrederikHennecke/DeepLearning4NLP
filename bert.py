import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask
import spacy
from tokenizer import BertTokenizer


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        Apply residual connection to any layer and normalize the output.
        This function is applied after the multi-head attention layer or the feed forward layer.

        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied
        ln_layer: the layer norm to be applied
        """
        ### TODO
        # raise NotImplementedError
        # Hint: Remember that BERT applies dropout to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized.

        fc_layer = dense_layer(output)
        fc_output = dropout(fc_layer)
        skip_connect = input + fc_output
        normalized_output = ln_layer(skip_connect)
        return normalized_output

    def forward(self, hidden_states, attention_mask):
        """
        A single pass of the bert layer.

        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
        attention_mask: the mask for the attention layer

        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        ### TODO
        # raise NotImplementedError

        attention_output = self.self_attention(hidden_states, attention_mask)
        # print(f"attention unnormalized shape: {attention_output.shape}")
        attention_output = self.add_norm(
            hidden_states,
            attention_output,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm,
        )
        # print(f"attention normalized shape: {attention_output.shape}")
        interm_output = self.interm_af(self.interm_dense(attention_output))
        # print(f"interm_output_shape: {interm_output.shape}")
        layer_output = self.add_norm(
            attention_output,
            interm_output,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm,
        )
        # print(f"layer_output_shape: {layer_output.shape}")
        return layer_output


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        # initialize parameters for more input features
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        ner_tags_spacy = self.nlp.get_pipe("ner").labels
        pos_tags_spacy = self.nlp.get_pipe("tagger").labels
        self.ner_tag_embedding = nn.Embedding(
            len(ner_tags_spacy) + 1, config.hidden_size
        )
        self.pos_tag_embedding = nn.Embedding(
            len(pos_tags_spacy) + 1, config.hidden_size
        )
        self.pos_tag_vocab = {tag: i for i, tag in enumerate(pos_tags_spacy)}
        self.ner_tag_vocab = {tag: i for i, tag in enumerate(ner_tags_spacy)}
        self.input_cache = {}
        self.init_weights()

    def embed(self, input_ids, additional_input=False):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        word_embeds = self.word_embedding(input_ids)

        ### TODO
        # raise NotImplementedError
        # Get position index and position embedding from self.pos_embedding into pos_embeds.

        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        ### TODO
        # raise NotImplementedError
        # Get token type ids, since we are not considering token type,
        # this is just a placeholder.
        tk_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device
        )
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        ### TODO
        # raise NotImplementedError
        # Add three embeddings together; then apply embed_layer_norm and dropout and
        # return the hidden states.

        if additional_input:
            # get the pos and ner tags

            all_pos_tags = []
            all_ner_tags = []
            for sequence_id in input_ids:
                sequence_id_tup = tuple(sequence_id.tolist())
                if sequence_id_tup in self.input_cache:
                    pos_tags, ner_tags = self.input_cache[sequence_id_tup]
                else:
                    tokens = self.tokenizer.convert_ids_to_tokens(sequence_id.tolist())
                    token_strings = [
                        token
                        for token in tokens
                        if token not in ["[PAD]", "[CLS]", "[SEP]"]
                    ]
                    input_string = self.tokenizer.convert_tokens_to_string(
                        token_strings
                    )
                    tokenized = self.nlp(input_string)
                    pos_tags = [0] * len(tokens)
                    ner_tags = [0] * len(tokens)
                    counter = -1
                    for i in range(len(token_strings)):
                        if not token_strings[i].startswith("##"):
                            counter += 1
                        pos_tags[i + 1] = self.pos_tag_vocab.get(
                            tokenized[counter].tag_, 0
                        )
                        ner_tags[i + 1] = self.ner_tag_vocab.get(
                            tokenized[counter].ent_type_, 0
                        )

                    self.input_cache[sequence_id_tup] = (pos_tags, ner_tags)

                all_pos_tags.append(pos_tags)
                all_ner_tags.append(ner_tags)

            pos_tags_ids = torch.tensor(
                all_pos_tags, dtype=torch.long, device=input_ids.device
            )
            ner_tags_ids = torch.tensor(
                all_ner_tags, dtype=torch.long, device=input_ids.device
            )

        else:
            pos_tags_ids = torch.zeros(
                input_shape, dtype=torch.long, device=input_ids.device
            )
            ner_tags_ids = torch.zeros(
                input_shape, dtype=torch.long, device=input_ids.device
            )
        pos_tag_embeds = self.pos_tag_embedding(pos_tags_ids)
        ner_tag_embeds = self.ner_tag_embedding(ner_tags_ids)
        embeds = (
            word_embeds + pos_embeds + tk_type_embeds + pos_tag_embeds + ner_tag_embeds
        )

        output_embeds = self.embed_layer_norm(embeds)
        output_embeds = self.embed_dropout(output_embeds)
        return output_embeds

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # pass the hidden states through the encoder layers
        all_hidden_states = []
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)
            all_hidden_states.append(hidden_states)
        return all_hidden_states

    def first_token(self, input_sequence):
        # get cls token hidden state

        first_tk = input_sequence[:, 0]
        pooled_output = self.pooler_dense(first_tk)
        pooled_output = self.pooler_af(pooled_output)
        return pooled_output

    def forward(self, input_ids, attention_mask, additional_input=False):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(
            input_ids=input_ids, additional_input=additional_input
        )

        # feed to a transformer (a stack of BertLayers)
        all_encoded_layers = self.encode(
            embedding_output, attention_mask=attention_mask
        )
        last_hidden_state = all_encoded_layers[-1]
        pooler_output = self.first_token(last_hidden_state)
        sequence_output2 = all_encoded_layers[-2]
        pooled_output2 = self.first_token(sequence_output2)
        sequence_output3 = all_encoded_layers[-3]
        pooled_output3 = self.first_token(sequence_output3)
        sequence_output4 = all_encoded_layers[-4]
        pooled_output4 = self.first_token(sequence_output4)
        sequence_output5 = all_encoded_layers[-5]
        pooled_output5 = self.first_token(sequence_output5)
        sequence_output6 = all_encoded_layers[-6]
        pooled_output6 = self.first_token(sequence_output6)
        sequence_output7 = all_encoded_layers[-7]
        pooled_output7 = self.first_token(sequence_output7)
        sequence_output8 = all_encoded_layers[-8]
        pooled_output8 = self.first_token(sequence_output8)
        sequence_output9 = all_encoded_layers[-9]
        pooled_output9 = self.first_token(sequence_output9)
        sequence_output10 = all_encoded_layers[-10]
        pooled_output10 = self.first_token(sequence_output10)
        sequence_output11 = all_encoded_layers[-11]
        pooled_output11 = self.first_token(sequence_output11)
        sequence_output12 = all_encoded_layers[-12]
        pooled_output12 = self.first_token(sequence_output12)
        return {
            "all_encoded_layers": all_encoded_layers,
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
            "pooled_output2": pooled_output2,
            "pooled_output3": pooled_output3,
            "pooled_output4": pooled_output4,
            "pooled_output5": pooled_output5,
            "pooled_output6": pooled_output6,
            "pooled_output7": pooled_output7,
            "pooled_output8": pooled_output8,
            "pooled_output9": pooled_output9,
            "pooled_output10": pooled_output10,
            "pooled_output11": pooled_output11,
            "pooled_output12": pooled_output12,
        }
