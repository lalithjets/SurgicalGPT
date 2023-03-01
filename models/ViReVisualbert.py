import math
import torch
import torch.nn as nn

from transformers.utils import logging
from transformers import VisualBertPreTrainedModel

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "VisualBertConfig"

from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN


class GCN(nn.Module):
    """
    Graph Convolution network for Global interaction space 
    init    : 
        num_state, num_node, bias=False
        
    forward : x
    """
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)        
        h = self.conv2(self.relu(h))

        return h


# class GloRe_Unit(nn.Module):
#     """
#     Global Reasoning Unit (GR/GloRe)
#     init    : 
#         num_in, num_mid, stride=(1, 1), kernel=1
        
#     forward : x
#     """    
#     def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
#         super(GloRe_Unit, self).__init__()

#         self.num_s = int(2 * num_mid)
#         self.num_n = int(1 * num_mid)

#         kernel_size = (kernel, kernel)
#         padding = (1, 1) if kernel == 3 else (0, 0)

#         # Reduce dimension
#         self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
#         # generate graph transformation function
#         self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
#         # ----------
#         self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
#         # ----------
#         # tail: extend dimension
#         self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),groups=1, bias=False)

#         self.blocker = nn.BatchNorm2d(num_in)

#     def forward(self, x):
#         '''
#         Parameter x dimension : (N, C, H, W)
#         '''
#         batch_size = x.size(0)
        
#         # (n, num_in, h, w) --> (n, num_state, h, w)
#         #                   --> (n, num_state, h*w)
#         x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
#         print('1',x_state_reshaped.size())
        
#         # (n, num_in, h, w) --> (n, num_node, h, w)
#         #                   --> (n, num_node, h*w)
#         x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
#         print('2',x_proj_reshaped.size())
        
#         # (n, num_in, h, w) --> (n, num_node, h, w)
#         #                   --> (n, num_node, h*w)
#         x_rproj_reshaped = x_proj_reshaped
#         print('3',x_rproj_reshaped.size())

#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#         # Projection: Coordinate space -> Interaction space
#         # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
#         x_n_state = torch.matmul( x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        
#         # normalize
#         x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

#         # reverse projection: interaction space -> coordinate space
#         # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
#         x_n_rel = self.gcn(x_n_state)

#         # Reverse projection: Interaction space -> Coordinate space
#         # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
#         x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#         # (n, num_state, h*w) --> (n, num_state, h, w)
#         x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

#         # -----------------
#         # (n, num_state, h, w) -> (n, num_in, h, w)
#         out = x + self.blocker(self.fc_2(x_state))
        
#         return out

class GloRe_Unit1D(nn.Module):
    """
    Global Reasoning Unit (GR/GloRe)
    init    : 
        num_in, num_mid, stride=(1, 1), kernel=1
        
    forward : x
    """    
    def __init__(self, num_in, num_mid, stride=1, kernel=1):
        super(GloRe_Unit1D, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = kernel
        padding = 1 if kernel == 3 else 0

        # Reduce dimension
        self.conv_state = nn.Conv1d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv1d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv1d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=1,groups=1, bias=False)

        self.blocker = nn.BatchNorm1d(num_in)

    def forward(self, x):
        '''
        Parameter x dimension : (N, C, H, W)
        '''
        batch_size = x.size(0)
        
        x_token_state = x.permute(0, 2, 1)                  # (n, tokens, token state[2048]) --> (n, 2048, tokens)
        
        x_state_reshaped = self.conv_state(x_token_state)   # (n, nummid, tokens)
        
        x_proj_reshaped = self.conv_proj(x_token_state)     # (n, tokens, numin) --> (n, num_node, tokens)
        
        x_rproj_reshaped = x_proj_reshaped                  # (n, num_node, tokens)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Projection: Coordinate space -> Interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul( x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        
        # normalize
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_n_rel = self.gcn(x_n_state)

        # Reverse projection: Interaction space -> Coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped) 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        # x_state = x_state_reshaped.view(batch_size, self.num_s, *x_token_state.size()[2:])
        
        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_gr = self.blocker(self.fc_2(x_state_reshaped))


        out = x_token_state + x_gr
        out = out.permute(0,2,1)

        return out


class ViReVisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(self.token_type_embeddings.weight.data.clone(), requires_grad=True)
            self.visual_position_embeddings.weight.data = nn.Parameter(self.position_embeddings.weight.data.clone(), requires_grad=True)

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, visual_embeds=None, visual_token_type_ids=None, image_text_alignment=None,):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(visual_embeds.size()[:-1], dtype=torch.long, device=self.position_ids.device)
                
            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = token_type_embeddings.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                    logger.warning(
                        "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                        " error."
                    )
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)

                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.size(1)} "
                            f"should be the same as `visual_embeds` length: {visual_embeds.size(1)}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, : visual_embeds.size(1), :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(
                    visual_position_ids
                )
            else:
                visual_position_ids = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)

            visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
            embeddings = torch.cat((embeddings, visual_embeddings), dim=1) #[1,[4+5],2048]
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ViReVisualBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads #8
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 2048/8 = 256
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 8*256

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # {8+5} + (8,256)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False,):

        mixed_query_layer = self.query(hidden_states) #[1,{8+5},2048]
        
        key_layer = self.transpose_for_scores(self.key(hidden_states))  #[1,8,{8+5},256]
        value_layer = self.transpose_for_scores(self.value(hidden_states)) #[1,8,{8+5},256]
        
        query_layer = self.transpose_for_scores(mixed_query_layer) #[1,8,{8+5},256]
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [1,8,{8+5},256]*[1,8,256,{8+5}] = [1,8,{8+5},{8+5}]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #[1,8,{8+5},{8+5}]
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in VisualBertSelfAttentionModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) #[1,8,{8+5},256] = [1, 8, 13, 13]*[1, 8, 13, 256]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #1, 13, 8, 256
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  #[1,{8+5},2048]
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->VisualBert
class VisualBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ViReVisualBertAttention(nn.Module):
    def __init__(self, config, version):
        super().__init__()
        self.self = ViReVisualBertSelfAttention(config)
        self.output = VisualBertSelfOutput(config)
        self.pruned_heads = set()
        self.version = version
        if self.version == 'v1' or self.version == 'v2':
            self.GloRe = GloRe_Unit1D(config.hidden_size, 128, stride=(1, 1), kernel=1)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False,):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions,) #([1, 13, 2048],[1, 8, 13, 13])
        
        if self.version == 'v1':
            GloTeViRel_attention = self.GloRe(self_outputs[0])
            selfAndRel_attention = self_outputs[0]+ GloTeViRel_attention
            attention_output = self.output(selfAndRel_attention, hidden_states) # [1, 13, 2048] = [1, 13, 2048],[1, 13, 2048]
        elif self.version == 'v2':
            GloTeViRel_attention = self.GloRe(self_outputs[0])
            attention_output = self.output(self_outputs[0], hidden_states) # [1, 13, 2048] = [1, 13, 2048],[1, 13, 2048]
            attention_output = attention_output+GloTeViRel_attention
        else:
            attention_output = self.output(self_outputs[0], hidden_states) # [1, 13, 2048] = [1, 13, 2048],[1, 13, 2048]

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->VisualBert
class VisualBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states) # [1, 33, 3072] <- [1, 33, 2048]
        hidden_states = self.intermediate_act_fn(hidden_states) # [1, 33, 3072]
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->VisualBert
class VisualBertOutput(nn.Module):
    def __init__(self, config, version):
        super().__init__()
        
        if version == 'v3':
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ViReVisualBertLayer(nn.Module):
    def __init__(self, config, version):
        super().__init__()
        self.version = version
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViReVisualBertAttention(config, self.version)
        
        
        if self.version == 'v1' or self.version == 'v2':
            self.intermediate = VisualBertIntermediate(config)
            self.output = VisualBertOutput(config, self.version)

        elif self.version == 'v3':
            self.GloRe = GloRe_Unit1D(config.hidden_size, 128, stride=1, kernel=1)
            self.output = VisualBertOutput(config, self.version)
        
        elif self.version == 'v4':
            self.GloRe = GloRe_Unit1D(config.hidden_size, 128, stride=1, kernel=1)
        

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False,):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions,)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        
        if self.version == 'v1' or self.version == 'v2':
            intermediate_output = self.intermediate(attention_output) #[1, 33, 3072] <- [1, 33, 2048]
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output
        
        elif self.version == 'v3':
            intermediate_output = self.GloRe(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output

        elif self.version == 'v4':
            intermediate_output = self.GloRe(attention_output)
            return intermediate_output
        


class ViReVisualBertEncoder(nn.Module):
    def __init__(self, config, version):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViReVisualBertLayer(config, version) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True,):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module),hidden_states,attention_mask,layer_head_mask,)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states,all_hidden_states,all_self_attentions,] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->VisualBert
class VisualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViReVisualBertModel(VisualBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config, version, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = ViReVisualBertEmbeddings(config)
        self.encoder = ViReVisualBertEncoder(config, version)

        self.pooler = VisualBertPooler(config) if add_pooling_layer else None

        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = ViReVisualBertLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                visual_embeds=None, visual_attention_mask=None, visual_token_type_ids=None, image_text_alignment=None, output_attentions=None, output_hidden_states=None, return_dict=None):        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if visual_embeds is not None:
            visual_input_shape = visual_embeds.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if visual_embeds is not None and visual_attention_mask is None:
            visual_attention_mask = torch.ones(visual_input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if visual_embeds is not None:
            combined_attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=-1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(combined_attention_mask, (batch_size, input_shape + visual_input_shape), device=device)
        else:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, (batch_size, input_shape))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment,)

        if self.bypass_transformer and visual_embeds is not None:
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_embedding_output = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :, text_length, :text_length]

            encoded_outputs = self.encoder(text_embedding_output, attention_mask=text_extended_attention_mask, output_attentions=output_attentions, 
                                            output_hidden_states=output_hidden_states, return_dict=return_dict,)
            sequence_output = encoded_outputs[0]
            concatenated_input = torch.cat((sequence_output, visual_embedding_output), dim=1)
            sequence_output = self.additional_layer(concatenated_input, extended_attention_mask)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        else:
            encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,output_attentions=output_attentions, 
                                            output_hidden_states=output_hidden_states, return_dict=return_dict,)
            sequence_output = encoder_outputs[0]

            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions,)
