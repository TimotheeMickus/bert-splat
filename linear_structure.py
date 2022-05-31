"""
ABANDON ALL HOPE YE WHO ENTER HERE

This code is a hadge-podge of monkey-patches  and hacks, based on the
HuggingFace Transformers library. It is not clean, it is not friendly, it is not
optimized. It is a hell of random additions piled up on the spot to get me what
I needed at the time.

With this in mind, here are few pointers to help you out through this specific
circle of programmer's hell.
    1/ The whole code has only been tested and used for transformers.BertModel
variants. I have no idea how well this would translate to other HuggingFace
models, you're on your own.
    2/ The model that's going to be worked on corresponds to the `MODEL_NAME`
variable. It should be a valid HuggingFace name, so a path to your custom Bert
probably works, but again, keep in mind this is hell.
    3/ The general logic of this script is that I duplicated the `forward(...)`
of every single Bert submodule, and hacked them to return both their normal
output as well as the linear subterm I was interested in. These subterms
correspond to the various `keywords` dictionary (because not only is my code
shite, my naming sense is awful as well!). These duplicated `forward(...)`
should all be named `run_XXX(...)`, with XXX the corresponding submodule
variable name.
    4/ sub-terms are accessed using the `get_factors(...)`,
`read_factors_last_layer(...)` & other similar functions.
    5/ beware of the `pickle_file_` variable! Its purpose is to drop stuff I
need to access later on, e.g. the feedforward inputs and outputs. You want to
explicitly set that to None to avoid having maxive dumps of data.
    6/ the importance metric stuff used to visualize the subterms relies on the
`tally_XXX(...)` functions.
    7/ calling this script on its own will either compute the raw data to
visualize cross-layer importance of the four terms, or run MLM predictions based
on the default output projection. This behavior is controled with the `MODE`
variable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

MODEL_NAME = "bert-base-uncased"
# MODEL_NAME = "dslim/bert-base-NER-uncased"
# MODEL_NAME = "twmkn9/bert-base-uncased-squad2"

model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()
torch.autograd.set_grad_enabled(False)

def run_layer_norm(ln_module, input_tensor):
    # torch.allclose(ln(ipt), ln.bias + ((ln.weight * (ipt - ipt.mean(2, keepdims=True))) / (ipt.var(2, keepdims=True, unbiased=False) + ln.eps).sqrt()), atol=ln.eps) # should be True
    output = ln_module(input_tensor)
    keywords = {
        'bias': ln_module.bias,
        'gain': ln_module.weight,
        'mean': input_tensor.mean(2, keepdims=True),
        'std': (input_tensor.var(2, keepdims=True, unbiased=False) + ln_module.eps).sqrt()
    }
    return output, keywords


def run_bert_embedding(embedding_layer, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]
    #
    seq_length = input_shape[1]
    #
    if position_ids is None:
        position_ids = embedding_layer.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    #
    # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
    # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
    # issue #5664
    if token_type_ids is None:
        if hasattr(embedding_layer, "token_type_ids"):
            buffered_token_type_ids = embedding_layer.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=embedding_layer.position_ids.device)
    #
    if inputs_embeds is None:
        inputs_embeds = embedding_layer.word_embeddings(input_ids)
    token_type_embeddings = embedding_layer.token_type_embeddings(token_type_ids)
    #
    embeddings = inputs_embeds + token_type_embeddings
    if embedding_layer.position_embedding_type == "absolute":
        position_embeddings = embedding_layer.position_embeddings(position_ids)
        embeddings += position_embeddings
    embeddings, layer_norm_keywords = run_layer_norm(embedding_layer.LayerNorm, embeddings)
    embeddings = embedding_layer.dropout(embeddings)
    keywords = {
        'layer_norm': layer_norm_keywords,
        'token_type_embeddings': token_type_embeddings,
        'position_embeddings': position_embeddings,
        'inputs_embeddings': inputs_embeds,
    }
    return embeddings, keywords

def run_bert_self_attention(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            unbiased_value_layer = self.transpose_for_scores(F.linear(encoder_hidden_states, self.value.weight))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            unbiased_value_layer = self.transpose_for_scores(F.linear(hidden_states, self.value.weight))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            unbiased_value_layer = self.transpose_for_scores(F.linear(hidden_states, self.value.weight))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        keywords = {
            'weights': attention_probs,
            'values': value_layer,
            'unbiased_value_layer': unbiased_value_layer,
            'value_bias': self.value.bias
        }
        return outputs, keywords

def run_bert_self_output(self, hidden_states, input_tensor):
    hidden_states_ = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states_)
    hidden_states, layer_norm_keywords = run_layer_norm(self.LayerNorm, hidden_states_ + input_tensor)
    keywords = {'output': hidden_states, 'raw_hidden_states':hidden_states_, 'layer_norm':layer_norm_keywords}
    return hidden_states, keywords

def run_bert_attention(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    self_outputs, self_keywords = run_bert_self_attention(
        self.self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        output_attentions,
    )

    unweighted_outputs = self_keywords['unbiased_value_layer']
    # unweighted_outputs = self.output.dense(unweighted_outputs.transpose(2, 1).contiguous().flatten(start_dim=2))

    H_n, H_d = self.self.num_attention_heads, self.self.all_head_size // self.self.num_attention_heads
    unweighted_attention_heads = []
    for token_mha_factor in unweighted_outputs.permute(0,2,1,3).squeeze(0).unbind(0):
        this_head = []
        for idx_to_copy in range(self.self.num_attention_heads):
            zero_padded = torch.zeros_like(token_mha_factor)
            zero_padded[idx_to_copy] = token_mha_factor[idx_to_copy]
            this_head.append(F.linear(zero_padded.view(-1), self.output.dense.weight))
        unweighted_attention_heads.append(torch.stack(this_head))
        #
        # zero_padded = torch.cat([attention_head, torch.zeros((H_n, H_n * H_d))], dim=-1).view(-1)[:H_n*H_n*H_d].view(H_n, H_n*H_d)
        # unweighted_attention_heads.append(F.linear(zero_padded, self.output.dense.weight))
    unweighted_outputs = torch.stack(unweighted_attention_heads).unsqueeze(0).transpose(1,2).contiguous()

    attention_output, attention_output_keywords = run_bert_self_output(self.output, self_outputs[0], hidden_states)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    keywords = {
        'attention_output':attention_output_keywords,
        'self_outputs':self_keywords,
        'outputs': outputs,
        'unweighted_outputs':unweighted_outputs,
        # as attention heights sum to one across heads, we can just ignore the breakdown over attended token
        # and pass the value bias through the dense output to get an equivalent output.
        # I've coded it like that to stress that we have two items (the bias for W_0 and the biases of each W_V),
        # but it has to be strictly equivalent to just passing the value bias into the output dense layer.
        'output_bias_correction': self.output.dense.bias + F.linear(self_keywords['value_bias'], self.output.dense.weight)
    }
    return outputs, keywords

def run_bert_intermediate(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states, {'hidden_states': hidden_states}

def run_bert_output(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states_ = self.dropout(hidden_states)
    hidden_states, keywords = run_layer_norm(self.LayerNorm, hidden_states_ + input_tensor)
    return hidden_states, {'hidden_states': hidden_states, 'raw_hidden_states': hidden_states_, 'layer_norm': keywords, 'bias': self.dense.bias}

def run_bert_layer(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    self_attention_outputs, attention_keywords = run_bert_attention(
        self.attention,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions=output_attentions,
        past_key_value=self_attn_past_key_value,
    )
    attention_output = self_attention_outputs[0]

    # if decoder, the last output is tuple of self-attn cache
    if self.is_decoder:
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
    else:
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    cross_attn_present_key_value = None
    if self.is_decoder and encoder_hidden_states is not None:
        assert hasattr(
            self, "crossattention"
        ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

        # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            cross_attn_past_key_value,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        cross_attn_present_key_value = cross_attention_outputs[-1]
        present_key_value = present_key_value + cross_attn_present_key_value

    # layer_output1 = apply_chunking_to_forward(
    #      self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
    # )
    layer_output, layer_output_keywords = run_feed_forward_chunk(self, attention_output)
    # assert torch.allclose(layer_output1, layer_output)
    outputs = (layer_output,) + outputs

    # if decoder, return the attn key/values as the last output
    if self.is_decoder:
        outputs = outputs + (present_key_value,)

    keywords = {
        'attention': attention_keywords,
        "layer_output": layer_output_keywords,
        "outputs":layer_output
    }

    return outputs, keywords

import pickle
pickle_file_ = open('ff_sole.pkl', 'wb')

def run_feed_forward_chunk(self, attention_output, pickle_file=pickle_file_):
    intermediate_output = self.intermediate(attention_output)
    output, keywords = run_bert_output(self.output, intermediate_output, attention_output)
    if pickle_file is not None:
        pickle.dump(attention_output.detach(), pickle_file)
        pickle.dump(keywords['raw_hidden_states'].detach(), pickle_file)
    return output, keywords

def run_bert_encoder(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    all_layer_keywords = []
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs, layer_keywords = run_bert_layer(
                layer_module,
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            all_layer_keywords.append(layer_keywords)

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        ), all_layer_keywords
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    ), all_layer_keywords


def run_bert_model(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    r"""
    encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
        Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        the model is configured as a decoder.
    encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

        If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
        (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
        instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
    use_cache (:obj:`bool`, `optional`):
        If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
        decoding (see :obj:`past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

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

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output, embedding_keywords = run_bert_embedding(
        self.embeddings,
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )
    encoder_outputs, encoder_keywords = run_bert_encoder(
        self.encoder,
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    model_keywords = {
        "embeddings": embedding_keywords,
        "encoder": encoder_keywords
    }
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        return ((sequence_output, pooled_output) + encoder_outputs[1:]), model_keywords

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    ), model_keywords
#
# with torch.no_grad():
#     inputs = tokenizer(["This is an example"], return_tensors="pt")
#     attested, test = run_bert_model(model, **inputs)

def extract_layer_norm_terms(keywords):
    layer_norm_keywords = [
        [keywords['embeddings']['layer_norm']],
    ]
    layer_norm_keywords += [
        [
            layer['attention']['attention_output']['layer_norm'],
            layer['layer_output']['layer_norm'],
        ]
        for layer in keywords['encoder']
    ]
    return layer_norm_keywords

def extract_input_terms(keywords):
    lookups = ['token_type_embeddings', 'position_embeddings', 'inputs_embeddings']
    return [keywords['embeddings'][k] for k in lookups]

def extract_feedforward_terms(keywords):
    return [
        [
            layer['layer_output']['raw_hidden_states'],
            layer['layer_output']['bias']
        ]
        for layer in keywords['encoder']
    ]

def extract_attention_terms(keywords):
    return [
        [
            layer['attention']['self_outputs']['weights'],
            layer['attention']['unweighted_outputs'],
            layer['attention']['output_bias_correction'],
        ]
        for layer in keywords['encoder']
    ]

from functools import reduce, lru_cache

@torch.no_grad()
def get_factors(keywords, up_to_layer=12):
    all_factors_at_every_layer = []
    LN_TERMS = extract_layer_norm_terms(keywords)
    IPT_TERMS = extract_input_terms(keywords)
    FF_TERMS = extract_feedforward_terms(keywords)
    MHA_TERMS = extract_attention_terms(keywords)

    zeroth_layer_factors = {
        'ipt': (LN_TERMS[0][0]['gain'] * (sum(IPT_TERMS) / LN_TERMS[0][0]['std'])).squeeze(0),
        'norm': (LN_TERMS[0][0]['bias'] - (LN_TERMS[0][0]['gain'] * (LN_TERMS[0][0]['mean'] / LN_TERMS[0][0]['std']))).squeeze(0)
    }
    # yield zeroth_layer_factors
    all_factors_at_every_layer.append(zeroth_layer_factors)

    for LAYER_NUMBER in range(up_to_layer):
        ln_terms = LN_TERMS[:LAYER_NUMBER+2]
        ipt_terms = IPT_TERMS
        ff_terms = FF_TERMS[:LAYER_NUMBER+1]
        mha_terms = MHA_TERMS[:LAYER_NUMBER+1]

        # 1. input trace
        all_gains = reduce(torch.mul, (ln['gain'] for layer in ln_terms for ln in layer))
        all_std = reduce(torch.mul, (ln['std'] for layer in ln_terms for ln in layer))
        ipt_contrib = (all_gains * (sum(ipt_terms) / all_std)).squeeze(0)

        # 2. feed forward trace
        renormalized_ff_layers = []
        relevant_ff_layers = lambda layer: [ln_terms[layer][-1]]+[ln for lyr in ln_terms[layer+1:] for ln in lyr]
        for layer, (ff_term, ff_bias) in enumerate(ff_terms, start=1):
            relevant_layers = relevant_ff_layers(layer)
            all_ff_gains = reduce(torch.mul, (ln['gain'] for ln in relevant_layers))
            all_ff_std = reduce(torch.mul, (ln['std'] for ln in relevant_layers))
            renormalized_ff_layers.append((all_ff_gains * (ff_term / all_ff_std)).squeeze(0))
        ff_contrib = (renormalized_ff_layers)

        # 3. multi-head attention trace
        renormalized_mha_layers = []
        relevant_mha_layers = lambda layer: [ln for lyr in ln_terms[layer:] for ln in lyr]
        all_output_bias_corrections = []
        for layer, (weights, unweighted_outputs, output_bias_correction) in enumerate(mha_terms, start=1):
            relevant_layers = relevant_mha_layers(layer)
            all_mha_gains = reduce(torch.mul, (ln['gain'] for ln in relevant_layers))
            all_mha_std = reduce(torch.mul, (ln['std'] for ln in relevant_layers))
            # to get the original raw hidden states, we'd need:
            # weights.unsqueeze(-1) * unweighted_outputs.unsqueeze(-3)).sum(1).sum(-2) + self.output.dense.bias
            # so we have to divde the bias term by the number of items that go through the linear projections.
            weight_applied = (weights.unsqueeze(-1) * unweighted_outputs.unsqueeze(-3))
            mha_terms_unbound = [
                t.unbind(-2)
                for t in (all_mha_gains * (weight_applied / all_mha_std)).squeeze(0).unbind(0)
            ]
            # mha_terms_unbound = [
            #     [
            #         attended_tok + output_bias_correction_
            #         for attended_tok in all_in_head.unbind(-2)
            #     ]
            #     for all_in_head in weight_applied.unbind(dim=1)
            # ]
            # mha_terms_unbound = [t / len(mha_terms_unbound)) for t in mha_terms_unbound]
            renormalized_mha_layers.append(mha_terms_unbound)
            # this will be considered in the 'norm' section
            all_output_bias_corrections.append(all_mha_gains * (output_bias_correction / all_mha_std))
        # n_heads = len(renormalized_mha_layers[0])
        # mha_contribs = [
        #     sum(layer[h_idx] for layer in renormalized_mha_layers)
        #     for h_idx in range(n_heads)
        # ]
        mha_contribs = renormalized_mha_layers

        # 4. correction trace
        all_corrective_terms = []
        flat_ln_terms = [ln for layer in ln_terms for ln in layer]
        bias_relevant = lambda layer: flat_ln_terms[layer+1:]
        mean_relevant = lambda layer: flat_ln_terms[layer:]
        for layer, ln_term in enumerate(flat_ln_terms):
            all_bias_relevant = bias_relevant(layer)
            if all_bias_relevant:
                all_bias_relevant_gains = reduce(torch.mul, (ln['gain'] for ln in all_bias_relevant))
                all_bias_relevant_std = reduce(torch.mul, (ln['std'] for ln in all_bias_relevant))
                bias_correction = all_bias_relevant_gains * (ln_term['bias'] / all_bias_relevant_std)
            else:
                bias_correction = ln_term['bias']
            all_mean_relevant = mean_relevant(layer)
            all_mean_relevant_gains = reduce(torch.mul, (ln['gain'] for ln in all_mean_relevant))
            all_mean_relevant_std = reduce(torch.mul, (ln['std'] for ln in all_mean_relevant))
            mean_correction = all_mean_relevant_gains  * (ln_term['mean']/ all_mean_relevant_std)
            all_corrective_terms.append((bias_correction - mean_correction).squeeze(0))
        correct_contrib = all_corrective_terms
        all_factors = {
            'ipt':ipt_contrib,
            'ff':ff_contrib,
            'mha':mha_contribs,
            'norm':correct_contrib,
        }
        all_factors_at_every_layer.append(all_factors)
    return all_factors_at_every_layer

@torch.no_grad()
def get_factors_coarse_mha(keywords, up_to_layer=12):
    all_factors_at_every_layer = []
    LN_TERMS = extract_layer_norm_terms(keywords)
    IPT_TERMS = extract_input_terms(keywords)
    FF_TERMS = extract_feedforward_terms(keywords)
    MHA_TERMS = extract_attention_terms(keywords)

    zeroth_layer_factors = {
        'ipt': (LN_TERMS[0][0]['gain'] * (sum(IPT_TERMS) / LN_TERMS[0][0]['std'])).squeeze(0),
        'norm': (LN_TERMS[0][0]['bias'] - (LN_TERMS[0][0]['gain'] * (LN_TERMS[0][0]['mean'] / LN_TERMS[0][0]['std']))).squeeze(0)
    }
    all_factors_at_every_layer.append(zeroth_layer_factors)

    for LAYER_NUMBER in range(up_to_layer):
        ln_terms = LN_TERMS[:LAYER_NUMBER+2]
        ipt_terms = IPT_TERMS
        ff_terms = FF_TERMS[:LAYER_NUMBER+1]
        mha_terms = MHA_TERMS[:LAYER_NUMBER+1]

        # 1. input trace
        all_gains = reduce(torch.mul, (ln['gain'] for layer in ln_terms for ln in layer))
        all_std = reduce(torch.mul, (ln['std'] for layer in ln_terms for ln in layer))
        ipt_contrib = (all_gains * (sum(ipt_terms) / all_std)).squeeze(0)

        # 2. feed forward trace
        renormalized_ff_layers = []
        all_ff_biases = []
        relevant_ff_layers = lambda layer: [ln_terms[layer][-1]]+[ln for lyr in ln_terms[layer+1:] for ln in lyr]
        for layer, (ff_term, ff_bias) in enumerate(ff_terms, start=1):
            relevant_layers = relevant_ff_layers(layer)
            all_ff_gains = reduce(torch.mul, (ln['gain'] for ln in relevant_layers))
            all_ff_std = reduce(torch.mul, (ln['std'] for ln in relevant_layers))
            renormalized_ff_layers.append((all_ff_gains * ((ff_term - ff_bias) / all_ff_std)).squeeze(0))
            # this will be considered in the 'norm' section below
            all_ff_biases.append(all_ff_gains * (ff_bias / all_ff_std))
        ff_contrib = sum(renormalized_ff_layers)

        # 3. multi-head attention trace
        renormalized_mha_layers = []
        all_output_bias_corrections = []
        relevant_mha_layers = lambda layer: [ln for lyr in ln_terms[layer:] for ln in lyr]
        for layer, (weights, unweighted_outputs, output_bias_correction) in enumerate(mha_terms, start=1):
            relevant_layers = relevant_mha_layers(layer)
            all_mha_gains = reduce(torch.mul, (ln['gain'] for ln in relevant_layers))
            all_mha_std = reduce(torch.mul, (ln['std'] for ln in relevant_layers))
            # to get the original raw hidden states, we'd need:
            mha_term = (weights.unsqueeze(-1) * unweighted_outputs.unsqueeze(-3)).sum(1).sum(-2)
            # so we have to divde the bias term by the number of items that go through the linear projections.
            # weight_applied = (weights.unsqueeze(-1) * unweighted_outputs.unsqueeze(-3))
            # mha_terms_unbound = [
            #     [
            #         attended_tok + (output_bias_correction / (weight_applied.size(1) * weight_applied.size(-2)))
            #         for attended_tok in all_in_head.unbind(-2)
            #     ]
            #     for all_in_head in weight_applied.unbind(dim=1)
            # ]
            # mha_terms_unbound = [t / len(mha_terms_unbound)) for t in mha_terms_unbound]
            renormalized_mha_layers.append((all_mha_gains * (mha_term / all_mha_std)).squeeze(0))
            # this will be considered in the 'norm' section below
            all_output_bias_corrections.append(all_mha_gains * (output_bias_correction / all_mha_std))
        # n_heads = len(renormalized_mha_layers[0])
        # mha_contribs = [
        #     sum(layer[h_idx] for layer in renormalized_mha_layers)
        #     for h_idx in range(n_heads)
        # ]
        mha_contrib = sum(renormalized_mha_layers)

        # 4. correction trace
        all_corrective_terms = []
        flat_ln_terms = [ln for layer in ln_terms for ln in layer]
        bias_relevant = lambda layer: flat_ln_terms[layer+1:]
        mean_relevant = lambda layer: flat_ln_terms[layer:]
        for layer, ln_term in enumerate(flat_ln_terms):
            all_bias_relevant = bias_relevant(layer)
            if all_bias_relevant:
                all_bias_relevant_gains = reduce(torch.mul, (ln['gain'] for ln in all_bias_relevant))
                all_bias_relevant_std = reduce(torch.mul, (ln['std'] for ln in all_bias_relevant))
                bias_correction = all_bias_relevant_gains * (ln_term['bias'] / all_bias_relevant_std)
            else:
                bias_correction = ln_term['bias']
            all_mean_relevant = mean_relevant(layer)
            all_mean_relevant_gains = reduce(torch.mul, (ln['gain'] for ln in all_mean_relevant))
            all_mean_relevant_std = reduce(torch.mul, (ln['std'] for ln in all_mean_relevant))
            mean_correction = all_mean_relevant_gains  * (ln_term['mean']/ all_mean_relevant_std)
            all_corrective_terms.append((bias_correction - mean_correction).squeeze(0))
        correct_contrib = sum(all_corrective_terms) + sum(all_output_bias_corrections).squeeze(0) + sum(all_ff_biases).squeeze(0)
        all_factors = {
            'ipt':ipt_contrib,
            'ff':ff_contrib,
            'mha':mha_contrib,
            'norm':correct_contrib,
        }
        all_factors_at_every_layer.append(all_factors)
    return all_factors_at_every_layer


def relative_dot_product(target_total, contribution):
    return (target_total.dot(contribution) / (torch.linalg.norm(target_total) ** 2)).item()

@torch.no_grad()
def tally_contributions(sentence):
    inputs = tokenizer([sentence], return_tensors="pt")
    attested, keywords = run_bert_model(model, **inputs, output_hidden_states=True)
    target_total = attested.last_hidden_state.squeeze(0)
    attested_hidden_states = [h.squeeze(0) for h in attested.hidden_states]
    all_factors_at_every_layer = get_factors(keywords)
    evol_by_layer = []

    layer_0_attested = attested_hidden_states[0]
    all_factors_0 = all_factors_at_every_layer[0]

    evol_by_layer.append(
        [
            {
                'tok':tokenizer.decode(inputs.input_ids[0,tokpos]),
                'pos': tokpos,
                'ipt': relative_dot_product(layer_0_attested[tokpos], all_factors_0['ipt'][tokpos]),
                'norm': relative_dot_product(layer_0_attested[tokpos], all_factors_0['norm'][tokpos]),
            }
            for tokpos in range(layer_0_attested.size(0))
        ]
    )
    for target, layer in zip(attested_hidden_states[1:], all_factors_at_every_layer[1:]):
        evol_by_layer.append(
            [
                {
                    'tok':tokenizer.decode(inputs.input_ids[0,tokpos]),
                    'pos': tokpos,
                    'ipt': relative_dot_product(target[tokpos], layer['ipt'][tokpos]),
                    'norm': [
                        relative_dot_product(target[tokpos], prev_layer[tokpos])
                        for prev_layer in layer['norm']
                    ],
                    'ff': [
                        relative_dot_product(target[tokpos], prev_layer[tokpos])
                        for prev_layer in layer['ff']
                    ],
                    'heads': [
                        [
                            [
                                relative_dot_product(target[tokpos], head[tokpos][attended])
                                for attended in range(len(head[tokpos]))
                            ]
                            for head in prev_layer
                        ]
                        for prev_layer in layer['mha']
                    ],
                }
                for tokpos in range(target.size(0))
            ]
        )

    # print(f"sentence: ``{sentence}''")
    # for i in range(target_total.size(0)):
    #     ipt_factor = relative_dot_product(target_total[i], ipt_contrib[i])
    #     ff_factor = relative_dot_product(target_total[i], ff_contrib[i])
    #     mha_factor = relative_dot_product(target_total[i], mha_contrib[i])
    #     norm_factor = relative_dot_product(target_total[i], norm_contrib[i])
    #     print(f"position {i}, token ``{tokenizer.decode(inputs.input_ids[0,i])}'':\n\tipt: {ipt_factor}, ff: {ff_factor}, mha: {mha_factor}, norm: {norm_factor}")
    # import pprint
    # pprint.pprint(evol_by_layer)
    return evol_by_layer


@torch.no_grad()
def get_factors_last_layer(keywords, up_to_layer=12):
    ln_terms = extract_layer_norm_terms(keywords)
    ipt_terms = extract_input_terms(keywords)
    ff_terms = extract_feedforward_terms(keywords)
    mha_terms = extract_attention_terms(keywords)

    # ln_terms = LN_TERMS[:LAYER_NUMBER+2]
    # ipt_terms = IPT_TERMS
    # ff_terms = FF_TERMS[:LAYER_NUMBER+1]
    # mha_terms = MHA_TERMS[:LAYER_NUMBER+1]

    # 1. input trace
    all_gains = reduce(torch.mul, (ln['gain'] for layer in ln_terms for ln in layer))
    all_std = reduce(torch.mul, (ln['std'] for layer in ln_terms for ln in layer))
    ipt_contrib = (all_gains * (sum(ipt_terms) / all_std)).squeeze(0)

    # 2. feed forward trace
    renormalized_ff_layers = []
    all_ff_biases = []
    relevant_ff_layers = lambda layer: [ln_terms[layer][-1]]+[ln for lyr in ln_terms[layer+1:] for ln in lyr]
    for layer, (ff_term, ff_bias) in enumerate(ff_terms, start=1):
        relevant_layers = relevant_ff_layers(layer)
        all_ff_gains = reduce(torch.mul, (ln['gain'] for ln in relevant_layers))
        all_ff_std = reduce(torch.mul, (ln['std'] for ln in relevant_layers))
        renormalized_ff_layers.append((all_ff_gains * ((ff_term - ff_bias) / all_ff_std)).squeeze(0))
        all_ff_biases.append(all_ff_gains * (ff_bias / all_ff_std))
    ff_contrib = sum(renormalized_ff_layers)

    # 3. multi-head attention trace
    renormalized_mha_layers = []
    all_output_bias_corrections = []
    relevant_mha_layers = lambda layer: [ln for lyr in ln_terms[layer:] for ln in lyr]
    for layer, (weights, unweighted_outputs, output_bias_correction) in enumerate(mha_terms, start=1):
        relevant_layers = relevant_mha_layers(layer)
        all_mha_gains = reduce(torch.mul, (ln['gain'] for ln in relevant_layers))
        all_mha_std = reduce(torch.mul, (ln['std'] for ln in relevant_layers))
        # to get the original raw hidden states, we'd need:
        # self_keywords['weights'].unsqueeze(-1) * unweighted_outputs.unsqueeze(-3)).sum(1).sum(-2) + self.output.dense.bias
        # so we have to divde the bias term by the number of items that go through the linear projections. Yay.
        mha_term = (weights.unsqueeze(-1) * unweighted_outputs.unsqueeze(-3)).sum(1).sum(-2)
        # mha_terms_unbound = [
        #     [
        #         attended_tok + (output_bias_correction / (weight_applied.size(1) * weight_applied.size(-2)))
        #         for attended_tok in all_in_head.unbind(-2)
        #     ]
        #     for all_in_head in weight_applied.unbind(dim=1)
        # ]
        # mha_terms_unbound = [t / len(mha_terms_unbound)) for t in mha_terms_unbound]
        renormalized_mha_layers.append((all_mha_gains * (mha_term / all_mha_std)).squeeze(0))
        all_output_bias_corrections.append(all_mha_gains * (output_bias_correction / all_mha_std))
    # n_heads = len(renormalized_mha_layers[0])
    # mha_contribs = [
    #     sum(layer[h_idx] for layer in renormalized_mha_layers)
    #     for h_idx in range(n_heads)
    # ]
    mha_contribs = sum(renormalized_mha_layers)

    # 4. correction trace
    all_corrective_terms = []
    flat_ln_terms = [ln for layer in ln_terms for ln in layer]
    bias_relevant = lambda layer: flat_ln_terms[layer+1:]
    mean_relevant = lambda layer: flat_ln_terms[layer:]
    for layer, ln_term in enumerate(flat_ln_terms):
        all_bias_relevant = bias_relevant(layer)
        if all_bias_relevant:
            all_bias_relevant_gains = reduce(torch.mul, (ln['gain'] for ln in all_bias_relevant))
            all_bias_relevant_std = reduce(torch.mul, (ln['std'] for ln in all_bias_relevant))
            bias_correction = all_bias_relevant_gains * (ln_term['bias'] / all_bias_relevant_std)
        else:
            bias_correction = ln_term['bias']
        all_mean_relevant = mean_relevant(layer)
        all_mean_relevant_gains = reduce(torch.mul, (ln['gain'] for ln in all_mean_relevant))
        all_mean_relevant_std = reduce(torch.mul, (ln['std'] for ln in all_mean_relevant))
        mean_correction = all_mean_relevant_gains  * (ln_term['mean']/ all_mean_relevant_std)
        all_corrective_terms.append((bias_correction - mean_correction).squeeze(0))
    correct_contrib = sum(all_corrective_terms) + sum(all_output_bias_corrections).squeeze(0) + sum(all_ff_biases).squeeze(0)
    all_factors = {
        'ipt':ipt_contrib,
        'ff':ff_contrib,
        'mha':mha_contribs,
        'norm':correct_contrib,
    }
    return all_factors

@torch.no_grad()
def tally_contributions_last_layer(sentence):
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs['offset_mapping'].squeeze(0)
    del inputs['offset_mapping']
    attested, keywords = run_bert_model(model, **inputs, output_hidden_states=True)
    target = attested.last_hidden_state.squeeze(0)
    layer = get_factors_last_layer(keywords)
    all_contribs = [
        {
            'tok':tokenizer.decode(inputs.input_ids[0,tokpos]),
            'pos': tokpos,
            'start_idx':offset_mapping[tokpos][0].item(),
            'end_idx':offset_mapping[tokpos][1].item(),
            'ipt': relative_dot_product(target[tokpos], layer['ipt'][tokpos]),
            'norm': relative_dot_product(target[tokpos], layer['norm'][tokpos]),
            'ff': relative_dot_product(target[tokpos], layer['ff'][tokpos]),
            'mha': relative_dot_product(target[tokpos], layer['mha'][tokpos]),
        }
        for tokpos in range(target.size(0))
    ]
    return all_contribs

@torch.no_grad()
def read_factors_last_layer(sentence, model_=model, tokenizer_=tokenizer):
    inputs = tokenizer_([sentence], return_tensors="pt", truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs['offset_mapping'].squeeze(0)
    del inputs['offset_mapping']
    attested, keywords = run_bert_model(model_, **inputs, output_hidden_states=True)
    target = attested.last_hidden_state.squeeze(0)
    layer = get_factors_last_layer(keywords)
    all_factors = [
        {
            'tok':tokenizer.decode(inputs.input_ids[0,tokpos]),
            'pos': tokpos,
            'start_idx':offset_mapping[tokpos][0].item(),
            'end_idx':offset_mapping[tokpos][1].item(),
            'ipt': layer['ipt'][tokpos],
            'norm': layer['norm'][tokpos],
            'ff': layer['ff'][tokpos],
            'mha': layer['mha'][tokpos],
        }
        for tokpos in range(target.size(0))
    ]
    return all_factors


@torch.no_grad()
def tally_ipt_contrib_accross_layers(sentence):
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
    attested, keywords = run_bert_model(model, **inputs, output_hidden_states=True)
    targets = [h.squeeze(0) for h in attested.hidden_states]
    ln_terms = extract_layer_norm_terms(keywords)
    ipt_terms = extract_input_terms(keywords)
    accross_layers = [
        (ln_terms[0][0]['gain'] * (sum(ipt_terms) / ln_terms[0][0]['std'])).squeeze(0),
    ]
    for LAYER_NUMBER in range(len(targets) - 1):
        ln_terms_ = ln_terms[:LAYER_NUMBER+2]

        all_gains = reduce(torch.mul, (ln['gain'] for layer in ln_terms_ for ln in layer))
        all_std = reduce(torch.mul, (ln['std'] for layer in ln_terms_ for ln in layer))
        accross_layers.append((all_gains * (sum(ipt_terms) / all_std)).squeeze(0))

    return [
        [
            relative_dot_product(target[pos], layer_ipt_contrib[pos])
            for target, layer_ipt_contrib in zip(targets, accross_layers)
        ]
        for pos in range(targets[0].size(0))
    ]



@torch.no_grad()
def tally_contributions_coarse_mha(sentence, pickle_file=None):
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
    attested, keywords = run_bert_model(model, **inputs, output_hidden_states=True)
    target_total = attested.last_hidden_state.squeeze(0)
    attested_hidden_states = [h.squeeze(0) for h in attested.hidden_states]
    all_factors_at_every_layer = get_factors_coarse_mha(keywords)
    evol_by_layer = []

    layer_0_attested = attested_hidden_states[0]
    all_factors_0 = all_factors_at_every_layer[0]

    evol_by_layer.append(
        [
            {
                'tok':tokenizer.decode(inputs.input_ids[0,tokpos]),
                'pos': tokpos,
                'ipt': relative_dot_product(layer_0_attested[tokpos], all_factors_0['ipt'][tokpos]),
                'norm': relative_dot_product(layer_0_attested[tokpos], all_factors_0['norm'][tokpos]),
                'ff':0,
                'mha':0,
            }
            for tokpos in range(layer_0_attested.size(0))
        ]
    )
    if pickle_file:
        pickle.dump(sum(all_factors_0.values()), pickle_file)
        pickle.dump(None, pickle_file)
    for target, layer in zip(attested_hidden_states[1:], all_factors_at_every_layer[1:]):
        evol_by_layer.append(
            [
                {
                    'tok':tokenizer.decode(inputs.input_ids[0,tokpos]),
                    'pos': tokpos,
                    'ipt': relative_dot_product(target[tokpos], layer['ipt'][tokpos]),
                    'norm': relative_dot_product(target[tokpos], layer['norm'][tokpos]),
                    'ff': relative_dot_product(target[tokpos], layer['ff'][tokpos]),
                    'mha': relative_dot_product(target[tokpos], layer['mha'][tokpos]),
                }
                for tokpos in range(target.size(0))
            ]
        )

        if pickle_file:
            pickle.dump(sum(layer.values()), pickle_file)
            pickle.dump(layer['ff'], pickle_file)


    # print(f"sentence: ``{sentence}''")
    # for i in range(target_total.size(0)):
    #     ipt_factor = relative_dot_product(target_total[i], ipt_contrib[i])
    #     ff_factor = relative_dot_product(target_total[i], ff_contrib[i])
    #     mha_factor = relative_dot_product(target_total[i], mha_contrib[i])
    #     norm_factor = relative_dot_product(target_total[i], norm_contrib[i])
    #     print(f"position {i}, token ``{tokenizer.decode(inputs.input_ids[0,i])}'':\n\tipt: {ipt_factor}, ff: {ff_factor}, mha: {mha_factor}, norm: {norm_factor}")
    # import pprint
    # pprint.pprint(evol_by_layer)
    return evol_by_layer



PATH_TO_EUROPARL = "../data/europarl/europarl-sample.txt"
with open(PATH_TO_EUROPARL, 'r') as istr:
    data = map(str.strip, istr)
    data = sorted(data, key=len, reverse=True)

import tqdm
import numpy as np
import multiprocessing as mp
#

def do_get_factors(sentence):
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
    _, keywords = run_bert_model(model, **inputs)
    for layer_idx, factor_group in tqdm.tqdm(enumerate(get_factors(keywords)), total=13, leave=False):
        yield {
            'sentence': sentence,
            'layer':layer_idx,
            'factors': factor_group
        }


# calls = map(do_get_factors, data)
# calls = tqdm.tqdm(calls, total=len(data))
# for factors in calls:
#     for factor_group in factors:
#         pickle.dump(factor_group, pickle_file)

def do_tally(sentence):
    return tally_contributions_last_layer(sentence)
    # inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
    # attested, keywords = run_bert_model(model, **inputs, output_hidden_states=True)
    # return None
    # return tally_contributions_coarse_mha(sentence, pickle_file)

MODE = "prop-across-layers"
# MODE = "last-layer-mlm-perf"

import joblib

if __name__ == "__main__" and MODE == 'prop-across-layers':
    print(MODE)
    def do_tally(sentence):
        return tally_contributions_coarse_mha(sentence)
    ipt, norm, mha, ff = [], [], [], []
    calls = map(do_tally, data)
    for contribs in tqdm.tqdm(calls, total=len(data)):
        for token in range(len(contribs[0])):
            ipt.append([layer[token]['ipt'] for layer in contribs])
            norm.append([layer[token]['norm'] for layer in contribs])
            mha.append([layer[token]['mha'] for layer in contribs])
            ff.append([layer[token]['ff'] for layer in contribs])
    ipt, norm, mha, ff = map(np.array, (ipt, norm, mha, ff))
    for layer_idx in range(ipt.shape[1]):
        print(f"layer {layer_idx}:")
        print(f"\tipt: {ipt[:,layer_idx].mean()} +- {ipt[:,layer_idx].std()}")
        print(f"\tnorm: {norm[:,layer_idx].mean()} +- {norm[:,layer_idx].std()}")
        print(f"\tff: {ff[:,layer_idx].mean()} +- {ff[:,layer_idx].std()}")
        print(f"\tmha: {mha[:,layer_idx].mean()} +- {mha[:,layer_idx].std()}")
    model_id = MODEL_NAME.split('/')[-1]
    joblib.dump(ipt, f"ipt_{model_id}.npy")
    joblib.dump(norm, f"norm_{model_id}.npy")
    joblib.dump(mha, f"mha_{model_id}.npy")
    joblib.dump(ff, f"ff_{model_id}.npy")
    pickle_file_.close()


if __name__ == '__main__' and MODE == 'last-layer-mlm-perf':
    print(MODE)
    torch.set_grad_enabled(False)
    import itertools

    KEYS_TO_SUM = 'ipt', 'mha', 'norm', 'ff'

    def powerset():
        combinations = (
            itertools.combinations(KEYS_TO_SUM, r)
            for r in range(1, len(KEYS_TO_SUM) + 1)
        )
        yield from itertools.chain.from_iterable(combinations)

    from transformers import AutoModelWithLMHead
    model_lm = AutoModelWithLMHead.from_pretrained(MODEL_NAME)
    model_lm.eval()
    first_true_idx = max(v for k,v in tokenizer.vocab.items() if k.startswith('[unused')) + 1

    def mask_me(inputs):
        rd_sample = torch.rand(inputs.input_ids.size())
        sampled_mask = rd_sample <= 0.15
        target_ids = inputs.input_ids
        random_wordpiece = torch.randint(first_true_idx, tokenizer.vocab_size, inputs.input_ids.size())
        inputs['input_ids'] = inputs['input_ids'].masked_fill(rd_sample <= 0.135, 0) + random_wordpiece.masked_fill(rd_sample > 0.135, 0)
        inputs['input_ids'] = inputs['input_ids'].masked_fill(rd_sample <= 0.12, tokenizer.mask_token_id).detach()
        return target_ids, sampled_mask, inputs

    def pred_lm(inputs):
        _, keywords = run_bert_model(model_lm.bert, **inputs)
        factors = get_factors_last_layer(keywords)
        pred_from = lambda *keys: model_lm.cls(sum(factors[k] for k in keys).unsqueeze(0))
        return {keys: pred_from(*keys) for keys in powerset()}

    n_items = 0
    import collections, tqdm
    import torch.nn.functional as F

    running_accs = collections.defaultdict(int)
    running_kldivs = collections.defaultdict(int)
    relevant_keys = set(powerset())
    for sentence in tqdm.tqdm(data):
        inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
        target_ids, sampled_mask, inputs = mask_me(inputs)
        targets_only = target_ids.masked_select(sampled_mask)
        dict_obs = pred_lm(inputs)
        n_items += targets_only.numel()
        for k in relevant_keys:
            # running_accs[k] += (dict_obs[k].argmax(-1).view(-1) == inputs.input_ids.view(-1)).sum().item()
            running_accs[k] += (dict_obs[k].argmax(-1).masked_select(sampled_mask) == targets_only).sum().item()
            # running_kldivs[k] += F.cross_entropy(dict_obs[k].squeeze(0), inputs.input_ids.view(-1), reduction='sum').item()
            reps_for_targets = dict_obs[k].masked_select(sampled_mask.unsqueeze(-1)).view(targets_only.numel(), dict_obs[k].size(-1))
            running_kldivs[k] += F.cross_entropy(reps_for_targets, targets_only, reduction='sum').item()
    print("\nraw accs")
    for k in running_accs:
        print(f"{'+'.join(k)}\t{running_accs[k] / n_items}")
    print("\nXent")
    for k in sorted(running_kldivs):
        print(f"{'+'.join(k)}\t{running_kldivs[k] / n_items}")
# if __name__ == "__main__":
#     ipt, norm, mha, ff = [], [], [], []
#     calls = map(do_tally, data
#     calls = tqdm.tqdm(calls, total=len(data))
#     for contribs in calls:
#         pass
        # for token in range(len(contribs[0])):
        #     ipt.append([layer[token]['ipt'] for layer in contribs])
        #     norm.append([layer[token]['norm'] for layer in contribs])
        #     mha.append([layer[token]['mha'] for layer in contribs])
        #     ff.append([layer[token]['ff'] for layer in contribs])

    # pickle_file.close()
#
# ipt, norm, mha, ff = map(np.array, (ipt, norm, mha, ff))
#
# print("\nEuroparl (EN), total:")
# for layer_idx in range(ipt.shape[1]):
#     print(f"layer {layer_idx}:")
#     print(f"\tipt: {ipt[:,layer_idx].mean()} +- {ipt[:,layer_idx].std()}")
#     print(f"\tnorm: {norm[:,layer_idx].mean()} +- {norm[:,layer_idx].std()}")
#     print(f"\tff: {ff[:,layer_idx].mean()} +- {ff[:,layer_idx].std()}")
#     print(f"\tmha: {mha[:,layer_idx].mean()} +- {mha[:,layer_idx].std()}")


# print()
# ipt, norm, mha, ff = [], [], [], []
# calls = map(tally_contributions_last_layer, data)
# calls = tqdm.tqdm(calls, total=len(data))
# for contribs in calls:
#     for token in contribs:
#         ipt.append(token['ipt'])
#         norm.append(token['norm'])
#         mha.append(token['mha'])
#         ff.append(token['ff'])
#
# ipt, norm, mha, ff = map(np.array, (ipt, norm, mha, ff))
#
# print("\nEuroparl (EN), overall:")
# print(f"ipt: {ipt.mean()} +- {ipt.std()}")
# print(f"mha: {mha.mean()} +- {mha.std()}")
# print(f"ff: {ff.mean()} +- {ff.std()}")
# print(f"norm: {norm.mean()} +- {norm.std()}")
# #
# print("Type in a sentence you'd like to query! CTRL+C to quit.")
# while True:
#     try:
#         sentence = input('> ')
#         tally_contributions(sentence)
#     except KeyboardInterrupt:
#         print("\nGoodbye!")
#         break
