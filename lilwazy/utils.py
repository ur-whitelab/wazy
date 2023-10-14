import numpy as np
import random
import jax
import jax.numpy as jnp
from functools import partial
import selfies as sf
from transformers import PreTrainedTokenizerFast, FlaxBartModel, FlaxBartForCausalLM, FlaxBertModel
import flax.linen as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertEncoder, FlaxBertPooler, FlaxBaseModelOutputWithPoolingAndCrossAttentions
from typing import *
from dataclasses import dataclass
from flax.core.frozen_dict import unfreeze, freeze

ALPHABET_robust_sf = sf.get_semantic_robust_alphabet() #Get the alphabet of robut symbols

A2U = np.zeros((len(ALPHABET), len(ALPHABET_Unirep)))
for i, s in enumerate(ALPHABET):
    A2U[i, ALPHABET_Unirep.index(s)] = 1
start_vec = np.zeros((1, 26))
start_vec[0, ALPHABET_Unirep.index("start")] = 1


def decode_useq(s):
    indices = jnp.argmax(s, axis=1)
    return [ALPHABET_Unirep[int(i)] for i in indices][1:]


def decode_seq(s):
    indices = jnp.argmax(s, axis=1)
    return [ALPHABET_robust_sf[i] for i in indices]


def encode_seq(s):  # s is a list
    e = np.zeros((len(s), len(ALPHABET_robust_sf)))
    e[np.arange(len(s)), [ALPHABET.index(si) for si in s]] = 1
    return e




@dataclass
class ConfigBert:
    hidden_size : int         = 512
    num_hidden_layers : int   = 6
    num_attention_heads : int = 8
    intermediate_size : int   = 256
    hidden_act : str          = 'gelu'

c = ConfigBert()
config = BertConfig(
    vocab_size = tokenizer.vocab_size,
    hidden_size = c.hidden_size,
    num_hidden_layers = c.num_hidden_layers,
    num_attention_heads = c.num_attention_heads,
    intermediate_size = c.intermediate_size,
    hidden_act = c.hidden_act,
    max_position_embeddings = 512,
    #hidden_dropout_prob = 0.1,
    # attention_probs_dropout_prob = 0.1,
    pad_token_id = tokenizer.pad_token_id,
)

class DiffFlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Dense(features=self.config.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, ohc, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(ohc)
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)        
        return hidden_states


class DiffFlaxBertModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = DiffFlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBertEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        ohc,
        attention_mask,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # TODO:? rewrite these to use number of columns?
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(attention_mask)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(attention_mask).shape[-1]), attention_mask.shape)

        hidden_states = self.embeddings(
            ohc, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]
        
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )




def jax_bert_config():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("maykcaldas/selfies-bart")
    tokenizer.tokenize_sfs = lambda batch: tokenizer([x.replace( "][", "] [" ) for x in batch],
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors="np")
    trained_model = FlaxBertModel.from_pretrained('maykcaldas/selfies-bart')
    # insert params from trained model to diff model
    diff_model = DiffFlaxBertModule(config)
    # modify key order of tokens(from trained_model) to match the input order of diff_model
    order = ['ohc'] +list(tokens.keys()) + ['position_ids']
    tokens['ohc'] = jax.nn.one_hot(tokens['input_ids'], tokenizer.vocab_size)
    x = dict(**tokens, position_ids = np.arange(len(token['input_ids'])))
    x = {k: x[k] for k in order}
    del x['input_ids']
    
    diff_params = {}
    diff_params['params'] = {}
    diff_params['params']['encoder'] = trained_model.params['encoder']
    diff_params['params']['pooler'] = trained_model.params['pooler']
    diff_params['params']['embeddings'] = {}
    diff_params['params']['embeddings']['word_embeddings'] = {}
    diff_params['params']['embeddings']['word_embeddings']['kernel'] = trained_model.params['embeddings']['word_embeddings']['embedding']
    diff_params['params']['embeddings']['position_embeddings'] = trained_model.params['embeddings']['position_embeddings']
    diff_params['params']['embeddings']['token_type_embeddings'] = trained_model.params['embeddings']['token_type_embeddings']
    diff_params['params']['embeddings']['LayerNorm'] = trained_model.params['embeddings']['LayerNorm']
    diff_params = freeze(diff_params)
    return diff_model, diff_params


def differentiable_jax_bert(model, params, ohc):

    #input dict {'ohc','token_type_ids','attention_mask', 'position_ids'}
    input_dict = {'ohc': ohc, 'token_type_ids': np.zeros(ohc.shape()[0]), 'attention_mask': np.ones(ohc.shape()[0]), 'position_ids': np.arange(phc.shape()[0])}
    out = model.apply(params, input_dict)
    reps = out['last_hidden_state']
    return reps


def resample(key, y, output_shape, nclasses=10):
    """
    Resample the given y-vector to have a uniform classes,
    where the classes are chosen via histogramming y.
    """
    if type(output_shape) is int:
        output_shape = (output_shape,)
    if len(y.shape) == 1:
        # regression
        _, bins = np.histogram(y, bins=nclasses)
        classes = np.digitize(y, bins)
    elif len(y.shape) == 2:
        # classification
        classes = np.argmax(y, axis=1)
        nclasses = y.shape[1]
    else:
        raise ValueError("y must rank 1 or 2")
    uc = np.unique(classes)
    nclasses = uc.shape[0]
    if nclasses == 1:
        return jax.random.choice(key, np.arange(y.shape[0]), shape=output_shape)
    idx = [np.where(classes == uc[i])[0] for i in range(nclasses)]
    c = jax.random.choice(key, np.arange(nclasses), shape=output_shape)
    keys = jax.random.split(key, nclasses)
    f = np.vectorize(lambda i: jax.random.choice(keys[i], idx[i]))
    return f(c)


def transform_var(s):
    # heuristic to make MLP output better behaved.
    return jax.nn.softplus(s) + 1e-6
