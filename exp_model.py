import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import re
from PIL import Image
import ast
from tqdm import tqdm
from typing import Optional
from datetime import datetime
import torchvision.models as models
import os.path

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
# from rouge_score.rouge_scorer import RougeScorer
from torchvision import transforms
# from vit_pytorch import ViT  
# from vit_pytorch.extractor import Extractor 
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, classification_report
# from vit_pytorch.regionvit import RegionViT 

from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pytorch_lightning as pl
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset
from img_transformer import ImageTransformerEncoder

# import clip
torch.cuda.set_device(0)
# torch.cuda.empty_cache()
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

# TARGET_COLUMN = 'code_mixed_explanation'
# TEXT_INPUT_PATH = '../Data/Text/'
# ACOUSTIC_INPUT_PATH = '../Data/Audio/'
# VISUAL_INPUT_PATH = '../Data/Video/'

# from sklearn.model_selection import train_test_split
# X_train, X_rem = train_test_split(data, train_size=0.7)
# X_valid, X_test = train_test_split(X_rem, test_size=0.5)
# X_train.reset_index(drop =True,inplace=True)
# X_valid.reset_index(drop =True,inplace=True)
# X_test.reset_index(drop =True,inplace=True)

MODEL_OUTPUT_DIR = r'/MAFsaved/'
RESULT_OUTPUT_DIR = r'/MAFsavedRes/'

path_to_images = r'/Images_naacl/'

path_to_train = r'Aspect_train_electronics.csv'

path_to_val = r'Aspect_valid_electronics_span.csv'

path_to_test = r'Aspect_test_electronics_span.csv'

LOWERCASE_UTTERANCES = False
UNFOLDED_DIALOGUE = True

if UNFOLDED_DIALOGUE:
    SOURCE_COLUMN = 'dialogue'
else:
    SOURCE_COLUMN_1 = 'target'
    SOURCE_COLUMN_2 = 'context'
    


SOURCE_MAX_LEN = 1024
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25

ACOUSTIC_DIM = 154
ACOUSTIC_MAX_LEN = 600

VISUAL_DIM = 2048
VISUAL_MAX_LEN = 49

BATCH_SIZE = 1
MAX_EPOCHS = 100

BASE_LEARNING_RATE = 3e-5 
NEW_LEARNING_RATE = 3e-5 
# BASE_LEARNING_RATE=7e-5
# NEW_LEARNING_RATE=7e-5
WEIGHT_DECAY = 1e-4

NUM_BEAMS = 4
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3

EARLY_STOPPING_THRESHOLD = 5
best_as1=-1
best_as2=-1
best_as3=-1
best_as4=-1
best_as5=-1
best_as6=-1
best_acc_cmp=-1
best_acc_em = -1
best_acc_se =-1
best_acc_sev =-1
best_bleu=-1
best_iouf1 = -1
best_tokenf1 = -1
best_jacc = -1
best_indi1 = -1
best_indi2 = -1
best_indi3 = -1
best_indi4 = -1
best_comp = -1
best_sarc = -1
best_emo = -1
best_senti = -1
best_sev = -1
def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)


# -------------------------------------------------------------- MODEL -------------------------------------------------------------- #


import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    BartTokenizerFast,
    AdamW
)

from transformers.models.bart.configuration_bart import BartConfig

from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    shift_tokens_right,
    _make_causal_mask,
    _expand_mask 
)


from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)


from transformer_encoder import TransformerEncoder

class MSEDataset(Dataset):
    def __init__(self, path_to_data_df, path_to_images, tokenizer, image_transform):
        self.data = pd.read_csv(path_to_data_df)
        # self.data = self.data.iloc[1:]   

        self.path_to_images = path_to_images
        self.tokenizer = tokenizer
        self.image_transform = image_transform
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        # print('*(((((((((((((((((((((((((((')
        # print(row)
        # time.sleep(120)


        image_name = row['Image_urls']
        src_text = str(row['Review_S'])
        # print(src_text)
        # print(type(src_text))
        target_text = str(row['exp_all'])

        max_length = 256
        encoded_dict = tokenizer.encode_plus(
            text=src_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            add_prefix_space = True
        )
        # print()
        src_ids = encoded_dict['input_ids'][0]
        src_mask = encoded_dict['attention_mask'][0]

        image_path = os.path.join(self.path_to_images, image_name)
        img = np.array(Image.open(image_path).convert('RGB'))
        img_inp = self.image_transform(img)
        

        encoded_dict = tokenizer(
          target_text,
          max_length=max_length,
          padding="max_length",
          truncation=True,
          return_tensors='pt',
          add_prefix_space = True
        )

        target_ids = encoded_dict['input_ids'][0]

        sample = {
            "input_ids": src_ids,
            "attention_mask": src_mask,
            "input_image": img_inp,
            "target_ids": target_ids,
        }
        return sample
    
    def __len__(self):
        return self.data.shape[0]

class MSEDataModule(pl.LightningDataModule):
    def __init__(self, path_to_train_df, path_to_val_df, path_to_test_df, path_to_images, tokenizer, image_transform, batch_size=1):
        super(MSEDataModule, self).__init__()
        self.path_to_train_df = path_to_train_df
        self.path_to_val_df = path_to_val_df
        self.path_to_test_df = path_to_test_df
        self.path_to_images = path_to_images
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_transform = image_transform
  
    def setup(self, stage=None):
        self.train_dataset = MSEDataset(self.path_to_train_df, self.path_to_images, self.tokenizer, self.image_transform)
        self.val_dataset = MSEDataset(self.path_to_val_df, self.path_to_images, self.tokenizer, self.image_transform)
        self.test_dataset = MSEDataset(self.path_to_test_df, self.path_to_images, self.tokenizer, self.image_transform)
  
    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler = RandomSampler(self.train_dataset), batch_size = self.batch_size)
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)
  
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1)


class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()
        
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, 
                                                     num_heads=1, 
                                                     dropout=self.dropout_rate, 
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=DEVICE)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)
        




    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):
        
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output



class MAF(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF, self).__init__()
        self.dropout_rate = dropout_rate
        
        # self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.visual_context_transform = nn.Linear(49, SOURCE_MAX_LEN, bias=False)
        
        # self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
        #                                                         dim_context=ACOUSTIC_DIM,
        #                                                         dropout_rate=dropout_rate)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=512,
                                                              dropout_rate=dropout_rate)        
        # self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None):
                    
        # # Audio as Context for Attention
        # acoustic_context = acoustic_context.permute(0, 2, 1)
        # acoustic_context = self.acoustic_context_transform(acoustic_context)
        # acoustic_context = acoustic_context.permute(0, 2, 1)
        
        # audio_out = self.acoustic_context_attention(q=text_input,
        #                                             k=text_input,
        #                                             v=text_input,
        #                                             context=acoustic_context)
        
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        # print(visual_context.shape)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)
        
        # Global Information Fusion Mechanism
        # weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input +
                                       # weight_a * audio_out + 
                                    weight_v * video_out)

        return output



class MultimodalBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False
        
        # ================================ Modifications ================================ #
        self.fusion_at_layer = [7]
        # 7
        # self.clipmodel, self.clippreprocess = clip.load("ViT-B/32", device=DEVICE)
        # self.vgg=models.vgg19_bn(pretrained=True)
        # self.image_encoder = list(self.vgg.children())[0]
        # self.img_transformer = ImageTransformerEncoder(d_model=224, num_layers=4, num_heads=8, dim_feedforward=224)
        # self.v = ViT(
        #     image_size = 224,
        #     patch_size = 32,
        #     num_classes = 2,
        #     dim = 1024,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # self.Rmodel = RegionViT(
        #     dim = (224, 256, 256,512),      # tuple of size 4, indicating dimension at each stage
        #     depth = (2, 8, 8, 2),           # depth of the region to local transformer at each stage
        #     window_size = 14,                # window size, which should be either 7 or 14
        #     num_classes = 10,             # number of output classes
        #     tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
        #     use_peg = True,                # whether to use positional generating module. they used this for object detection for a boost in performance
        # )
        # self.feat11_lrproj = nn.Linear(512, 256)
        # self.feat12_lrproj = nn.Linear(512, 256)
        # self.feat21_lrproj = nn.Linear(768, 256)
        # self.feat22_lrproj = nn.Linear(224, 256)
        self.tanh = torch.nn.Tanh()
        self.output_net2 = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 768)
        )
        # self.Rmodel = Extractor(self.Rmodel, layer = self.Rmodel.layers[-1][-1])
        # self.clipmodel,self.pre = clip.load("ViT-B/32", device=DEVICE)
        
        # self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM, 
        #                                              n_layers=4,
        #                                              n_heads=8, 
        #                                              d_ff=VISUAL_DIM)
        # self.acoustic_transformer = TransformerEncoder(d_model=ACOUSTIC_DIM, 
        #                                                num_layers=4,
        #                                                num_heads=2, 
        #                                                dim_feedforward=ACOUSTIC_DIM)
        self.MAF_layer = MAF(dim_model=embed_dim,
                             dropout_rate=0.2)
        # =============================================================================== #



    def MLB2(self, feat1, feat2):
        # feat1_lr = self.feat21_lrproj(feat1)
        # # print(feat1_lr.shape)
        # feat2_lr = self.feat22_lrproj(feat2)
        # print(feat2_lr.shape)
        # print(feat2_lr.shape)
        # z = torch.mul(feat1_lr, feat2_lr)
        z=self.MAF_layer(feat1, feat2)
        z_out = self.tanh(z)
        # print(z_out.shape)
        mm_feat = self.output_net2(z_out)
        return mm_feat
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        visual_inf=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_len=None
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed_pos = self.embed_positions(input_shape)
        embed_pos = self.embed_positions(input_ids)            

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            
            # ================================ Modifications ================================ #
            # if idx in self.fusion_at_layer:
            #     # acoustic_input = self.acoustic_transformer(acoustic_input)[-1]
            #     # print(visual_input.shape)
            #     # vgg_image_features = self.image_encoder(visual_inf)
            #     # # print(vgg_image_features.shape)
            #     # # # print('vgg: {}'.format(vgg_image_features.shape))
            #     # # # print('1')
                
            #     # vgg_image_features = vgg_image_features.permute(0, 2, 3, 1)
            #     # vgg_image_features = vgg_image_features.reshape(
            #     #     -1, 
            #     #     vgg_image_features.size()[1]*vgg_image_features.size()[2], 
            #     #     512
            #     #     )

            #     # print(vgg_image_features.shape)

            #     # visual_input = self.img_transformer(visual_input)[-1]
            #     # print(visual_input.shape)
            #     # visual_input = visual_input[None, :]
            #     # print(visual_input.shape)

            #     # ------------------perfect----------------------
            #     # print(visual_inf.shape)

            #     # print(vgg_image_features2.shape)
            #     # print(vgg_image_features.shape)
            #     # self.visual_transformer(visual_input)[-1]


            #     # print(visual_input.shape)
            #     # print('MLB2')
            #     # text_feat=TOKENIZER.batch_decode(input_ids)
            #     # # print(text_feat)
            #     # text_feat=text_feat[0].split('<pad>',1)[0]
            #     # # text_feat=text_feat[:77]
            #     # # print(len(text_feat))
            #     # text_feat = clip.tokenize(text_feat,truncate=True).to(DEVICE)
                

            #     # text_feat=self.clipmodel.encode_text(text_feat)
            #     # visual_input=self.clipmodel.encode_image(visual_inf)
            #     # # print(visual_input.shape)
            #     # visual_input=visual_input.reshape(-1,2,256)
            #     # print(text_feat.shape)
            #     # print(hidden_states.shape)
            #     # print('****************')
            #     # vgg_image_features = self.image_encoder(visual_inf)
            #     # # print(vgg_image_features.shape)
            #     # # # print('vgg: {}'.format(vgg_image_features.shape))
            #     # # # print('1')
                
            #     # vgg_image_features = vgg_image_features.permute(0, 2, 3, 1)
            #     # vgg_image_features = vgg_image_features.reshape(
            #     #     -1, 
            #     #     vgg_image_features.size()[1]*vgg_image_features.size()[2], 
            #     #     512
            #     #     )

            #     image_features = image_features.reshape(
            #         image_features.size()[0]*image_features.size()[1], 
            #         49, 
            #         512
            #         )
        

            #     trys=self.MLB2(hidden_states, image_features)
            #     #------------------perfect----------------------


            #     # hidden_states = self.MAF_layer(text_input=hidden_states,
            #     #                                # acoustic_context=acoustic_input, 
            #     #                                visual_context=vgg_image_features)
            #     hidden_states=trys

            #     #clip-----------
            #     # image_features = self.climodel.encode_image(image_features)
                

            # =============================================================================== #
            
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                                 

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of 
        visual_inf=None,
        # image_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_len=None
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # acoustic_input=acoustic_input,      # New addition of acoustic_input
                visual_input=visual_input,      # New addition of visual_input
                visual_inf=visual_inf,
                # image_features=image_features,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                image_len=image_len
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
class MultimodalBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        visual_inf=None,
        # image_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_len=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            # acoustic_input=acoustic_input,      # New addition of acoustic_input
            visual_input=visual_input,      # New addition of visual_input
            visual_inf=visual_inf,
            # image_features=image_features,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_len=image_len
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data


# src=[]
# label=[]
# visual_feat=[]

def prepare_dataset(text_path,visual_path,image_transform,trainornot):
    data = pd.read_csv(text_path)
    # print(data.head())
    # print(data.head())
    # data = data.iloc[1:]     
    # data = data.dropna()
    # print(data.head())
    # if trainornot==True:
    #     data=data.sample(frac = 0.01)


    path_to_images = visual_path
    src_text=data['Review_S'].tolist()
    labels=data['exp_all'].tolist()
    # pid=data['Image_urls'].tolist()
    image_name =[]
    for i in data["Image_urls"]:
      if(str(i)!='nan'):
        i =i.replace(" ", "")
        i =i.replace("'", "")
        i = (i.split(";")[0])
        image_name.append(i.split("/")[-1])
    # print(pid)
    visual_feat=[]
    for p in image_name:
        image_path = os.path.join(path_to_images, p)
        if os.path.isfile(image_path)==True:
            # print(image_path)
            img = np.array(Image.open(image_path).convert('RGB'))
            # print(img.shape)
            img_inp=img
            img_inp = image_transform(img)
            # print(img_inp.shape)
            visual_feat.append(img_inp)
        else:
            print(image_path)
            img_inp=torch.zeros(3, 224,224)
            visual_feat.append(img_inp)
            # print(torch.zeros(3, 224,224).shape)
            # input('enter')

        
    # image_features=[]
    # for p in image_name:
    #     path_feats=os.path.join('imagefeats3/',str(p.split('.png')[0]+'.npy'))
    #     # print(path_feats)
    #     if os.path.isfile(path_feats)==True:
    #         print(np.load(path_feats).shape)
    #         temp=torch.from_numpy(np.load(path_feats))
    #         print(temp.shape)
    #         image_features.append(temp)
    #     else:
    #         temp=torch.zeros(1, 49,512)
    #         image_features.append(temp)




    print(visual_feat[0].shape)
    df =  pd.DataFrame(list(zip(src_text, labels,visual_feat)),columns=['src_text', 'labels','visual_feat']) 
    df = df.dropna()
    return df



	# row = data.iloc[idx, :]
	# pid_i = row['pid']
	# src_text = str(row['text'])
	# labels=str(row['labels'])
 #    image_path = os.path.join(spath_to_images, pid_i)
 #    img = np.array(Image.open(image_path).convert('RGB'))
 #    img_inp = image_transform(img)
 #    df =  pd.DataFrame(list(zip(src_text, labels)),columns=['src_text', 'labels']) 



# def prepare_dataset(text_path: str,
#                     acosutic_path: str,
#                     visual_path: str,
#                     lowercase_utterances: bool=False,
#                     unfolded_dialogue: bool=True):
#     data = read_json_data(text_path)
    
#     code_mixed_explanation = []   

#     if unfolded_dialogue:
#         dialogue = []
#         for i in range(1, len(data)+1):
#             data_point = data[str(i)]

#             example_target_speaker = str(data_point['target_speaker']).upper()
#             example_target_utterance = str(data_point['target_utterance'])

#             example_dialogue = "[CONTEXT] "
#             for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
#                 example_dialogue  = example_dialogue + str(speaker).upper() + " : " +  str(utterance) + " | "

#             example_dialogue = example_dialogue + " [TARGET] " + example_target_speaker + " : " +  example_target_utterance + " | "
#             example_dialogue = re.sub(' +', ' ', example_dialogue)
#             dialogue.append(example_dialogue)

#             code_mixed_explanation.append(str(data_point['code_mixed_explanation']))

#         df =  pd.DataFrame(list(zip(dialogue, code_mixed_explanation)),
#                             columns=['dialogue', 'code_mixed_explanation'])
#         TOKENIZER.add_tokens(['[CONTEXT]', '[TARGET]'], special_tokens=True)
#         MODEL.resize_token_embeddings(len(TOKENIZER))
        
#         del dialogue
#         del example_dialogue
        
#     else:
#         target = []
#         context = []
#         for i in range(1, len(data)+1):
#             data_point = data[str(i)]

#             example_target_speaker = str(data_point['target_speaker']).upper()
#             example_target_utterance = str(data_point['target_utterance'])
#             example_target_utterance = example_target_speaker + " : " + example_target_utterance
#             example_target_utterance = re.sub(' +', ' ', example_target_utterance)
#             target.append(example_target_utterance)

#             example_context_utterance = ""
#             for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
#                 example_context_utterance  = example_context_utterance + str(speaker).upper() + " : " +  str(utterance) + " | "
            
#             example_context_utterance = re.sub(' +', ' ', example_context_utterance)
#             context.append(example_context_utterance)

#             code_mixed_explanation.append(str(data_point['code_mixed_explanation']))

#         df =  pd.DataFrame(list(zip(context, target, code_mixed_explanation)),
#                             columns=['context', 'target', 'code_mixed_explanation']) 
#         del target
#         del context
#         del example_context_utterance

#     # Reading Audio Data
#     acosutic_data = pd.read_pickle(acosutic_path)
#     df['audio_features'] = acosutic_data['audio_feats']
    
#     # Reading Video Data
#     visaul_data = pd.read_pickle(visual_path)
#     df['visual_features'] = visaul_data['video_feats']
 
#     df =  df[df['code_mixed_explanation'] != "?"]
#     df = df.dropna()
#     if lowercase_utterances:
#         df = df.apply(lambda x: x.astype(str).str.lower())

#     del data
#     del text_path
#     del acosutic_path
#     del visual_path
#     del acosutic_data
#     del visaul_data
#     del code_mixed_explanation
#     del example_target_speaker
#     del example_target_utterance
#     gc.collect()
#     return df


def pad_seq(tensor: torch.tensor,
            dim: int,
            max_len: int):
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    else:
        return tensor[:max_len]



def preprocess_dataset(dataset):
    source=[SOURCE_PREFIX + s for s in dataset['src_text'].values.tolist()]
    model_inputs = TOKENIZER(source,
                             max_length=SOURCE_MAX_LEN,
                             padding='max_length',
                             truncation=True) 

    
    # if unfolded_dialogue:
    #     source = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN].values.tolist()]
    #     model_inputs = TOKENIZER(source,
    #                              max_length=SOURCE_MAX_LEN,
    #                              padding='max_length',
    #                              truncation=True)        
    #     del source
        
    # else:
    #     source_1 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_1].values.tolist()]
    #     source_2 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_2].values.tolist()]
    #     model_inputs = TOKENIZER(source_1,
    #                              source_2,
    #                              max_length=SOURCE_MAX_LEN,
    #                              padding='max_length',
    #                              truncation=True)
            
    #     del source_1
    #     del source_2
    
    target = [TARGET_PREFIX + t for t in dataset['labels'].values.tolist()]
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(target,
                           max_length=TARGET_MAX_LEN,
                           padding='max_length',
                           truncation=True)    
	    # IMP: 
	    # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)

	# model_inputs['acoustic_input'] = torch.stack([pad_seq(torch.tensor(af, dtype=torch.float),
	#                                                       dim=ACOUSTIC_DIM,
	#                                                       max_len=ACOUSTIC_MAX_LEN)
	#                                               for af in dataset['audio_features'].values.tolist()], 0).to(DEVICE)
    model_inputs['visual_inf']=torch.stack(dataset['visual_feat'].values.tolist())
    # model_inputs['image_features']=torch.stack(dataset['image_features'].values.tolist())
    print(type(model_inputs['visual_inf']))
    for l in model_inputs['visual_inf']:
    	print(l.shape)
	# # for l in dataset['visual_feat']:
	# 	print(l.shape)
	# model['visual_inf']=torch.tensor([l for l in dataset['visual_feat']], dtype=torch.float, device=DEVICE)
    model_inputs['visual_input'] = torch.stack([pad_seq(torch.tensor(vf[0], dtype=torch.float),
                                                        dim=VISUAL_DIM,
                                                        max_len=VISUAL_MAX_LEN)
                                                for vf in dataset['visual_feat'].values.tolist()], 0).to(DEVICE)
    print(type(model_inputs['visual_input']))
    # for l in model_inputs['visual_input']:
    # 	print(l.shape)

    model_inputs['labels'] = torch.tensor([l for l in labels['input_ids']], dtype=torch.long, device=DEVICE)
    # for l in model_inputs['labels']:
    # 	# print(l)
    # 	print(l.shape)

    del target
    del labels
    gc.collect()
    return model_inputs

 
def set_up_data_loader(text_path: str,
                       visual_path: str,image_transform,
                       trainornot):
    dataset = preprocess_dataset(prepare_dataset(text_path,
                                                 visual_path,image_transform,trainornot))
    print(dataset.keys())
    dataset = TensorDataset(dataset['input_ids'],
                            dataset['attention_mask'], 
                            dataset['visual_input'], 
                            dataset['labels'],
                            dataset['visual_inf'])
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



def get_scores(reference_list: list,
               hypothesis_list: list):
    count=0
    met=0
    bleu_1=0
    bleu_2=0
    bleu_3=0
    bleu_4=0
    rouge1=0
    rouge2=0
    rougel = 0
    weights_1 = (1./1.,)
    weights_2 = (1./2. , 1./2.)
    weights_3 = (1./3., 1./3., 1./3.)
    weights_4 = (1./4., 1./4., 1./4., 1./4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        met += meteor_score([reference], hypothesis)

        reference = reference.split()
        hypothesis = hypothesis.split()
        bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1*100/count,
        "rouge_2": rouge2*100/count,
        "rouge_L": rougel*100/count,
        "bleu_1": bleu_1*100/count,
        "bleu_2": bleu_2*100/count,
        "bleu_3": bleu_3*100/count,
        "bleu_4": bleu_4*100/count,
        "meteor": met*100/count,
    }



def _save(model, 
          output_dir: str,
          tokenizer=None,
          state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def save_model(model, 
               output_dir: str,
               tokenizer=None, 
               state_dict=None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        Will only save from the main process.
        """
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)




# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def train_epoch(model,
                data_loader,
                optimizer):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, visual_input, labels,visual_inf = batch
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        # acoustic_input=acoustic_input,
                        visual_input=visual_input,
                        visual_inf=visual_inf,
                        labels=labels)
        loss = outputs['loss']
        epoch_train_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_train_loss/ step




def val_epoch(model,
              data_loader,
              optimizer):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, labels,visual_inf = batch
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            # acoustic_input=acoustic_input,
                            visual_input=visual_input,
                            visual_inf=visual_inf,
                            labels=labels)
            loss = outputs['loss']
            epoch_val_loss += loss.item()  
    
    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache() 
    
    return epoch_val_loss/ step




def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               **gen_kwargs):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, labels,visual_inf = batch

            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           # acoustic_input=acoustic_input,
                                           visual_input=visual_input,
                                           visual_inf=visual_inf,
                                           **gen_kwargs)
                            
            generated_ids = generated_ids.detach().cpu().numpy()
            generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels = labels.detach().cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print("Pred", decoded_preds)
            # print("gold", decoded_labels)
            predictions.extend(decoded_preds)
            gold.extend(decoded_labels)
        print(len(predictions))
        print(len(gold))
    
    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del generated_ids
    del decoded_preds
    del decoded_labels
    gc.collect()
    torch.cuda.empty_cache() 
    
    return predictions, gold




def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   **gen_kwargs):
    predictions, gold = test_epoch(model,
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
    # result = get_scores(predictions, gold)

        
    if "Test" in desc:
        # print('gold: {}'.format(gold[:10]))
        # print('pred: {}'.format(predictions[:10]))
        acc=0
        cmp_acc =0
        em_acc =0
        se_acc =0
        sev_acc =0
        cnt=0
        # for i in range(len(gold)):
        #     tokenizer = RegexpTokenizer(r'\w+')
        #     reference1=tokenizer.tokenize(gold[i].lower())
        #     reference2=tokenizer.tokenize(predictions[i].lower())
        #     print("pred", reference1)
        #     print("gold", reference2)
        #     if reference1 == reference2:
        #         acc+=1
        ALL_ASPECTS_BOOKS = np.array(["content", "packaging", "price", "quality"]) 
        BOOKS = 0
        ALL_ASPECTS_EDIBLE = np.array(["taste", "packaging", "smell", "price", "quality"])
        EDIBLE = 0
        ALL_ASPECTS_FASHION = np.array(["fit", "packaging", "price", "quality" , "color", "style"])
        FASHION = 0
        ALL_ASPECTS_ELECTRONICS = np.array(["design", "packaging", "price", "quality" , "hardware", "software"])

        def get_one_hot_encode(labels):
            one_hot = []
            for lab in labels:
                    one_hot.append(np.isin(ALL_ASPECTS_ELECTRONICS, lab).astype(float))
            return one_hot
        
        df = pd.read_csv(path_to_test)

        label = df[['Aspect Term', 'Aspect Term.1', 'Aspect Term.2', 'Aspect Term.3']].values.tolist()
        l_true = get_one_hot_encode(label)
        

        def compare(txt1, txt2, x):
            j=0
            i=0
            while i<len(txt1) and j<len(txt2):
                if(txt1[i]==txt2[j]):
                    c=0
                    while(j+1<len(txt2) and i+1<len(txt1) and txt1[i+1]==txt2[j+1]):
                        x[i]=1
                        c=1
                        i+=1
                        j+=1
                    if c:
                        x[i]=1
                i+=1
            return x

        def key_gold(k_goldie, ALL_LABELS):
            keywords_t = []
            for i in range(len(df)):
                txt = df["Review_S"][i].lower().split()
                temp = k_goldie[i]
                temp = temp.replace("[", "")
                temp = temp.replace("]", "")
                temp = temp.replace("'", "")
                temp = temp.replace('"', "")
                # temp = temp.replace(" ", "")
                k_gold = (temp.split(","))
                jh = []
                for j in range(len(l_true[i])):
                    x= []
                    for m in range(len(txt)):
                        x.append(0)
                    if l_true[i][j]:
                        if(df["Aspect Term"][i]==ALL_LABELS[j]):
                            if(len(k_gold) >= 7):
                                k_goldi = k_gold[6].lower().split()
                                x = compare(txt,k_goldi,x)

                        elif(df["Aspect Term.1"][i]==ALL_LABELS[j]):
                            if(len(k_gold) >= 10):
                                k_goldi = k_gold[9].lower().split()
                                x = compare(txt,k_goldi,x)

                        elif(df["Aspect Term.2"][i]==ALL_LABELS[j]):
                            if(len(k_gold) >= 13):
                                k_goldi = k_gold[12].lower().split()
                                x = compare(txt,k_goldi,x)

                        elif(df["Aspect Term.3"][i]==ALL_LABELS[j]):
                            if(len(k_gold) >= 16):
                                k_goldi = k_gold[15].lower().split()
                                x = compare(txt,k_goldi,x)
                        jh.append(x)
                    else:
                        jh.append(x)
                keywords_t.append(jh)

            return keywords_t   


        from sklearn.metrics import f1_score, jaccard_score, accuracy_score, classification_report

        def key_matrix(l_true, k_pred, k_true):
            l_true = np.array(l_true)
            k_pred = np.array( k_pred)
            k_true = np.array(k_true)

            jlist = []
            s = c = 0
            token_pred = []
            token_true = []
            for  lt, kp, kt in zip( l_true, k_pred, k_true):
                # jidx = [jaccard_score(t, p, zero_division = 0) < 0.5 if l else False for p, t, l in zip(kp, kt, lt)]
                # lp[jidx] = 0

                jidx = [jaccard_score(t, p, zero_division = 0) >= 0.5 if l else False for p, t, l in zip(kp, kt, lt)]
                jlist.append(jidx)

                s += sum([jaccard_score(t, p, zero_division = 0) for p, t, l in zip(kp, kt, lt) if l])
                c += sum(lt)

                token_pred.extend([p for p, l in zip(kp, lt) if l])
                token_true.extend([t for t, l in zip(kt, lt) if l])


            token_f1 = 0
            iou_f1 = f1_score(l_true, jlist, zero_division = 0, average = "macro")

            # tf1 = f1_score(l_true, l_pred, zero_division = 0, average="macro")
            # tacc = accuracy_score(l_true, l_pred)
            return {"exp_jacc": (s/c), 'iou_f1': iou_f1, 'token_f1': token_f1}



        def get_category_encode1(x_goldi, x_predict):
            y_goldi =[]
            y_predict = []
            if(len(x_goldi)>=6) and (len(x_predict)>=6):
                if ((x_predict[5]) == ' 0' or (x_predict[5]) == ' 1') and ((x_goldi[5]) == ' 0' or (x_goldi[5]) == ' 1'): 
                    x_goldi[5] = x_goldi[5].replace(" ", "")
                    x_predict[5] = x_predict[5].replace(" ", "")
                    y_goldi.append(int(x_goldi[5]))
                    y_predict.append(int(x_predict[5]))
                else:
                    return [], []

            if(len(x_goldi)>=9) and (len(x_predict)>=9):
                if ((x_predict[8]) == ' 0' or (x_predict[8]) == ' 1') and ((x_goldi[8]) == ' 0' or (x_goldi[8]) == ' 1'): 
                    x_goldi[8] = x_goldi[8].replace(" ", "")
                    x_predict[8] = x_predict[8].replace(" ", "")
                    y_goldi.append(int(x_goldi[8]))
                    y_predict.append(int(x_predict[8]))
                else:
                    return [], []

            if(len(x_goldi)>=12) and (len(x_predict)>=12):
                if ((x_predict[11]) == ' 0' or (x_predict[11]) == ' 1') and ((x_goldi[11]) == ' 0' or (x_goldi[11]) == ' 1'): 
                    x_goldi[11] = x_goldi[11].replace(" ", "")
                    x_predict[11] = x_predict[11].replace(" ", "")
                    y_goldi.append(int(x_goldi[11]))
                    y_predict.append(int(x_predict[11]))
                else:
                    return [], []

            if(len(x_goldi)>=15) and (len(x_predict)>=15):
                if ((x_predict[14]) == ' 0' or (x_predict[14]) == ' 1') and ((x_goldi[14]) == ' 0' or (x_goldi[14]) == ' 1'): 
                    x_goldi[14] = x_goldi[14].replace(" ", "")
                    x_predict[14] = x_predict[14].replace(" ", "")
                    y_goldi.append(int(x_goldi[14]))
                    y_predict.append(int(x_predict[14]))
                else:
                    return [], []
            
            y_predict = y_predict[:len(y_goldi)]
            return y_goldi, y_predict

        def get_category_encode(x_goldi, x_predict):
            y_goldi =[]
            y_predict = []
            if(len(x_goldi)>=6) and (len(x_predict)>=6):
                if(x_goldi[4] == x_predict[4]):
                    if (x_predict[5]) == '0' or (x_predict[5]) == '1': 
                        # x_goldi[1] = x_goldi[1].replace(" ", "")
                        # x_predict[1] = x_predict[1].replace(" ", "")
                        y_goldi.append(int(x_goldi[5]))
                        y_predict.append(int(x_predict[5]))
                    else:
                        return [], []

            if(len(x_goldi)>=9) and (len(x_predict)>=9):
                if(x_goldi[7] == x_predict[7]):
                    if (x_predict[8]) == '0' or (x_predict[8]) == '1': 
                        # x_goldi[4] = x_goldi[4].replace(" ", "")
                        # x_predict[4] = x_predict[4].replace(" ", "")
                        y_goldi.append(int(x_goldi[8]))
                        y_predict.append(int(x_predict[8]))
                    else:
                        return [], []

            if(len(x_goldi)>=12) and (len(x_predict)>=12):
                if(x_goldi[10] == x_predict[10]):
                    if (x_predict[11]) == '0' or (x_predict[11]) == '1': 
                        # x_goldi[7] = x_goldi[7].replace(" ", "")
                        # x_predict[7] = x_predict[7].replace(" ", "")
                        y_goldi.append(int(x_goldi[11]))
                        y_predict.append(int(x_predict[11]))
                    else:
                        return [], []

            if(len(x_goldi)>=15) and (len(x_predict)>=15):
                if(x_goldi[13] == x_predict[13]):
                    if (x_predict[14]) == '0' or (x_predict[14]) == '1': 
                        # x_goldi[10] = x_goldi[10].replace(" ", "")
                        # x_predict[10] = x_predict[10].replace(" ", "")
                        y_goldi.append(int(x_goldi[14]))
                        y_predict.append(int(x_predict[14]))
                    else:
                        return [], []

            return y_goldi, y_predict
        
        def jaccard(list1, list2):
            intersection = len(list(set(list1).intersection(list2)))
            union = (len(list1) + len(list2)) - intersection

            if union == 0 : return 0
            return float(intersection) / union
        


        def jacc_scores(x_goldi, x_predict):
            y_predict = []
            y_goldi = []
            if(len(x_goldi)>=7) and (len(x_predict)>=7):
                # if str(x_goldi[2]) != " nan":
                y_predict.extend(x_predict[6].lower().split())
                y_goldi.extend(x_goldi[6].lower().split())
            
            if(len(x_goldi)>=10) and (len(x_predict)>=10):
                # if str(x_goldi[5]) != " nan":
                y_predict.extend(x_predict[9].lower().split())
                y_goldi.extend(x_goldi[9].lower().split())
            
            if(len(x_goldi)>=13) and (len(x_predict)>=13):
                # if str(x_goldi[8]) != " nan":
                y_predict.extend(x_predict[12].lower().split())
                y_goldi.extend(x_goldi[12].lower().split())
            
            if(len(x_goldi)>=16) and (len(x_predict)>=16):
                # if str(x_goldi[11]) != " nan":
                y_predict.extend(x_predict[15].lower().split())
                y_goldi.extend(x_goldi[15].lower().split())

            print("exp_gold:", y_goldi)
            print("exp_predict:", y_predict)

            # score_exp = sentence_bleu([y_goldi], y_predict)
            score_exp = jaccard(y_goldi, y_predict)
            return score_exp
        

        def indi_jacc(x_goldi, x_predict):
            bool1 = 0
            bool2 = 0
            bool3 = 0
            bool4 = 0
            if(len(x_goldi)>=7) and (len(x_predict)>=7):
                # if str(x_goldi[2]) != " nan":
                y_predict = x_predict[6].lower().split()
                y_goldi = x_goldi[6].lower().split()
                score_exp1 = jaccard(y_goldi, y_predict)
                bool1 = 1

            else:
                score_exp1 = 0

            if(len(x_goldi)>=10) and (len(x_predict)>=10):
                # if str(x_goldi[5]) != " nan":
                y_predict = x_predict[9].lower().split()
                y_goldi = x_goldi[9].lower().split()
                score_exp2 = jaccard(y_goldi, y_predict)
                bool2 = 1
            else:
                score_exp2 = 0
            
            if(len(x_goldi)>=13) and (len(x_predict)>=13):
                # if str(x_goldi[8]) != " nan":
                y_predict = x_predict[12].lower().split()
                y_goldi = x_goldi[12].lower().split()
                score_exp3 = jaccard(y_goldi, y_predict)
                bool3 = 1
            else:
                score_exp3 = 0

            if(len(x_goldi)>=16) and (len(x_predict)>=16):
                # if str(x_goldi[11]) != " nan":
                y_predict = x_predict[15].lower().split()
                y_goldi = x_goldi[15].lower().split()
                score_exp4 = jaccard(y_goldi, y_predict)
                bool4 = 1
            else:
                score_exp4 = 0

            # score_exp = sentence_bleu([y_goldi], y_predict)
            return score_exp1, score_exp2, score_exp3, score_exp4, bool1, bool2, bool3, bool4
        

        def comp_score(x_goldi, x_predict):
            if(len(x_goldi)>=2) and (len(x_predict)>=2):
                x_predict[1] = x_predict[1].replace(" ", "")
                x_goldi[1] = x_goldi[1].replace(" ", "")
                if(x_goldi[1]==x_predict[1]):
                    # print("Comp :", x_goldi[1])
                    # print("Comp :", x_predict[1])
                    return 1
                
            return 0
        
        
        def emo_score(x_goldi, x_predict):
            if(len(x_goldi)>=3) and (len(x_predict)>=3):
                x_predict[2] = x_predict[2].replace(" ", "")
                x_goldi[2] = x_goldi[2].replace(" ", "")
                if(x_goldi[2]==x_predict[2]):
                    # print("Emo :", x_goldi[2])
                    # print("Emo :", x_predict[2])
                    return 1
                
            return 0
         

        def senti_score(x_goldi, x_predict):
            if(len(x_goldi)>=4) and (len(x_predict)>=4):
                x_predict[3] = x_predict[3].replace(" ", "")
                x_goldi[3] = x_goldi[3].replace(" ", "")
                if(x_goldi[3]==x_predict[3]):
                    # print("Senti :", x_goldi[3])
                    # print("Senti :", x_predict[3])
                    return 1
                
            return 0
        
        def sev_score(x_goldi, x_predict):
            if(len(x_goldi)>=1) and (len(x_predict)>=1):
                x_predict[0] = x_predict[0].replace(" ", "")
                x_goldi[0] = x_goldi[0].replace(" ", "")
                if(x_goldi[0]==x_predict[0]):
                    # print("Sev :", x_goldi[0])
                    # print("Sev :", x_predict[0])
                    return 1
                
            return 0
        
        as1=0
        as2=0
        mac_cat1 = 0
        acc_cat1 = 0
        mac_cat = 0
        acc_cat = 0
        goldies = []
        predicted = []
        count=0
        count1=0
        curr_bleu = 0
        curr_indi1 = 0
        curr_indi2 = 0
        curr_indi3 = 0 
        curr_indi4 = 0
        count_indi1 = 0
        count_indi2 = 0
        count_indi3 = 0
        count_indi4 = 0

        curr_comp = 0
        curr_sarc = 0
        curr_emo = 0
        curr_senti = 0
        curr_sev = 0
        for i in range(len(gold)):

            # print("preds", predictions[i])
            # print("gold", gold[i])
            goldi = gold[i]
            goldi = goldi.replace("[", "")
            goldi = goldi.replace("]", "")
            goldi = goldi.replace("'", "")
            goldi = goldi.replace('"', "")
            # goldi = goldi.replace(" ", "")
            x_goldi = (goldi.split(","))
            print("gold", x_goldi)
            y =[]
            if(len(x_goldi)>=5):
                x_goldi[4] = x_goldi[4].replace(" ", "")
                y.append(x_goldi[4])
            if(len(x_goldi)>=8):
                x_goldi[7] = x_goldi[7].replace(" ", "")
                y.append(x_goldi[7])
            if(len(x_goldi)>=11):
                x_goldi[10] = x_goldi[10].replace(" ", "")
                y.append(x_goldi[10])
            if(len(x_goldi)>=14):
                x_goldi[13] = x_goldi[13].replace(" ", "")
                y.append(x_goldi[13])
            
            goldies.append(y)

            predict = predictions[i]
            predict = predict.replace("[", "")
            predict = predict.replace("]", "")
            predict = predict.replace("'", "")
            predict = predict.replace('"', "")
            # predict = predict.replace(" ", "")
            x_predict = (predict.split(","))
            # # print("predict", type(x_predict))
            print("predict", (x_predict))
            y =[]
            if(len(x_predict)>=5):
                x_predict[4] = x_predict[4].replace(" ", "")
                y.append(x_predict[4])
            if(len(x_predict)>=8):
                x_predict[7] = x_predict[7].replace(" ", "")
                y.append(x_predict[7])
            if(len(x_predict)>=11):
                x_predict[10] = x_predict[10].replace(" ", "")
                y.append(x_predict[10])
            if(len(x_predict)>=14):
                x_predict[13] = x_predict[13].replace(" ", "")
                y.append(x_predict[13])

            predicted.append(y)

            # cat_p = get_categroy_encode(predict.split(","))
            # cat_g = get_categroy_encode(goldi.split(","))
            # cat_p = cat_p[:len(cat_g)]

            # cat_g, cat_p = get_category_encode(x_goldi, x_predict)
            cat_g1, cat_p1 = get_category_encode1(x_goldi, x_predict)
            # cat_g, cat_p = get_category_encode(x_goldi, x_predict)
            curr_bleu += jacc_scores(x_goldi, x_predict)
            temp_indi1, temp_indi2, temp_indi3, temp_indi4, bool_indi1, bool_indi2, bool_indi3, bool_indi4 = indi_jacc(x_goldi, x_predict)
            curr_indi1 += temp_indi1
            curr_indi2 += temp_indi2
            curr_indi3 += temp_indi3
            curr_indi4 += temp_indi4
            count_indi1 += bool_indi1
            count_indi2 += bool_indi2
            count_indi3 += bool_indi3
            count_indi4 += bool_indi4

            curr_comp += comp_score(x_goldi, x_predict)
            # curr_sarc += sarc_score(x_goldi, x_predict)
            curr_emo += emo_score(x_goldi, x_predict)
            curr_senti += senti_score(x_goldi, x_predict)
            curr_sev += sev_score(x_goldi, x_predict)
            # print("gold", cat_g)
            # print("predict", cat_p)
            if(len(cat_g1)!=0):
                count1+=1
                mac_cat1+=f1_score(cat_g1, cat_p1, zero_division = 0, average="macro")
                acc_cat1+=accuracy_score(cat_g1, cat_p1)
            # if(len(cat_g)!=0):
            #     count+=1
            #     mac_cat+=f1_score(cat_g, cat_p, zero_division = 0, average="macro")
            #     acc_cat+=accuracy_score(cat_g, cat_p)
            # print("preds", predict)
            # print("gold", goldi)
        
        goldies = np.array(get_one_hot_encode(goldies))
        predicted = np.array(get_one_hot_encode(predicted))
        
        # t_keywords = key_gold(gold, ALL_ASPECTS_ELECTRONICS)
        # p_keywords = key_gold(predictions, ALL_ASPECTS_ELECTRONICS)
        # mat_score = key_matrix(l_true, p_keywords, t_keywords)
        
        as1 = f1_score(goldies, predicted, zero_division = 0, average="micro")
        as2 = f1_score(goldies, predicted, zero_division = 0, average="macro")
        print(count1)
        print(count)
        acc_cat1 = acc_cat1/len(gold)
        mac_cat1 = mac_cat1/len(gold)
        # acc_cat = acc_cat/len(gold)
        # mac_cat = mac_cat/len(gold)
        curr_bleu = curr_bleu/len(gold)
        curr_indi1 = curr_indi1/count_indi1 if count_indi1 !=0 else 0
        curr_indi2 = curr_indi2/count_indi2 if count_indi2 !=0 else 0
        curr_indi3 = curr_indi3/count_indi3 if count_indi3 !=0 else 0
        curr_indi4 = curr_indi4/count_indi4 if count_indi4 !=0 else 0

        curr_comp = curr_comp/len(gold)
        # curr_sarc = curr_sarc/len(gold)
        curr_emo = curr_emo/len(gold)
        curr_senti = curr_senti/len(gold)
        curr_sev = curr_sev/len(gold)

        global best_as1
        if best_as1<(as1):
            best_as1=as1

        
        global best_as2
        if best_as2<(as2):
            best_as2=as2

        global best_as3
        if best_as3<(acc_cat1):
            best_as3=acc_cat1

        global best_as4
        if best_as4<(mac_cat1):
            best_as4=mac_cat1
        
        # global best_as5
        # if best_as5<(acc_cat):
        #     best_as5=acc_cat

        # global best_as6
        # if best_as6<(mac_cat):
        #     best_as6=mac_cat

        global best_bleu
        if best_bleu<(curr_bleu):
            best_bleu=curr_bleu

        # global best_iouf1
        # if best_iouf1<(mat_score["iou_f1"]):
        #     best_iouf1=mat_score["iou_f1"]
        
        # global best_tokenf1
        # if best_tokenf1<(mat_score["token_f1"]):
        #     best_tokenf1=mat_score["token_f1"]
        
        # global best_jacc
        # if best_jacc<(mat_score["exp_jacc"]):
        #     best_jacc=mat_score["exp_jacc"]

        global best_indi1
        if best_indi1<(curr_indi1):
            best_indi1=curr_indi1

        global best_indi2
        if best_indi2<(curr_indi2):
            best_indi2=curr_indi2

        global best_indi3
        if best_indi3<(curr_indi3):
            best_indi3=curr_indi3

        global best_indi4
        if best_indi4<(curr_indi4):
            best_indi4=curr_indi4

        global best_comp
        if best_comp<(curr_comp):
            best_comp=curr_comp

        # global best_sarc
        # if best_sarc<(curr_sarc):
        #     best_sarc=curr_sarc

        global best_emo
        if best_emo<(curr_emo):
            best_emo=curr_emo

        global best_senti
        if best_senti<(curr_senti):
            best_senti=curr_senti

        global best_sev
        if best_sev<(curr_sev):
            best_sev=curr_sev
        
        print('curr micro: {}'.format(as1))
        print('best micro: {}'.format(best_as1))

        print('curr macro: {}'.format(as2))
        print('best macro: {}'.format(best_as2))

        print('curr acc: {}'.format(acc_cat1))
        print('best acc: {}'.format(best_as3))

        print('curr macro_cat: {}'.format(mac_cat1))
        print('best macro_cat: {}'.format(best_as4))

        # print('curr acc_same: {}'.format(acc_cat))
        # print('best acc_same: {}'.format(best_as5))

        # print('curr macro_cat_same: {}'.format(mac_cat))
        # print('best macro_cat_same: {}'.format(best_as6))

        print('curr bleu: {}'.format(curr_bleu))
        print('best bleu: {}'.format(best_bleu))

        # print('curr iou_f1: {}'.format(mat_score["iou_f1"]))
        # print('best iou_f1: {}'.format(best_iouf1))

        # print('curr token_f1: {}'.format(mat_score["token_f1"]))
        # print('best token_f1: {}'.format(best_tokenf1))

        # print('curr jacc: {}'.format(mat_score["exp_jacc"]))
        # print('best jacc: {}'.format(best_jacc))

        print('curr indi1: {}'.format(curr_indi1))
        print('best indi1: {}'.format(best_indi1))

        print('curr indi2: {}'.format(curr_indi2))
        print('best indi2: {}'.format(best_bleu))

        print('curr indi3: {}'.format(curr_indi3))
        print('best indi3: {}'.format(best_indi3))

        print('curr indi4: {}'.format(curr_indi4))
        print('best indi4: {}'.format(best_indi4))

        print('curr comp: {}'.format(curr_comp))
        print('best comp: {}'.format(best_comp))

        print('curr emo: {}'.format(curr_emo))
        print('best emo: {}'.format(best_emo))

        print('curr senti: {}'.format(curr_senti))
        print('best senti: {}'.format(best_senti))

        print('curr sev: {}'.format(curr_sev))
        print('best sev: {}'.format(best_sev))


        # for i in range(len(gold)):
        #     tokenizer = RegexpTokenizer(r'\w+')
        #     reference1 = gold[i].split("&")
        #     reference2 = predictions[i].split("&")

        #     for j in range(len(reference1)):
        #         reference1[j] = tokenizer.tokenize(reference1[j].lower())
            
        #     for j in range(len(reference2)):
        #         reference2[j] = tokenizer.tokenize(reference2[j].lower())

        #     if len(reference2)>0:
        #         if reference1[0] == reference2[0]:
        #             sev_acc+=1
        #         if len(reference2)>1:
        #             if reference1[1] == reference2[1]:
        #                 cmp_acc+=1
        #             if len(reference2)>2:
        #                 if reference1[2] == reference2[2]:
        #                     em_acc+=1
        #                 if len(reference2)>3:
        #                     if reference1[3] == reference2[3]:
        #                         se_acc+=1
        #     print("preds", reference2)
        #     print("gold", reference1)
        
        # global best_acc_cmp
        # if best_acc_cmp<(cmp_acc/len(gold)):
        #     best_acc_cmp=cmp_acc/len(gold)

        # print('curr cmp: {}'.format(cmp_acc/len(gold)))
        # print('best cmp: {}'.format(best_acc_cmp))


        # global best_acc_em
        # if best_acc_em<(em_acc/len(gold)):
        #     best_acc_em=em_acc/len(gold)

        # print('curr em: {}'.format(em_acc/len(gold)))
        # print('best em: {}'.format(best_acc_em))

        # global best_acc_se
        # if best_acc_se<(se_acc/len(gold)):
        #     best_acc_se=se_acc/len(gold)

        # print('curr se: {}'.format(se_acc/len(gold)))
        # print('best se: {}'.format(best_acc_se))


        # global best_acc_sev
        # if best_acc_sev<(sev_acc/len(gold)):
        #     best_acc_sev=sev_acc/len(gold)

        # print('curr sev: {}'.format(sev_acc/len(gold)))
        # print('best sev: {}'.format(best_acc_sev))






            # print(reference1)

        # responseslist=[]
        # predictedresponselist=[]
        # bleulist=[]

        # for i in range(len(gold)):
        #     bestcurr=-1

        #     currgold=gold[i].split('\', \'')
        #     # print(currgold)
        #     currgold2=[]
        #     for j in range(len(currgold)):
        #         currgold2.append(currgold[j].split('\", \"'))

        #     currgold2=[item for sublist in currgold2 for item in sublist]
        #     # print(currgold2)
        #     currgold3=[]
        #     for j in range(len(currgold2)):
        #         currgold3.append(currgold2[j].split('\', \"'))
        #     currgold3=[item for sublist in currgold3 for item in sublist]
        #     # print(currgold3)
        #     # currgold4=[]
        #     # for j in range(len(currgold4)):
        #     #     currgold4.append(currgold3[j].split('\", \''))

        #     # currgold4=[item for sublist in currgold3 for item in sublist]
        #     # print(currgold4)
        #     print(currgold3)
        #     # print(currgold4)
        #     # input('enetr')

        #     for ref in currgold3:

        #         reference=ref
        #     # responseslist.append(reference)
        #     # predictedresponselist.append(predictions[i])
        #     # print(responseslist)
        #     # print(predictedresponselist)  
        #     # scores = scorer.score(reference.lower(),candidate.lower())
        #     # print(scores)
        #         cc = SmoothingFunction()
        #         tokenizer = RegexpTokenizer(r'\w+')
        #         reference1=tokenizer.tokenize(reference.lower())
        #         candidate1=tokenizer.tokenize(predictions[i].lower())
        #         # print('predicted: {}'.format(candidate1))
        #         # print('true: {}'.format(reference1))
        #         # input('enter')
        #     # ref = "let it go".split()
        #     # hyp = "let go it".split()
        #         currscore=sentence_bleu([reference1],candidate1,smoothing_function=cc.method3)
        #         # global bestbleu
        #         bestcurr=max(bestcurr,currscore)
        #     print('currbl: {}'.format(bestcurr))
        #     bleulist.append(bestcurr)

        # currbleuscore=0
        # for b in bleulist:
        #     currbleuscore+=b

        # currbleuscore=currbleuscore/len(bleulist)
        # global bestbleu
        # print('curr: {}'.format(currbleuscore))
        # if bestbleu<currbleuscore:
        #     bestbleu=currbleuscore
        # print('BEST: {}'.format(bestbleu))

            # if len(predictions[i])==0:
        #         continue
        #     cnt+=1
        #     if gold[i].strip()[0]==predictions[i].strip()[0]:
        #         acc+=1
        # print(acc/cnt)
        # global best_acc
        # if best_acc<(acc/len(gold)):
        #     best_acc=acc/len(gold)
        # print('best: {}'.format(best_acc))
        test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
        file_name = RESULT_OUTPUT_DIR + "test/MAF_TAV_BART_epoch_" + str(epoch+1) + "_test_results.csv"
        test_df.to_csv(file_name, index=False)  
        print("Test File saved")
    
    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache() 

	# return result  




def prepare_for_training(model, 
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)
            
    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}            
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )
    
    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache() 
    
    return optimizer




def train(model,
          tokenizer,
          train_data_loader,
          val_data_loader,
          test_data_loader,
          base_learning_rate,
          new_learning_rate,
          weight_decay,
          **gen_kwargs):
    
    optimizer = prepare_for_training(model=model,
                                     base_learning_rate=base_learning_rate,
                                     new_learning_rate=new_learning_rate,
                                     weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    val_rouge_2 = []
    patience = 1
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model,
                                 train_data_loader, 
                                 optimizer)
        train_losses.append(train_loss)
        
        val_loss = val_epoch(model,
                             val_data_loader, 
                             optimizer)
        val_losses.append(val_loss)

        # val_results = get_val_scores(model,
        #                              tokenizer,
        #                              val_data_loader,
        #                              desc="Validation Generation Iteration",
        #                              epoch=epoch,
        #                              **gen_kwargs)
        # val_rouge_2.append(val_results['rouge_2'])
        
        get_val_scores(model,tokenizer,test_data_loader,desc="Test Generation Iteration",epoch=epoch,**gen_kwargs)

        # test_results = get_val_scores(model,
        #                               tokenizer,
        #                               test_data_loader,
        #                               desc="Test Generation Iteration",
        #                               epoch=epoch,
        #                               **gen_kwargs)
    
        # print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
        
        # print("\nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
        # val_results['rouge_1'], val_results['rouge_2'], val_results['rouge_L'], val_results['bleu_1'], val_results['bleu_2'], val_results['bleu_3'], val_results['bleu_4'], val_results['meteor']))
        
        # print("\ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}".format(
        # test_results['rouge_1'], test_results['rouge_2'], test_results['rouge_L'], test_results['bleu_1'], test_results['bleu_2'], test_results['bleu_3'], test_results['bleu_4'], test_results['meteor']))
        
        path = MODEL_OUTPUT_DIR + "MAF_TAV_BART_epoch__epoch_" + str(epoch+1) + "_" + datetime.now().strftime('%d-%m-%Y-%H:%M')
        print(path)
        save_model(model,
                   path,
                   tokenizer)
        print("Model saved at path: ", path)
        
        # if val_results['rouge_2'] < max(val_rouge_2):
        #     patience = patience + 1          
        #     if patience == EARLY_STOPPING_THRESHOLD:
        #         break
                
        # else:
        #     patience = 1

        del train_loss
        del val_loss
        del path
        gc.collect()
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    TOKENIZER = BartTokenizerFast.from_pretrained('facebook/bart-base')
    print("Tokenizer loaded...\n")
    MODEL = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base')
    print("Model loaded...\n")
    for name, param in MODEL.state_dict().items():
        print(name, param.size())
    # input('Enter')
    # pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    # print(pytorch_total_train_params)
    MODEL.to(DEVICE)





    SOURCE_PREFIX = ''
    TARGET_PREFIX = ''
    
    # print(TARGET_COLUMN)
    # print(MODEL_OUTPUT_DIR)
    # print(RESULT_OUTPUT_DIR)
    # print(SOURCE_PREFIX)
    # print(TARGET_PREFIX)
    
    gc.collect()
    
    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print("Total parameters: ", pytorch_total_params)
    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print("Total trainable parameters: ", pytorch_total_train_params)
    # input('ENter')
    
    for name, param in MODEL.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name: 
            print(name)
    
    
    # ------------------------------ READ DATASET ------------------------------ #
    
    image_transform = transforms.Compose([
    transforms.ToTensor(),                               
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

    mse_data = MSEDataModule(path_to_train, path_to_val, 
                         path_to_test, path_to_images, 
                         TOKENIZER, image_transform, batch_size=1)
    print('DONE===>')

    train_dataset = set_up_data_loader(path_to_train,
                                       path_to_images,image_transform,trainornot=True)
    print("\nTraining Data Loaded...")
    
    val_dataset = set_up_data_loader(path_to_val,path_to_images,image_transform,trainornot=False)
    print("\nValidation Data Loaded...")
    
    test_dataset = set_up_data_loader(path_to_test,path_to_images,image_transform,trainornot=False)
    print("\nTest Data Loaded...")
    gc.collect()  
    
    # ------------------------------ TRAINING SETUP ------------------------------ #
        
    gen_kwargs = {
        'num_beams': NUM_BEAMS,
        'max_length': TARGET_MAX_LEN,
        'early_stopping': EARLY_STOPPING,
        'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE
    }
    
    train(model=MODEL,
          tokenizer=TOKENIZER,
          train_data_loader=train_dataset,
          val_data_loader=val_dataset,
          test_data_loader=test_dataset,
          base_learning_rate=BASE_LEARNING_RATE,
          new_learning_rate=NEW_LEARNING_RATE,
          weight_decay=WEIGHT_DECAY,
          **gen_kwargs)
    
    print("Model Trained!")

