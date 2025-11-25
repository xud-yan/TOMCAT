import torch.nn.functional as F

import math

from torch import no_grad
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class TOMCAT_BM(nn.Module):

    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.device = config.device

        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset
        self.enable_pos_emb = True
        self.attr_dropout = nn.Dropout(config.attr_dropout)

        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length, device='cpu')
        self.dtype = self.clip.dtype
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)

        img_cls_dim = self.clip.visual.output_dim
        img_patch_dim = 1024

        self.token_ids, self.soft_att_obj, comp_ctx_vectors = self.construct_soft_prompt()

        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        if self.config.use_adapter:
            self.additional_visual_params = self.add_visual_tunable_params()

        self.soft_a_and_o_embedding = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors)


        # loss_func
        self.comp_cls_loss_weight = config.comp_cls_loss_weight
        self.classification_loss_func = CrossEntropyLoss()

    def release_text_encoder(self):
        del self.token_ids
        del self.soft_att_obj
        del self.comp_ctx_vectors
        del self.tokenizer
        del self.text_encoder
        del self.soft_a_and_o_embedding
        torch.cuda.empty_cache()

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width,
                                        bottleneck=self.config.adapter_dim,
                                        dropout=self.config.adapter_dropout
                                        ) for _ in range(adapter_num)])
        return params

    def encode_image(self, x: torch.Tensor):
        if self.config.use_adapter:
            return self.encode_image_with_adapter(x)
        else:
            return self.encode_image_without_adapter(x)

    def encode_image_without_adapter(self, x: torch.tensor):
        return self.clip.encode_image(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                  dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        '''img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature'''
        img_cls_feature = img_feature[:, 0, :]
        img_cls_feature = self.clip.visual.ln_post(img_cls_feature)
        if self.clip.visual.proj is not None:
            img_cls_feature = img_cls_feature @ self.clip.visual.proj
        return img_cls_feature, img_feature


    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    # yeah
    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template[0:1],
                                   context_length=self.config.context_length)

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized)
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init[0:1]
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length)
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)
        return token_ids.squeeze(), soft_att_obj, comp_ctx_vectors


    def classification_loss(self, comp_logits, target):
        loss_fn = self.classification_loss_func

        batch_target = target.to(self.device)
        loss_comp = loss_fn(comp_logits, batch_target)

        return loss_comp


    def get_label_text_features(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        pair_num = len(pair_idx)
        class_token_ids = self.token_ids.repeat(pair_num, 1)
        class_token_embedding = self.clip.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip.dtype)

        eos_idx = int(self.token_ids.argmax())
        soft_a_and_o_embedding = self.attr_dropout(self.soft_a_and_o_embedding)
        # comp
        class_token_embedding[:, 1: len(self.comp_ctx_vectors) + 1, :] = self.comp_ctx_vectors.type(self.clip.dtype)
        class_token_embedding[:, eos_idx - 2, :] = soft_a_and_o_embedding[attr_idx].type(self.clip.dtype)
        class_token_embedding[:, eos_idx - 1, :] = soft_a_and_o_embedding[obj_idx + self.offset].type(self.clip.dtype)

        a = True  # self.config.dataset != 'cgqa'
        if self.config.dataset != 'cgqa':  # and self.config.dataset != 'ut-zappos':
            label_text_features, _ = self.encode_text(
                token_ids=self.token_ids,
                token_tensors=class_token_embedding,
                enable_pos_emb=self.enable_pos_emb,
            )
        else:
            with torch.no_grad():
                #to reduce GPU memory for 3090
                #if your GPU memory is enough, you don't need to split
                split = class_token_embedding.size(0) // 2
                label_text_features, _ = self.encode_text(
                    token_ids=self.token_ids,
                    token_tensors=class_token_embedding[:split],
                    enable_pos_emb=self.enable_pos_emb,
                )
                label_text_features2, _ = self.encode_text(
                    token_ids=self.token_ids,
                    token_tensors=class_token_embedding[split:],
                    enable_pos_emb=self.enable_pos_emb,
                )
                label_text_features = torch.cat((label_text_features, label_text_features2), dim=0)
        return label_text_features

    def encode_text_from_pure_text(self, label_text):
        token_ids = self.tokenizer(label_text,
                                   context_length=self.config.context_length).to(self.config.device)
        label_text_features, _ = self.encode_text(
            token_ids=token_ids,
            token_tensors=None,
            enable_pos_emb=self.enable_pos_emb,
        )
        return label_text_features

    def encode_text_for_open(self, idx):
        text_features = self.get_label_text_features(idx)
        return text_features

    def forward_for_open(self, batch, label_text_features):
        with torch.no_grad():
            bs = batch[0].shape[0]

            img_cls, _ = self.encode_image(batch[0].type(self.clip.dtype))
            image_features = img_cls

        # calculate composition classification logits
        comp_cls_logits = self.cos_sim_func_4com_cls(image_features, label_text_features)

        return image_features, comp_cls_logits

    def cos_sim_func_4com_cls(self, a, b):
        normalized_a = a / a.norm(dim=-1, keepdim=True)
        normalized_b = b / b.norm(dim=-1, keepdim=True)

        # composition classification logits
        sim = self.clip.logit_scale.exp() * normalized_a @ normalized_b.t()
        return sim

    def forward(self, batch, idx):
        bs = batch[0].shape[0]
        l, _ = idx.shape
        img_cls, _= self.encode_image(batch[0].type(self.clip.dtype))


        image_features = img_cls
        # encode label txt
        label_text_features = self.get_label_text_features(idx)

        # calculate composition classification logits
        comp_cls_logits = self.cos_sim_func_4com_cls(image_features, label_text_features)

        # for inference, only the above pipline is needed
        if not self.training:
            return comp_cls_logits

        comp_cls_loss = self.classification_loss(comp_cls_logits, batch[3])

        loss = self.comp_cls_loss_weight * comp_cls_loss
        return loss