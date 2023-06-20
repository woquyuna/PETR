from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.drop import build_dropout

import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import xavier_init, constant_init
import math


@ATTENTION.register_module()
class CustomMultiHeadCrossAttention(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_heads=8,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 attn_drop=0.0,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super(CustomMultiHeadCrossAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.wq = nn.Linear(embed_dims, embed_dims)
        self.wk = nn.Linear(embed_dims, embed_dims)
        self.wv = nn.Linear(embed_dims, embed_dims)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.wq, distribution='uniform', bias=0.)
        xavier_init(self.wk, distribution='uniform', bias=0.)
        xavier_init(self.wv, distribution='uniform', bias=0.)
        xavier_init(self.out_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None
                ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # (L, B, C)
        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        # query = query / torch.norm(query, p=2, dim=-1, keepdim=True)
        # key = key / torch.norm(key, p=2, dim=-1, keepdim=True)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        if torch.onnx.is_in_onnx_export():
            Lq, B, C = 900, 1, 256
            Lk = 2250
        else:
            Lq, B, C = query.shape
            Lk, _, _ = key.shape

        query = query / math.sqrt(C // self.num_heads)
        print(C, self.num_heads)
        print(math.sqrt(C // self.num_heads))

        query = query.contiguous().view(Lq, B * self.num_heads, C // self.num_heads).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)

        attn = torch.bmm(query, key.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxLpNormRelu(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_heads=8,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 attn_drop=0.0,
                 norm_p=2,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super(MHCAFreeSoftmaxLpNormRelu, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.wq = nn.Linear(embed_dims, embed_dims)
        self.wk = nn.Linear(embed_dims, embed_dims)
        self.wv = nn.Linear(embed_dims, embed_dims)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.norm_p = norm_p
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.wq, distribution='uniform', bias=0.)
        xavier_init(self.wk, distribution='uniform', bias=0.)
        xavier_init(self.wv, distribution='uniform', bias=0.)
        xavier_init(self.out_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None
                ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # (L, B, C)
        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        if self.norm_p == 'inf':
            normq = torch.max(torch.abs(query), dim=-1, keepdim=True)
            normk = torch.max(torch.abs(key), dim=-1, keepdim=True)
            query = query / normq[0]
            key = key / normk[0]
        else:
            query = query / torch.norm(query, p=self.norm_p, dim=-1, keepdim=True)
            key = key / torch.norm(key, p=self.norm_p, dim=-1, keepdim=True)

        if torch.onnx.is_in_onnx_export():
            Lq, B, C = 900, 1, 256
            Lk = 2250
        else:
            Lq, B, C = query.shape
            Lk, _, _ = key.shape

        query = query.contiguous().view(Lq, B * self.num_heads, C // self.num_heads).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        # attn = attn.softmax(dim=-1)
        attn = torch.nn.ReLU(inplace=True)(attn)
        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxReluL2Norm(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_heads=8,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 attn_drop=0.0,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super(MHCAFreeSoftmaxReluL2Norm, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.wq = nn.Linear(embed_dims, embed_dims)
        self.wk = nn.Linear(embed_dims, embed_dims)
        self.wv = nn.Linear(embed_dims, embed_dims)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.wq, distribution='uniform', bias=0.)
        xavier_init(self.wk, distribution='uniform', bias=0.)
        xavier_init(self.wv, distribution='uniform', bias=0.)
        xavier_init(self.out_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None
                ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # (L, B, C)
        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        # query = query / torch.norm(query, p=2, dim=-1, keepdim=True)
        # key = key / torch.norm(key, p=2, dim=-1, keepdim=True)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        Lq, B, C = query.shape
        Lk, _, _ = key.shape

        query = query.contiguous().view(Lq, B * self.num_heads, C // self.num_heads).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        # attn = attn.softmax(dim=-1)
        # attn = torch.nn.ReLU(inplace=True)(attn) + 1e-5
        attn = torch.nn.LeakyReLU(inplace=True)(attn)
        attn = attn / torch.norm(attn, p=2, dim=-1, keepdim=True)
        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxLpNormHeadRelu(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_heads=8,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 attn_drop=0.0,
                 norm_p=2,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super(MHCAFreeSoftmaxLpNormHeadRelu, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.wq = nn.Linear(embed_dims, embed_dims)
        self.wk = nn.Linear(embed_dims, embed_dims)
        self.wv = nn.Linear(embed_dims, embed_dims)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.norm_p = norm_p
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.wq, distribution='uniform', bias=0.)
        xavier_init(self.wk, distribution='uniform', bias=0.)
        xavier_init(self.wv, distribution='uniform', bias=0.)
        xavier_init(self.out_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None
                ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # (L, B, C)
        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        if torch.onnx.is_in_onnx_export():
            Lq, B, C = 900, 1, 256
            Lk = 2250
        else:
            Lq, B, C = query.shape
            Lk, _, _ = key.shape

        query = query.contiguous().view(Lq, B * self.num_heads, C // self.num_heads).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)

        if self.norm_p == 'inf':
            normq = torch.max(torch.abs(query), dim=-1, keepdim=True)
            normk = torch.max(torch.abs(key), dim=-1, keepdim=True)
            query = query / normq[0]
            key = key / normk[0]
        else:
            query = query / torch.norm(query, p=self.norm_p, dim=-1, keepdim=True)
            key = key / torch.norm(key, p=self.norm_p, dim=-1, keepdim=True)

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        # attn = attn.softmax(dim=-1)
        attn = torch.nn.ReLU(inplace=True)(attn)
        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxBNLNHeadRelu(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_heads=8,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 attn_drop=0.0,
                 #  norm_p=2,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super(MHCAFreeSoftmaxBNLNHeadRelu, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.wq = nn.Linear(embed_dims, embed_dims)
        self.wk = nn.Linear(embed_dims, embed_dims)
        self.wv = nn.Linear(embed_dims, embed_dims)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        # self.norm_p = norm_p
        self.normq = nn.LayerNorm(self.embed_dims // self.num_heads)
        # self.normk = nn.BatchNorm1d(self.embed_dims // self.num_head)
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.wq, distribution='uniform', bias=0.)
        xavier_init(self.wk, distribution='uniform', bias=0.)
        xavier_init(self.wv, distribution='uniform', bias=0.)
        xavier_init(self.out_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None
                ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # (L, B, C)
        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        print(query.shape, key.shape)

        if torch.onnx.is_in_onnx_export():
            Lq, B, C = 900, 1, 256
            Lk = 2250
        else:
            Lq, B, C = query.shape
            Lk, _, _ = key.shape

        query = query.contiguous().view(Lq, B * self.num_heads, C // self.num_heads).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_heads, C // self.num_heads).transpose(0, 1)

        query = self.normq(query)
        # key = self.normk(key)

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        # attn = attn.softmax(dim=-1)
        attn = torch.nn.ReLU(inplace=True)(attn)
        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)
