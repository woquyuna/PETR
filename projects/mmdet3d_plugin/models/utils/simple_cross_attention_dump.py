from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.drop import build_dropout

import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import xavier_init, constant_init
import math

import time

output_path = '/mnt/apollo/userdata/hjj/petr_analysis/'


@ATTENTION.register_module()
class CustomMultiHeadCrossAttention(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_head=8,
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
        self.num_head = num_head
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
        timestamp = int(time.time())
        time_tag = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp))

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

        query = query.contiguous().view(Lq, B * self.num_head, C // self.num_head).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        # save qkv
        query.clone().cpu().detach().numpy().tofile(
            output_path + '/customMHCA/' + 'query_{}x{}x{}_'.format(B * self.num_head, Lq, C // self.num_head) + time_tag + '.bin')
        key.clone().cpu().detach().numpy().tofile(
            output_path + '/customMHCA/' + 'key_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')
        value.clone().cpu().detach().numpy().tofile(
            output_path + '/customMHCA/' + 'value_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_head)
        # save attn
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/customMHCA/' + 'attn_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = attn.softmax(dim=-1)
        # save softmax
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/customMHCA/' + 'softmax_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)
        # save output
        output.clone().cpu().detach().numpy().tofile(
            output_path + '/customMHCA/' + 'output_{}x{}x{}_'.format(Lq, B, C) + time_tag + '.bin')

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxLpNormRelu(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_head=8,
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
        self.num_head = num_head
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

        timestamp = int(time.time())
        time_tag = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp))

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

        Lq, B, C = query.shape
        Lk, _, _ = key.shape

        # save qkv
        query.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'query_{}x{}x{}_'.format(B * self.num_head, Lq, C // self.num_head) + time_tag + '.bin')
        key.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'key_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')
        value.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'value_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')

        if self.norm_p == 'inf':
            normq = torch.max(torch.abs(query), dim=-1, keepdim=True)
            normk = torch.max(torch.abs(key), dim=-1, keepdim=True)
            query = query / normq[0]
            key = key / normk[0]
        else:
            query = query / torch.norm(query, p=self.norm_p, dim=-1, keepdim=True)
            key = key / torch.norm(key, p=self.norm_p, dim=-1, keepdim=True)

        query = query.contiguous().view(Lq, B * self.num_head, C // self.num_head).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        # save qkv norm
        query.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'queryNorm_{}x{}x{}_'.format(B * self.num_head, Lq, C // self.num_head) + time_tag + '.bin')
        key.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'keyNorm_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')
        value.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'valueNorm_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_head)
        # save attn
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'attn_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = torch.nn.ReLU(inplace=True)(attn)
        # save relu
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'relu_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)
        # save output
        output.clone().cpu().detach().numpy().tofile(
            output_path + '/LinfNormRelu/' + 'output_{}x{}x{}_'.format(Lq, B, C) + time_tag + '.bin')

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxReluL2Norm(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_head=8,
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
        self.num_head = num_head
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
        timestamp = int(time.time())
        time_tag = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp))

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

        query = query.contiguous().view(Lq, B * self.num_head, C // self.num_head).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        # save qkv
        query.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'query_{}x{}x{}_'.format(B * self.num_head, Lq, C // self.num_head) + time_tag + '.bin')
        key.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'key_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')
        value.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'value_{}x{}x{}_'.format(B * self.num_head, Lk, C // self.num_head) + time_tag + '.bin')

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_head)
        # save attn
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'attn_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = torch.nn.LeakyReLU(inplace=True)(attn)
        # save leakyRelu
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'leakyRelu_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = attn / torch.norm(attn, p=2, dim=-1, keepdim=True)
        # save L2norm
        attn.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'L2norm_{}x{}x{}_'.format(B * self.num_head, Lq, Lk) + time_tag + '.bin')

        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)
        # save output
        output.clone().cpu().detach().numpy().tofile(
            output_path + '/ReluL2Norm/' + 'output_{}x{}x{}_'.format(Lq, B, C) + time_tag + '.bin')

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)

@ATTENTION.register_module()
class MHCAFreeSoftmaxAttentionRelu(BaseModule):
    def __init__(self,
                 batch_first=False,
                 num_head=8,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 attn_drop=0.0,
                 #  norm_p=2,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super(MHCAFreeSoftmaxAttentionRelu, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.batch_first = batch_first
        self.wq = nn.Linear(embed_dims, embed_dims)
        self.wk = nn.Linear(embed_dims, embed_dims)
        self.wv = nn.Linear(embed_dims, embed_dims)
        # self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        # self.norm_p = norm_p
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

        Lq, B, C = query.shape
        Lk, _, _ = key.shape

        query = query.contiguous().view(Lq, B * self.num_head, C // self.num_head).transpose(0, 1)  # (B`,L, C`)
        key = key.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)
        value = value.contiguous().view(Lk, B * self.num_head, C // self.num_head).transpose(0, 1)

        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(C // self.num_head)
        attn = torch.nn.ReLU(inplace=True)(attn)

        output = torch.bmm(attn, value) * self.temperature  # (B, Lq, C)

        output = output.transpose(0, 1).contiguous().view(Lq, B, C)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = self.out_proj(output)

        return identity + self.dropout_layer(output)



