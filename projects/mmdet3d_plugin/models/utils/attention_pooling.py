from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule

import torch.nn as nn


@ATTENTION.register_module()
class PoolingChannel(BaseModule):
    def __init__(self,
                 pool_size=3,
                 batch_first=False,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 init_cfg=None):
        super(PoolingChannel, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.pool = nn.AvgPool1d(pool_size,
                                 stride=1,
                                 padding=pool_size // 2,
                                 count_include_pad=False)

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
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)

        output = self.pool(query)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


@ATTENTION.register_module()
class PoolingToken(BaseModule):
    def __init__(self,
                 pool_size=3,
                 batch_first=False,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 init_cfg=None):
        super(PoolingToken, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.pool = nn.AvgPool1d(pool_size,
                                 stride=1,
                                 padding=pool_size // 2,
                                 count_include_pad=False)

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
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 2, 0)
        else:
            query = query.permute(0, 2, 1)

        output = self.pool(query)

        if not self.batch_first:
            output = output.permute(2, 0, 1)
        else:
            output = output.permute(0, 2, 1)

        return output


@ATTENTION.register_module()
class ConvToken(BaseModule):
    def __init__(self,
                 kernel_size=3,
                 batch_first=False,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 init_cfg=None):
        super(ConvToken, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.conv = nn.Conv1d(embed_dims, embed_dims, kernel_size, stride=1, padding=kernel_size // 2)

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
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)

        query = query.permute(0, 2, 1)

        output = self.conv(query)

        output = output.permute(0, 2, 1)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


@ATTENTION.register_module()
class ConvChannel(BaseModule):
    def __init__(self,
                 kernel_size=3,
                 batch_first=False,
                 num_query=900,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 init_cfg=None):
        super(ConvChannel, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.batch_first = batch_first
        self.conv = nn.Conv1d(num_query, num_query, kernel_size, stride=1, padding=kernel_size // 2)

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
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)

        output = self.conv(query)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


@ATTENTION.register_module()
class LinearChannel(BaseModule):
    def __init__(self,
                 batch_first=False,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 init_cfg=None):
        super(LinearChannel, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.proj = nn.Linear(embed_dims, embed_dims)

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
        if query_pos is not None:
            query = query + query_pos

        # if not self.batch_first:
        #     query = query.permute(1, 0, 2)

        output = self.proj(query)

        # if not self.batch_first:
        #     output = output.permute(1, 0, 2)

        return output


@ATTENTION.register_module()
class PoolingChannel2(BaseModule):
    def __init__(self,
                 pool_size=3,
                 batch_first=False,
                 embed_dims=256,  # for transformer layer embed_dims confirm
                 init_cfg=None):
        super(PoolingChannel2, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.pool = nn.AvgPool1d(pool_size,
                                 stride=1,
                                 padding=pool_size // 2,
                                 count_include_pad=True)

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
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)

        output = self.pool(query)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output