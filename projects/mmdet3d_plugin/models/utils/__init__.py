# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
DUMP = False

from .dgcnn_attn import DGCNNAttn
from .detr import Deformable3DDetrTransformerDecoder
from .detr3d_transformer import Detr3DTransformer, Detr3DTransformerDecoder, Detr3DCrossAtten
from .positional_encoding import SinePositionalEncoding3D, LearnedPositionalEncoding3D
from .petr_transformer import PETRTransformer, PETRDNTransformer, PETRMultiheadAttention, PETRTransformerEncoder, \
    PETRTransformerDecoder
from .attention_pooling import PoolingChannel, PoolingToken, ConvToken, ConvChannel, LinearChannel, PoolingChannel2
from .petr_transformer_opt import PETRTransformerDecoderLayerBN

if DUMP:
    from .simple_cross_attention_dump import CustomMultiHeadCrossAttention, MHCAFreeSoftmaxReluL2Norm, \
        MHCAFreeSoftmaxLpNormRelu
else:
    from .simple_cross_attention import CustomMultiHeadCrossAttention, MHCAFreeSoftmaxReluL2Norm, \
        MHCAFreeSoftmaxLpNormRelu, MHCAFreeSoftmaxLpNormHeadRelu, MHCAFreeSoftmaxBNLNHeadRelu

__all__ = ['DGCNNAttn', 'Deformable3DDetrTransformerDecoder', 
           'Detr3DTransformer', 'Detr3DTransformerDecoder', 'Detr3DCrossAtten',
           'SinePositionalEncoding3D', 'LearnedPositionalEncoding3D',
           'PETRTransformer', 'PETRDNTransformer', 'PETRMultiheadAttention', 
           'PETRTransformerEncoder', 'PETRTransformerDecoder',
           'PoolingChannel', 'PoolingToken',
           'ConvToken', 'ConvChannel', 'LinearChannel', 'PoolingChannel2',
           'CustomMultiHeadCrossAttention', 'MHCAFreeSoftmaxReluL2Norm', 'MHCAFreeSoftmaxLpNormRelu',
           'MHCAFreeSoftmaxLpNormHeadRelu', 'MHCAFreeSoftmaxBNLNHeadRelu',
           'PETRTransformerDecoderLayerBN'
           ]


