import torch
from torch import nn
from torch.nn import functional as F
# from .apnb import APNB


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]
    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        # (1,3,6,8)
        # (5,9,13)
        super(PSPModule, self).__init__()
        # print(sizes)
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
            # prior = Pooling(size)
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center



class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1,
                 norm_type=None, psp_size=(1, 3, 6, 8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        # 65536 110 (1,3,6,8)
        # 65536 275 (5,9,13)
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros(65536 ,
        #                 275))
        # self.attn = WindowAttention(
        #     256,
        #     window_size=to_2tuple(7),
        #     num_heads=num_heads,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     attn_drop=0.,
        #     proj_drop=0.)
        # self.context = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     APNB(in_channels=256, out_channels=256, key_channels=256, value_channels=256,
        #          dropout=0.05, sizes=([1]), )
        # )
        # self.cc =EfficientAttention(self.out_channels,self.out_channels,self.out_channels)

    def forward(self, low_feats, high_feats):
        batch_size, h, w = high_feats.size(0), high_feats.size(2), high_feats.size(3)
        # if self.scale > 1:
        #     x = self.pool(x)
        # a = high_feats.clone()
        # a = self.cc(a)
        value = self.psp(self.f_value(low_feats))

        query = self.f_query(high_feats).view(batch_size, self.key_channels, -1)
        # print(query.shape)
        query = query.permute(0, 2, 1)
        key = self.f_key(low_feats)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        # print(sim_map.shape)
        # sim_map = self.relative_position_bias_table.unsqueeze(0)+sim_map
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *high_feats.size()[2:])
        context = self.W(context)
        # context = context + a
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1,
                 norm_type=None, psp_size=(1, 3, 6, 8)):
        super(SelfAttentionBlock2D, self).__init__(low_in_channels,
                                                   high_in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_type,
                                                   psp_size=psp_size
                                                   )

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dw = nn.Conv2d(256, 256, 3, 1, 1, groups=256)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        # print(self.hidden_features)
        x = self.fc1(x)
        
        # print(x.shape)
        x = self.dw(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x
class AFNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout,
                 sizes=([1]), norm_type=None, psp_size=(1,3,6,8)):
        super(AFNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size = psp_size
        self.stages = nn.ModuleList(
            [self._make_stage([low_in_channels, high_in_channels], out_channels, key_channels, value_channels, size) for
             size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.mlp = Mlp(in_features=out_channels, hidden_features=out_channels * 4, act_layer=nn.GELU, drop=0.)

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels[0],
                                    in_channels[1],
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    psp_size=self.psp_size)

    def forward(self, low_feats, high_feats):
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, high_feats], 1))

        # output = output + self.mlp(self.norm1(output))
        return output