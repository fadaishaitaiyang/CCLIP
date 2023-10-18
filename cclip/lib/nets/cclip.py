from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math
from lib.nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.nets.encoder import encoder
from .apnb import AFNB
from .afnb import APNB
from .clipvit import CLIPVisionTransformer, CLIPTextEncoder, ContextDecoder, CLIPResNetWithAttention
from .untils import tokenize


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# class SHHFusion(nn.Module):
#     def __init__(self):
#         super(SHHFusion, self).__init__()
#         self.fu1 = AFNB(1024,1024,1024,1024,1024,dropout=0.05, sizes=([1]))
#         self.context = nn.Sequential(
#             nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             APNB(in_channels=1024, out_channels=1024, key_channels=1024, value_channels=1024,
#                  dropout=0.05, sizes=([1]), )
#         )
#         # self.norm = nn.LayerNorm(1024)
#
#
#     def forward(self,scene_feat,face_feat,head):
#         B, C, H, W = scene_feat.shape
#         head = head.reshape(B, C, H, W)
#         fu1 = self.fu1(head,scene_feat)
#         fu2 = self.fu1(head,face_feat)
#         scene_feat = fu1 + fu2
#         input_ = scene_feat
#         scene_feat = self.context(scene_feat)
#         scene_feat = scene_feat + input_
#
#         return scene_feat
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, value_channels, 1)
        self.norm = nn.LayerNorm(value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.reshape(B, C, H * W).permute(0, 2, 1)
        x2 = x2.reshape(B, C, H * W).permute(0, 2, 1)
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        # reprojected_value = self.reprojection(aggregated_values).reshape(B,  D, N).permute(0, 2, 1)
        # reprojected_value = self.norm(reprojected_value)
        # reprojected_value = reprojected_value.permute(0,2,1).reshape(B,C,H,W)

        return aggregated_values


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)



    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x, y):
        q, k, v = self.conv1(x), self.conv2(y), self.conv3(y)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


class SHHFusion(nn.Module):
    def __init__(self):
        super(SHHFusion, self).__init__()
        self.fu1 = AFNB(1024, 1024, 1024, 1024, 1024, dropout=0.05, sizes=([1]))
        self.context = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            APNB(in_channels=1024, out_channels=1024, key_channels=1024, value_channels=1024,
                 dropout=0.05, sizes=([1]), )
        )
        # self.ln = ACmix(1024, 1024)
        self.cn = nn.Conv2d(2072, 1024, 1)
        self.cn1 = nn.Conv2d(2048, 1024, 1)
        self.norm2 = nn.LayerNorm(1024)
        mlp_hidden_dim = int(1024 * 4)
        self.mlp = Mlp(in_features=1024, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        # self.norm = nn.LayerNorm(1024)

    def forward(self, scene_feat, face_feat, head):
        scene_feat = self.cn(scene_feat)
        face_feat = self.cn1(face_feat)
        B, C, H, W = scene_feat.shape
        tt = scene_feat
        scene_feat = scene_feat.view(B,H*W,C)
        scene_feat = self.norm2(scene_feat)
        scene_feat = scene_feat.view(B,C,H,W)
        face_feat = face_feat.contiguous().view(B, H*W, C)
        face_feat = self.norm2(face_feat)
        face_feat = face_feat.view(B, C, H, W)
        
        head = head.reshape(B, C, H, W)
        scene_feat = self.fu1( scene_feat,face_feat,)
        # scene_feat = self.context(scene_feat)
        scene_feat = tt + scene_feat
        input_ = scene_feat
        scene_feat = scene_feat.view(B, H * W, C)
        scene_feat = self.norm2(scene_feat)
        scene_feat = scene_feat.view(B, C, H, W)
        scene_feat = self.mlp(scene_feat)
        scene_feat = scene_feat + input_

        rr = scene_feat
        scene_feat = scene_feat.view(B, H * W, C)
        scene_feat = self.norm2(scene_feat)
        scene_feat = scene_feat.view(B, C, H, W)

        head = head.view(B, H * W, C)
        head = self.norm2(head)
        head = head.view(B, C, H, W)
        
        scene_feat = self.fu1(scene_feat,head, )
        # scene_feat = self.context(scene_feat)
        scene_feat = scene_feat + rr

        input1_ = scene_feat
        scene_feat = scene_feat.view(B, H * W, C)
        scene_feat = self.norm2(scene_feat)
        scene_feat = scene_feat.view(B, C, H, W)
        scene_feat = self.mlp(scene_feat)
        scene_feat = scene_feat + input1_


        return scene_feat




def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class Pixel_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel_shuffle, self).__init__()

        self.pixel_shuffle = nn.Sequential(
            nn.PixelShuffle(2),
            conv2d(in_channels, out_channels, 3)
        )

    def forward(self, x, ):
        x = self.pixel_shuffle(x)
        return x


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[1], 1),
        conv2d(filters_list[1], filters_list[0], 3),
        conv2d(filters_list[0], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class CCLip(nn.Module):
    def __init__(self, anchors_mask, num_classes, train_mode):
        super(CCLip, self).__init__()

        # self.backbone = Resnet(phi='resnet50', pretrained=False)
        # self.backbone = darknet53(None)
        if train_mode == 0:
            self.conv1 = make_three_conv([512, 1024], 1024)
            self.SPP = SpatialPyramidPooling()
            self.conv2 = make_three_conv([512, 1024], 2048)

            self.upsample1 = Upsample(512, 256)
            self.conv_for_P4 = conv2d(512, 256, 1)
            self.make_five_conv1 = make_five_conv([256, 512], 512)

            self.upsample2 = Upsample(256, 128)
            self.conv_for_P3 = conv2d(256, 128, 1)
            self.make_five_conv2 = make_five_conv([128, 256], 256)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

            self.down_sample1 = conv2d(128, 256, 3, stride=2)
            self.make_five_conv3 = make_five_conv([256, 512], 512)

            self.down_sample2 = conv2d(256, 512, 3, stride=2)
            self.make_five_conv4 = make_five_conv([512, 1024], 1024)

            # Pixle_shuffle
            self.ps1 = Pixel_shuffle(128, 256)
            self.ps2 = Pixel_shuffle(64, 128)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.c5_conv = nn.Conv2d(512, 128, (1, 1))
            self.c4_conv = nn.Conv2d(256, 128, (1, 1))
            self.c3_conv = nn.Conv2d(128, 128, (1, 1))
            self.p5_conv = nn.Conv2d(128, 128, (3, 3), padding=1)
            self.p4_conv = nn.Conv2d(128, 128, (3, 3), padding=1)
            self.p3_conv = nn.Conv2d(128, 128, (3, 3), padding=1)
        else:
            self.conv1 = make_three_conv([512, 1024], 1024)
            self.SPP = SpatialPyramidPooling()
            self.conv2 = make_three_conv([512, 1024], 2048)

            self.upsample1 = Upsample(512, 256)
            self.conv_for_P4 = conv2d(512, 256, 1)
            self.make_five_conv1 = make_five_conv([256, 512], 512)

            self.upsample2 = Upsample(256, 128)
            self.conv_for_P3 = conv2d(256, 128, 1)
            self.make_five_conv2 = make_five_conv([128, 256], 256)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

            self.down_sample1 = conv2d(128, 256, 3, stride=2)
            self.make_five_conv3 = make_five_conv([256, 512], 512)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

            self.down_sample2 = conv2d(256, 512, 3, stride=2)
            self.make_five_conv4 = make_five_conv([512, 1024], 1024)

            # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
            self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

        # GOO network
        # common
        self.context_length = 13
        class_names = (
        'Kellogg Corn Flakes', 'Honey Stars', 'Koko Krunch', 'Fita', 'Mamon', 'Buttercake', 'Wafer', 'Choco Crunchies',
        'iHop', 'Butterlicious', 'Pik-nik', 'Pringles',
        'Super Fruits', 'Locally Mango', 'Camomille', 'Oishi Prawn', 'Kimchi Noodles', 'Grahams', 'Hansel', 'Danisa',
        'Chocolate Chips', 'Fresh Milk', 'Bear brand', 'Chuckie',)
        # self.backbone = CLIPVisionTransformer(patch_size=16,
        #                                     width=768,
        #                                 output_dim=1024,
        #                                 get_embeddings=True,
        #                                 drop_path_rate=0.1,
        #                                 layers=12,
        #                                 out_indices=[3,5,7,11],
        #                                 pretrained='./ViT-B-16.pt')
        self.backbone = CLIPResNetWithAttention(layers=[3, 4, 6, 3], output_dim=1024, input_resolution=224,
                                                style='pytorch', pretrained='./RN50.pt')
        self.texts = torch.cat([tokenize(f"a photo of a people looks at one {c} on a shelf") for c in class_names])
        # self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.text_encoder = CLIPTextEncoder(context_length=77, embed_dim=1024, transformer_width=1024,
                                            transformer_heads=8,
                                            transformer_layers=12, style='pytorch', pretrained='./ViT-B-16.pt')
        self.context_decoder = ContextDecoder(transformer_width=512, transformer_heads=4, transformer_layers=3,
                                              visual_dim=1024,
                                              dropout=0.1, style='pytorch')
        # learnable textual contexts
        token_embed_dim = 1024
        text_dim = 1024
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, 5, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        # for p in self.se.parameters():
        #     p.requires_grad_(False)
        #     # print('oK!')
        self.fpn1 = nn.Sequential(
            nn.GroupNorm(1, 768),
            nn.ConvTranspose2d(768, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.GroupNorm(1, 768),
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Sequential(
            nn.GroupNorm(1, 768),
            nn.ConvTranspose2d(768, 512, kernel_size=1, stride=1),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(768, 768 * 4),
            QuickGELU(),
            nn.Linear(768 * 4, 768)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(768, 768 * 4),
            QuickGELU(),
            nn.Linear(768 * 4, 768)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(768, 768 * 4),
            QuickGELU(),
            nn.Linear(768 * 4, 768)
        )
        width = 768
        scale = width ** -0.5
        get_embeddings = True
        self.get_embeddings = get_embeddings
        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, 1024))
            self.max = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.head = _FCNHead(1024, 512)
        self.ss = SHHFusion()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # head pathway
        # use e[1]
        # self.conv_face_scene = nn.Conv2d(2584, 2048, kernel_size=1)
        # use x[3]
        self.conv_face_scene = nn.Conv2d(4632, 2048, kernel_size=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU()
        )
        # attention
        # use e[1]
        # self.attn = nn.Linear(1808, 1 * 7 * 7)
        # use x[3]
        self.attn = nn.Linear(2832, 1 * 7 * 7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        self.p1 = nn.Parameter(torch.randn(32, 24, 512))
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.aaa = nn.Conv2d(in_channels=1049, out_channels=1024, kernel_size=1, bias=False)
        self.fc1 = nn.Sequential(
            nn.Linear(2072, 2072 * 4),
            QuickGELU(),
            nn.Linear(2072 * 4, 2072)
        )
        self.fc2 = nn.Linear(2048, 2048)

    def forward(self, images, head, face, train_mode):

        if train_mode == 0:
            #  backbone
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x2, x1, x0, d, e = list(self.backbone(
                images))  # e[0] (16,1024) e[1] (16,1024,7,7) x0(197,16,768) x1(197,16,768) x2(197,16,768) d(197,16,768)
            global_feat, visual_embeddings = e
            B, C, H, W = visual_embeddings.shape
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, N, C  visual_context (16,50=7*7+1,1024)

            text_embeddings = self.text_encoder(self.texts.to(device), self.contexts).expand(32, -1,
                                                                                             -1)  # text_embeddings(16,25,1024)
            # print(B) 32 
            # print(text_embeddings.shape) [32, 24, 1024]
            # text_embeddings = torch.cat((text_embeddings,self.p1),dim=1) #text_embeddings(32,25+24,512)
            # print(text_embeddings.shape) # [32, 25, 512]
            # text_embeddings = text_embeddings+e[1]
            text_diff = self.context_decoder(text_embeddings, visual_context)  # text_diff(16,25,1024)

            text_embeddings = text_embeddings + self.gamma * text_diff  # text_embeddings(16,25,1024)
            text_features = F.normalize(text_embeddings, dim=-1)
            visual_embeddings = F.normalize(visual_embeddings, dim=1)
            score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / 0.07  # (16,25,7,7)
            # scene_feat = torch.cat([visual_embeddings,score_map3],dim=1)
            scene_feat = torch.cat([d, score_map3], dim=1)
            # scene_feat = scene_feat.view(scene_feat.size(0), -1)
            # scene_feat = self.fc1(scene_feat)

            # score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
            # score_map0 = F.upsample(score_map3, x[0].shape[2:], mode='bilinear')
            # score_maps = [score_map0, None, None, score_map3]
            # x[3] = torch.cat([x[3], score_maps[3]], dim=1)
            # print(text_diff.shape)
            # text_diff = text_diff.reshape(32,7,7,1024).permute(0,3,1,2)

            # print(text_diff.shape) # [32, 1024, 7, 7]
            # text = self.compute_text_features(e,dummy=False)
            # print(e[1].shape)
            _, _, _, face_feat, e1 = list(self.backbone(face))
            # face_feat = self.fc2(face_feat)
            # _,face_feat = e1

            # tm = x2
            # x2 = self.mlp1(x2)
            # x2 = tm + x2
            # x2 = x2.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            # x2 = self.fpn1(x2)
            #
            # tm1 = x1
            # x1 = self.mlp2(x1)
            # x1 = tm1 + x1
            # x1 = x1.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            # x1 = self.fpn2(x1)
            #
            # tm0 = x0
            # x0 = self.mlp3(x0)
            # x0 = x0 + tm0
            # x0 = x0.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            # x0 = self.fpn3(x0)
            # a,b,c,face_feat = list(self.se(face))

            # print(a.shape)#(768,14,14) self.fpn1(768,56,56)
            # print(b.shape)#(768,14,14)
            # print(c.shape)#(768,14,14)
            # print(d.shape)#(768,14,14)
            # print(e.shape) #global_embedding(32,512)
            # print(f.shape) #visual_embedding([32, 512, 14, 14])
            # scene_feat, face_feat, x2, x1, x0 = self.backbone(images, face, train_mode)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
            # if self.get_embeddings:
            #     e = e.permute(1, 0, 2)
            #     e = self.ln_post(e)
            #     e = e @ self.proj

            #     scene_feat = e[:, 1:].reshape(32, 14,14, -1).permute(0, 3, 1, 2)  # B C H W
            #     scene_feat = self.max(scene_feat)
            # # print(scene_feat.shape)
            # if self.get_embeddings:
            #     e1 = e1.permute(1, 0, 2)
            #     e1 = self.ln_post(e1)
            #     e1 = e1 @ self.proj

            #     face_feat = e1[:, 1:].reshape(32, 14,14, -1).permute(0, 3, 1, 2)  # B C H W
            #     face_feat = self.max(face_feat)
            # print(scene_feat.shape)
            # p5 = self.c5_conv(x0)
            # p4 = self.upsample(p5) + self.c4_conv(x1)
            # p3 = self.upsample(p4) + self.c3_conv(x2)
            # P5 = self.relu(self.p5_conv(p5))
            # P4 = self.relu(self.p4_conv(p4))
            # P3 = self.relu(self.p3_conv(p3))
            P5 = self.conv1(x0)
            testP5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
            P5 = self.SPP(P5)
            # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.conv2(P5)

            # 13,13,512 -> 13,13,256 -> 26,26,256
            P5_upsample = self.ps1(P5)  # self.upsample1(P5)
            # 26,26,512 -> 26,26,256
            P4 = self.conv_for_P4(x1)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P4, P5_upsample], axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv1(P4)

            # 26,26,256 -> 26,26,128 -> 52,52,128
            P4_upsample = self.ps2(P4)  # self.upsample2(P4)
            # 52,52,256 -> 52,52,128
            P3 = self.conv_for_P3(x2)
            # 52,52,128 + 52,52,128 -> 52,52,256
            P3 = torch.cat([P3, P4_upsample], axis=1)
            # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
            P3 = self.make_five_conv2(P3)

            out2 = self.yolo_head3(P3)

            # GOO
            # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
            # head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
            # # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            # face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
            # # get and reshape attention weights such that it can be multiplied with scene feature map
            # # xz=torch.cat(head_reduced, face_feat_reduced)
            # # x0=self.attn(x,1)
            # attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            # attn_weights = attn_weights.view(-1, 1, 49)
            # attn_weights = F.softmax(attn_weights, dim=2)  # soft attention weights single-channel
            # attn_weights = attn_weights.view(-1, 1, 7, 7)

            # head_conv = self.conv_block(head)

            # # attn_weights = torch.ones(attn_weights.shape)/49.0
            # scene_feat = torch.cat((scene_feat, head_conv), 1)
            # attn_applied_scene_feat = torch.mul(attn_weights,
            #                                     scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

            # scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
            # scene_face_feat = self.conv_face_scene(scene_face_feat)

            # scene + face feat -> encoding -> decoding
            # head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
            # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            # face_feat_reduced = self.avgpool(face_feat).view(-1, 2048)
            # get and reshape attention weights such that it can be multiplied with scene feature map
            # xz=torch.cat(head_reduced, face_feat_reduced)
            # x0=self.attn(x,1)
            # attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            # attn_weights = attn_weights.view(-1, 1, 49)
            # attn_weights = F.softmax(attn_weights, dim=2)  # soft attention weights single-channel
            # attn_weights = attn_weights.view(-1, 1, 7, 7)
            #
            # head_conv = self.conv_block(head)
            #
            # # attn_weights = torch.ones(attn_weights.shape)/49.0
            # scene_feat = torch.cat((scene_feat, head_conv), 1)
            # attn_applied_scene_feat = torch.mul(attn_weights,
            #                                     scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat
            #
            # scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
            # scene_face_feat = self.conv_face_scene(scene_face_feat)

            # scene + face feat -> encoding -> decoding
            # print(scene_feat.shape) ([16, 2072, 7, 7])
            # print(face_feat.shape)([16, 2048, 7, 7])
            scene_face_feat = self.ss(scene_feat, face_feat, head)
            encoding = self.compress_conv1(scene_face_feat)
            encoding = self.compress_bn1(encoding)
            encoding = self.relu(encoding)
            encoding = self.compress_conv2(encoding)
            encoding = self.compress_bn2(encoding)
            encoding = self.relu(encoding)
            x = self.deconv1(encoding)
            x = self.deconv_bn1(x)
            x = self.relu(x)
            x = self.deconv2(x)
            x = self.deconv_bn2(x)
            x = self.relu(x)
            x = self.deconv3(x)
            x = self.deconv_bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
            print(x.shape)

            return x, out2
        else:
            #  backbone
           
            x2, x1, x0, x, e = list(self.backbone(
                images))  # e[0] (16,1024) e[1] (16,1024,7,7) x0(197,16,768) x1(197,16,768) x2(197,16,768) d(197,16,768)
            
            # x2, x1, x0 = self.backbone(images, face, train_mode)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
            P5 = self.conv1(x0)
            P5 = self.SPP(P5)
            # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.conv2(P5)

            # 13,13,512 -> 13,13,256 -> 26,26,256
            P5_upsample = self.upsample1(P5)
            # 26,26,512 -> 26,26,256
            P4 = self.conv_for_P4(x1)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P4, P5_upsample], axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv1(P4)

            # 26,26,256 -> 26,26,128 -> 52,52,128
            P4_upsample = self.upsample2(P4)
            # 52,52,256 -> 52,52,128
            P3 = self.conv_for_P3(x2)
            # 52,52,128 + 52,52,128 -> 52,52,256
            P3 = torch.cat([P3, P4_upsample], axis=1)
            # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
            P3 = self.make_five_conv2(P3)

            # 52,52,128 -> 26,26,256
            P3_downsample = self.down_sample1(P3)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P3_downsample, P4], axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv3(P4)

            # 26,26,256 -> 13,13,512
            P4_downsample = self.down_sample2(P4)
            # 13,13,512 + 13,13,512 -> 13,13,1024
            P5 = torch.cat([P4_downsample, P5], axis=1)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.make_five_conv4(P5)

            # ---------------------------------------------------#
            #   第三个特征层
            #   y3=(batch_size,75,52,52)
            # ---------------------------------------------------#
            out2 = self.yolo_head3(P3)
            # ---------------------------------------------------#
            #   第二个特征层
            #   y2=(batch_size,75,26,26)
            # ---------------------------------------------------#
            out1 = self.yolo_head2(P4)
            # ---------------------------------------------------#
            #   第一个特征层
            #   y1=(batch_size,75,13,13)
            # ---------------------------------------------------#
            out0 = self.yolo_head1(P5)

            x = 0

            return x, out0, out1, out2
