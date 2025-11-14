
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


def build_upsample_layer(cfg):
    if cfg['type'] == 'deconv':
        in_channels = cfg['in_channels']
        out_channels = cfg['out_channels']
        kernel_size = cfg['kernel_size']
        stride = cfg['stride']
        padding = (kernel_size - 1) // 2

        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

def adj_index(h, k, node_num):
    dist = torch.cdist(h, h, p=2)     
    each_adj_index = torch.topk(dist, k, dim=2).indices   
    adj = torch.zeros(     
        h.size(0), node_num, node_num,
        dtype=torch.int, device=h.device, requires_grad=False
    ).scatter_(dim=2, index = each_adj_index, value=1)   
    return adj


class GraphAttentionLayer(nn.Module):


    def __init__(self, in_features, out_features, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵W和注意力机制参数a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.activation = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self._init_weights()

    # 帮助模型稳定训练
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)    # 计算h和W的矩阵乘积
        e = self._prepare_attentional_mechanism_input(Wh)
        # 设置未连接节点分数设置非常低的-9e15，以便在接下来的softmax中趋于0
        attention = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # 计算注意力参数a对应的两部分与Wh的矩阵乘积
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)   # 将Wh1和Wh2转置后相加，形成注意力的未标准化分数
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class OCGA(nn.Module):
    
    def __init__(self, in_feature, out_feature, top_k=11, token=3, alpha=0.2, num_heads=1):
        super(OCGA, self).__init__()
        self.top_k = top_k
        hidden_feature = in_feature
        self.conv = nn.Sequential(
            nn.Conv2d(in_feature, hidden_feature, token, stride=token),
            nn.BatchNorm2d(hidden_feature),
            nn.ReLU(inplace=True)
        )
        self.attentions = [
            GraphAttentionLayer(
                hidden_feature, hidden_feature, alpha=alpha, concat=True
            ) for _ in range(num_heads)
        ]

        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 定义一个输出的注意力层
        self.out_att = GraphAttentionLayer(
            hidden_feature * num_heads, out_feature, alpha=alpha, concat=False)

        self.deconv = build_upsample_layer(
            cfg=dict(type='deconv',
                     in_channels=out_feature, out_channels=out_feature,
                     kernel_size=token, stride=token)
        )
        self.activation = nn.ELU()
        self._init_weights()

    
    def _init_weights(self):
        for m in [self.deconv]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv(x)
        batch_size, in_feature, column, row = h.shape
        # 计算节点数量
        node_num = column * row
        h = h.view(
            batch_size, in_feature, node_num).permute(0, 2, 1)
        adj = adj_index(h, self.top_k, node_num)

        # 如有多头的设置，需要拼接
        h = torch.cat([att(h, adj) for att in self.attentions], dim=2)
        h = self.activation(self.out_att(h, adj))

        h = h.view(batch_size, column, row, -1).permute(0, 3, 1, 2)
        h = F.interpolate(
            self.deconv(h), x.shape[-2:], mode='bilinear', align_corners=True)
        return F.relu(h + x)

class Attention_Map_unit(nn.Module):
    def __init__(self, F_int):
        super(Attention_Map_unit, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_int, F_int // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int // 2)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_int, F_int // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int // 2)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)

        psi = self.psi(psi)

        return psi

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))

        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))

        return x


class PE_attention(nn.Module):
    def __init__(self, dim):
        super(PE_attention, self).__init__()
        self.pe_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pe_conv(x))

class Sp_BatchNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(Sp_BatchNorm, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

class SFF(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, k_size=3, drop=0.):
        super(SFF, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features),
            nn.BatchNorm2d(hidden_features),
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=(k_size + (k_size - 1) * (1 - 1)) // 2,
            dilation=1, groups=hidden_features),
            nn.BatchNorm2d(hidden_features),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=(k_size + (k_size - 1) * (2 - 1)) // 2,
            dilation=2, groups=hidden_features),
            nn.BatchNorm2d(hidden_features),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=(k_size + (k_size - 1) * (3 - 1)) // 2,
            dilation=3, groups=hidden_features),
            nn.BatchNorm2d(hidden_features),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=3, stride=1, padding=1, groups=hidden_features * 3),
            nn.BatchNorm2d(hidden_features * 3),
        )

        self.fc2 = nn.Conv2d(hidden_features * 4, out_features, 1, 1)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)

        x = self.conv3x3(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)
        x3 = self.conv3_3(x)
        x_sum = torch.cat((x1, x2, x3), dim=1)
        x_sum = self.conv3(x_sum)
        x = self.fc2(torch.cat((x, x_sum), dim=1))

        return x

class MFFE(nn.Module):

    def __init__(self, in_channels, att_dim=None, key_channels_redution=2, num_heads=16, qkv_bias=False, window_size=8,
                 with_pos=True, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, drop_path_rate=0., drop=0.):
        super(MFFE, self).__init__()
        self.in_channels = in_channels
        if att_dim is None:
            att_dim = in_channels
        self.att_dim = att_dim
        self.key_channels_reduction = key_channels_redution
        self.num_heads = num_heads
        head_dim = att_dim // self.num_heads
        self.channel_scale = (head_dim // key_channels_redution) ** -0.5
        self.qkv_bias = qkv_bias
        self.ws = window_size
        self.with_pos = with_pos

        if self.with_pos == True:
            self.pos = PE_attention(in_channels)

        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=att_dim // key_channels_redution,
                                    kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=att_dim // key_channels_redution, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=att_dim, kernel_size=1)

        self.head_proj = nn.Conv2d(att_dim, att_dim, 1, 1)
        self.proj = Sp_BatchNorm(att_dim, att_dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm1 = norm_layer(att_dim)
        self.norm2 = norm_layer(att_dim)

        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.sff = SFF(in_features=att_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps, 0, 0), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape

        if self.with_pos:
            x = self.pos(x)

        stage1_shorcut = x

        x = self.norm1(x)

        q = self.conv_query(x)
        q = rearrange(q, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                      h=self.num_heads,
                      d=self.att_dim // self.key_channels_reduction // self.num_heads,
                      hh=Hp // self.ws,
                      ww=Wp // self.ws,
                      ws1=self.ws,
                      ws2=self.ws)

        k = self.conv_key(x)
        k = rearrange(k, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                      h=self.num_heads,
                      d=self.att_dim // self.key_channels_reduction // self.num_heads,
                      hh=Hp // self.ws,
                      ww=Wp // self.ws,
                      ws1=self.ws,
                      ws2=self.ws)

        v = self.conv_value(x)
        v = rearrange(v, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                      h=self.num_heads,
                      d=self.att_dim // self.num_heads,
                      hh=Hp // self.ws,
                      ww=Wp // self.ws,
                      ws1=self.ws,
                      ws2=self.ws)

        q = nn.functional.softmax(q, dim=-1)
        k = nn.functional.softmax(k, dim=-2)
        context = k.permute(0, 1, 3, 2) @ v
        attn_spatial = q @ context

        f_channel = (q.transpose(-2, -1) @ k).transpose(-2, -1) * self.channel_scale
        f_channel_max = nn.functional.adaptive_max_pool2d(f_channel, 1)
        f_channel_avg = nn.functional.adaptive_avg_pool2d(f_channel, 1)
        f_channel = f_channel_avg + f_channel_max

        attn_sptial_channel = attn_spatial * f_channel

        attn_sptial_channel = rearrange(attn_sptial_channel, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
                                        h=self.num_heads,
                                        d=self.att_dim // self.num_heads,
                                        hh=Hp // self.ws,
                                        ww=Wp // self.ws,
                                        ws1=self.ws,
                                        ws2=self.ws)

        stage1 = attn_sptial_channel[:, :, :H, :W]
        stage1 = self.head_proj(stage1)

        stage1_shorcut = stage1_shorcut[:, :, :H, :W]
        stage1 = self.drop_path(stage1) + stage1_shorcut

        stage2_shortcut = stage1

        stage2 = self.norm2(stage1)

        stage2 = self.sff(stage2)
        x = stage2_shortcut + self.drop_path(stage2)

        return x


class TDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(TDecoderBlock, self).__init__()
        self.tdecoder = MFFE(in_channels=in_channels // 4, key_channels_redution=4, num_heads=16 // 4, window_size=16, with_pos=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.tdecoder(x)

        x = self.relu2(self.norm2(self.deconv2(x)))

        x = self.relu3(self.norm3(self.conv3(x)))

        return x


class MADSNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MADSNet, self).__init__()

        filters = [64, 128, 256, 512]
        # filters = [32, 64, 128, 256]
        # filters = [96, 192, 384, 768]
        # filters = [48, 96, 192, 384]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #####----------------*-*-*--------------------
        #####-reshape

        self.ocga1 = OCGA(filters[0], filters[0])
        self.ocga2 = OCGA(filters[1], filters[1])
        self.ocga3 = OCGA(filters[2], filters[2])
        self.ocga4 = OCGA(filters[3], filters[3])

        self.decoder4 = TDecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        ## multi-level attention model ##
        self.down4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(filters[2], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )

        self.maxpooling1 = nn.MaxPool2d(2)

        self.attention4 = Attention_Map_unit(filters[0])


        self.refine4 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )

        self.predict4_2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.predict3_2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.predict2_2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.predict1_2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x_ = x
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e1 = self.ocga1(e1)
        e2 = self.ocga2(e2)
        e3 = self.ocga3(e3)
        e4 = self.ocga4(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        ##########################
        down4 = F.upsample(self.down4(d4), size=d2.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(d3), size=d2.size()[2:], mode='bilinear')
        down2 = self.down2(d2)
        down1 = self.maxpooling1(self.down1(d1))

        #########
        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(down4, fuse1)
        attention3 = self.attention4(down3, fuse1)
        attention2 = self.attention4(down2, fuse1)
        attention1 = self.attention4(down1, fuse1)

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        refine4 = F.upsample(refine4, size=x_.size()[2:], mode='bilinear')
        refine3 = F.upsample(refine3, size=x_.size()[2:], mode='bilinear')
        refine2 = F.upsample(refine2, size=x_.size()[2:], mode='bilinear')
        refine1 = F.upsample(refine1, size=x_.size()[2:], mode='bilinear')

        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)
        final = (predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4

        return torch.sigmoid(final)

def MADSNet():
    model = MADSNet()
    return model