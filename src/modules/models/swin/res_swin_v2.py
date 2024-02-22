import torch
import torchvision
from torch import nn
from torch.utils import checkpoint

from src.modules.models.swin.stage_module import StageModule


class Conv_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, alpha=0.2):
        super(Conv_3, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv3(x)


class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, alpha=0.2):
        super(DConv, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv3(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, alpha=0.2):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class Channel_wise(nn.Module):
    def __init__(self, in_channels, out_channels, sizes):
        super().__init__()
        self.avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.LayerNorm(sizes),
        )

    def forward(self, x):
        return self.avg(x)


class DConv_3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = DConv(channels, channels, 3, 1, 2, dilation=2)
        self.layer2 = DConv(channels, channels, 3, 1, 4, dilation=4)
        self.layer3 = DConv(channels, channels, 3, 1, 8, dilation=8)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        e3 = self.layer3(e2)
        e3 = e3 + e1
        return e3


class DConv_2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = DConv(channels, channels, 3, 1, 2, dilation=2)
        self.layer2 = DConv(channels, channels, 3, 1, 4, dilation=4)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        return e2


class DConv_5(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = DConv(channels, channels, 3, 1, 2, dilation=2)
        self.layer2 = DConv(channels, channels, 3, 1, 4, dilation=4)
        self.layer3 = DConv(channels, channels, 3, 1, 8, dilation=8)
        self.layer4 = DConv(channels, channels, 3, 1, 4, dilation=4)
        self.layer5 = DConv(channels, channels, 3, 1, 2, dilation=2)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        e3 = self.layer3(e2)
        e3 = e3 + e1
        e4 = self.layer4(e3)
        e4 = e4 + e2
        e5 = self.layer5(e4)
        e5 = e5 + e3
        return e5


# Mix Block with attention mechanism
class MixBlock(nn.Module):
    def __init__(self, c_in):
        super(MixBlock, self).__init__()
        self.local_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.global_query = nn.Conv2d(c_in, c_in, (1, 1))

        self.local_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.global_key = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.global_gamma = nn.Parameter(torch.zeros(1))
        self.local_gamma = nn.Parameter(torch.zeros(1))

        self.local_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.local_bn = nn.BatchNorm2d(c_in)
        self.global_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.global_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_local, x_global):
        B, C, W, H = x_local.size()
        assert W == H

        q_local = self.local_query(x_local).reshape(-1, W, H)  # [BC, W, H]
        q_global = self.global_query(x_global).reshape(-1, W, H)
        M_query = torch.cat([q_local, q_global], dim=2)  # [BC, W, 2H]

        k_local = self.local_key(x_local).reshape(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_global = self.global_key(x_global).reshape(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_local, k_global], dim=1)  # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  # [BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_global = x_global * attention * (torch.sigmoid(self.global_gamma) * 2.0 - 1.0)
        y_local = x_local + self.local_bn(self.local_conv(att_global))

        att_local = x_local * attention * (torch.sigmoid(self.local_gamma) * 2.0 - 1.0)
        y_global = x_global + self.global_bn(self.global_conv(att_local))
        return y_local, y_global


class Res_Swin(nn.Module):
    def __init__(
        self,
        img_size=256,
        hidden_dim=64,
        layers=(2, 2, 18, 2),
        heads=(3, 6, 12, 24),
        channels=98,
        head_dim=32,
        window_size=8,
        downscaling_factors=(2, 2, 2, 2),
        relative_pos_embedding=True,
        use_checkpoint=False,
    ):
        self.checkpoint = use_checkpoint

        super(Res_Swin, self).__init__()

        self.base_model = torchvision.models.resnet34(True)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(
            Conv_3(channels, hidden_dim, 7, 2, 3),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
        )

        self.stage1 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        self.avg1 = Channel_wise(hidden_dim, hidden_dim, [hidden_dim, img_size // 4, img_size // 4])

        self.layer1 = DConv_3(hidden_dim)

        self.res_layer1 = nn.Sequential(*self.base_layers[3:5])

        self.mix1 = MixBlock(hidden_dim)

        self.conv1 = Conv_3(hidden_dim * 2, hidden_dim, 3, 1, 1)

        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        self.avg2 = Channel_wise(hidden_dim, hidden_dim * 2, [hidden_dim * 2, img_size // 8, img_size // 8])

        self.layer2 = DConv_3(hidden_dim * 2)

        self.res_layer2 = self.base_layers[5]

        self.mix2 = MixBlock(hidden_dim * 2)

        self.conv2 = Conv_3(hidden_dim * 4, hidden_dim * 2, 3, 1, 1)

        self.avg3 = Channel_wise(hidden_dim * 2, hidden_dim * 4, [hidden_dim * 4, img_size // 16, img_size // 16])

        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        self.layer3 = DConv_5(hidden_dim * 4)

        self.res_layer3 = self.base_layers[6]

        self.mix3 = MixBlock(hidden_dim * 4)

        self.conv3 = Conv_3(hidden_dim * 8, hidden_dim * 4, 3, 1, 1)

        self.avg4 = Channel_wise(hidden_dim * 4, hidden_dim * 8, [hidden_dim * 8, img_size // 32, img_size // 32])

        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        self.layer4 = DConv_2(hidden_dim * 8)

        self.res_layer4 = self.base_layers[7]

        self.mix4 = MixBlock(hidden_dim * 8)

        self.conv4 = Conv_3(hidden_dim * 16, hidden_dim * 8, 3, 1, 1)

        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 128 + 128, 128)
        self.decode2 = Decoder(128, 64 + 64, 64)
        self.decode1 = Decoder(64, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
        )
        self.conv_last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e0 = self.layer0(x)

        e1_res = self.res_layer1(e0)
        if self.checkpoint:
            e1_swin_tmp = checkpoint.checkpoint(self.custom(self.stage1), e0) + self.avg1(e0)
        else:
            e1_swin_tmp = self.stage1(e0) + self.avg1(e0)
        e1_swin = self.layer1(e1_swin_tmp) + e1_swin_tmp
        e1_res, e1_swin = self.mix1(e1_res, e1_swin)
        e1 = torch.cat((e1_res, e1_swin), dim=1)
        e1 = self.conv1(e1)

        e2_res = self.res_layer2(e1_res)
        if self.checkpoint:
            e2_swin_tmp = checkpoint.checkpoint(self.custom(self.stage2), e1_swin) + self.avg2(e1_swin)
        else:
            e2_swin_tmp = self.stage2(e1_swin) + self.avg2(e1_swin)
        e2_swin = self.layer2(e2_swin_tmp) + e2_swin_tmp
        e2_res, e2_swin = self.mix2(e2_res, e2_swin)
        e2 = torch.cat((e2_res, e2_swin), dim=1)
        e2 = self.conv2(e2)

        e3_res = self.res_layer3(e2_res)
        if self.checkpoint:
            e3_swin_tmp = checkpoint.checkpoint(self.custom(self.stage3), e2_swin) + self.avg3(e2_swin)
        else:
            e3_swin_tmp = self.stage3(e2_swin) + self.avg3(e2_swin)
        e3_swin = self.layer3(e3_swin_tmp) + e3_swin_tmp
        e3_res, e3_swin = self.mix3(e3_res, e3_swin)
        e3 = torch.cat((e3_res, e3_swin), dim=1)
        e3 = self.conv3(e3)

        e4_res = self.res_layer4(e3_res)
        if self.checkpoint:
            e4_swin_tmp = checkpoint.checkpoint(self.custom(self.stage4), e3_swin) + self.avg4(e3_swin)
        else:
            e4_swin_tmp = self.stage4(e3_swin) + self.avg4(e3_swin)
        e4_swin = self.layer4(e4_swin_tmp) + e4_swin_tmp
        e4_res, e4_swin = self.mix4(e4_res, e4_swin)
        e4 = torch.cat((e4_res, e4_swin), dim=1)
        e4 = self.conv4(e4)

        d4 = self.decode4(e4, e3)  # 256,16,16
        d3 = self.decode3(d4, e2)  # 256,32,32
        d2 = self.decode2(d3, e1)  # 128,64,64
        d1 = self.decode1(d2, e0)  # 64,128,128
        d0 = self.decode0(d1)  # 64,256,256
        out = self.conv_last(d0)  # 1,256,256
        return torch.sigmoid(out)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs

        return custom_forward
