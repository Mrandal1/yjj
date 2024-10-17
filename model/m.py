import copy
import pathlib
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, base

detach_reset = False

backend = "torch"


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u, scale


def weight_quant_per_channel_dw(w):
    scale = 1.0 / torch.mean(w.abs().flatten(2, 3), dim=2, keepdim=True).clamp_(min=1e-5).unsqueeze(-1)
    u = (w * scale).round().clamp_(-1, 1)
    return u, scale.permute(1, 0, 2, 3)


class Ternarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, neg_gate):
        ctx.save_for_backward(x)
        ctx.gate = gate
        ctx.neg_gate = neg_gate
        out_s = torch.sign(x)
        out_s[(x < gate) & (x > neg_gate)] = torch.tensor(0.)
        return out_s

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.clone()
        x = ctx.saved_tensors[0]
        grad_x[(x < ctx.neg_gate - 0.5) |
               ((ctx.neg_gate + 0.5 < x) & (x < ctx.gate - 0.5)) |
               (x > ctx.gate + 0.5)] = 0.
        return grad_x, None, None


class HardTernaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=0, bias=False, group=1):
        super(HardTernaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=group
        )

        self.quan = weight_quant_per_channel_dw if group > 1 else weight_quant

    def forward(self, x):
        real_weights = self.weight
        w, scale = self.quan(real_weights)
        ternary_weights = real_weights + (w - real_weights).detach()

        # ternary convolution
        y = F.conv2d(x, ternary_weights, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        y = y / scale
        return y


class BiSeparableConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, first_bi=True):
        super(BiSeparableConv, self).__init__()
        bias = False
        if first_bi:
            self.conv = HardTernaryConv(in_chn, out_chn, kernel_size, 1, kernel_size // 2, bias=bias, group=in_chn)
        else:
            self.conv = nn.Conv2d(in_chn, out_chn, kernel_size, 1, kernel_size // 2, bias=bias, groups=in_chn)

        self.point_conv = HardTernaryConv(out_chn, out_chn, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.point_conv(x)
        return x


class ASTH(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan(), detach_reset: bool = False, dim=32,
                 kernel_size=3):

        assert isinstance(v_reset, float) or v_reset is None

        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', None)

        self.register_memory("spike", None)

        self.dim = dim

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('v_reset', v_reset)

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.s_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.Conv2d(dim, dim, 1, padding=0)
        )

        self.s_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.Conv2d(dim, dim, 1, padding=0)
        )
        self.x_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.Conv2d(dim, dim, 1, padding=0)
        )
        self.full_precise_func = surrogate.ATan(spiking=False)
        self.apply(self.init_weights)

    def init_weights(self, p):
        if isinstance(p, nn.Conv2d):
            nn.init.orthogonal_(p.weight)

    def reset(self):
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def forward_spike(self, x: torch.Tensor):
        beta = torch.sigmoid(self.s_1(self.spike) + self.x_1(x))
        self.v = x + beta * self.v + self.s_2(self.spike)
        self.spike = self.full_precise_func(self.v)
        return self.spike

    def forward(self, x):
        if self.spike is None:
            B, C, H, W = x.shape
            self.spike = torch.zeros((B, self.dim, H, W)).to(x.device)
        if self.v is None:
            B, C, H, W = x.shape
            self.v = torch.zeros((B, self.dim, H, W)).to(x.device)

        spike = self.forward_spike(x)
        return spike


class STH(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan(), detach_reset: bool = False, dim=32,
                 kernel_size=3, multi_spike=False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', None)

        self.register_memory("spike", None)  # spike

        self.multi_spike = multi_spike

        self.dim = dim

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('v_reset', v_reset)

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.s_1 = BiSeparableConv(dim, dim, kernel_size, first_bi=False)
        self.s_2 = BiSeparableConv(dim, dim, kernel_size, first_bi=False)
        self.x_1 = BiSeparableConv(dim, dim, kernel_size, first_bi=True)

        self.gates = nn.Parameter(torch.tensor([1.0, 0.5, 0.25]).squeeze(), requires_grad=False)
        self.neg_gates = nn.Parameter(torch.tensor([-2.75, -2.25, -2.]).squeeze(), requires_grad=False)

        self.g = 1.25
        self.g_n = -2.5

        self.apply(self.init_weights)

    def init_weights(self, p):
        if isinstance(p, nn.Conv2d):
            nn.init.orthogonal_(p.weight)
            if p.bias:
                nn.init.constant_(p.bias, 0.0)

    def reset(self):
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def forward_multi_spike(self, x):

        beta = torch.sigmoid(self.s_1(self.spike) + self.x_1(x))

        self.v = x + beta * self.v + self.s_2(self.spike)

        s_1 = Ternarizer.apply(self.v, self.gates[0], self.neg_gates[0])
        self.v = self.v - self.gates[0] * s_1

        s_2 = Ternarizer.apply(self.v, self.gates[1], self.neg_gates[1])
        self.v = self.v - self.gates[1] * s_2

        s_3 = Ternarizer.apply(self.v, self.gates[2], self.neg_gates[2])
        self.v = self.v - self.gates[2] * s_3

        self.spike = self.gates[0] * s_1 + self.gates[1] * s_2 + self.gates[2] * s_3

        return self.spike

    def forward_spike(self, x: torch.Tensor):

        update = torch.sigmoid(self.s_1(self.spike) + self.x_1(x))

        self.v = x + update * self.v + self.s_2(self.spike)

        self.spike = Ternarizer.apply(self.v, self.g, self.g_n)
        self.v = self.v - self.g * self.spike

        return self.spike

    def forward(self, x):
        if self.spike is None:
            B, C, H, W = x.shape
            self.spike = torch.zeros((B, self.dim, H, W)).to(x.device)
        if self.v is None:
            B, C, H, W = x.shape
            self.v = torch.zeros((B, self.dim, H, W)).to(x.device)

        if self.multi_spike:
            spike = self.forward_multi_spike(x)
        else:
            spike = self.forward_spike(x)
        return spike


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.out_c = out_channels

        self.activation = STH(dim=in_channels)

        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.activation(x)
        x = self.conv2d(x)
        x = self.norm_layer(x)

        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(DownSample, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, 1)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, stride, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = residual + x
        x = self.conv2(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(UpSampleLayer, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, factor=1., bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, int(channel * factor), 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channel * factor), channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class MBConv(nn.Module):
    def __init__(self, in_chn, factor=4):
        super(MBConv, self).__init__()
        mid_chn = in_chn * factor
        self.head = nn.Sequential(
            nn.Conv2d(in_chn, mid_chn, 1, bias=False),
            nn.BatchNorm2d(mid_chn),
            nn.ReLU(True)
        )
        self.mid = nn.Sequential(
            nn.Conv2d(mid_chn, mid_chn, kernel_size=5, padding=2, groups=mid_chn),
            nn.BatchNorm2d(mid_chn),
            nn.ReLU(True)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(mid_chn, in_chn, 1, bias=False),
            nn.BatchNorm2d(in_chn),
        )
        self.f = nn.Conv2d(in_chn, in_chn, 1)

        self.CA = CALayer(mid_chn, 1)

    def forward(self, x):
        res = self.head(x)
        res = self.mid(res)
        res = self.CA(res)
        res = self.tail(res)
        res += x

        return self.f(res)


class TFC(nn.Module):
    def __init__(self, n_feat=32, hidden_feat=32, o_feat=32, norm='ortho'):  # 'ortho'
        super(TFC, self).__init__()
        fft_dim = n_feat * 2
        hidden_dim = hidden_feat * 2
        o_dim = o_feat * 2

        self.tfc = nn.Sequential(
            nn.Conv2d(fft_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            ASTH(dim=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            MBConv(in_chn=hidden_dim),
            nn.Conv2d(hidden_dim, o_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(o_dim)
        )

        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        # x = x.to(torch.float32)
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real

        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.tfc(y_f)
        # y = y.to(torch.float32)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        return y


class PredictLayer(nn.Module):
    def __init__(self, in_channel=32):
        super(PredictLayer, self).__init__()
        self.f1 = nn.Sequential(
            STH(dim=in_channel, multi_spike=True),
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        o = self.f1(x)
        return o


class PredictLayer1(nn.Module):
    def __init__(self, in_channel=32):
        super(PredictLayer1, self).__init__()

        self.f1 = nn.Sequential(
            STH(dim=in_channel, multi_spike=True),
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.f2 = TFC(in_channel, in_channel, 1)

    def forward(self, x):
        o = self.f1(x) + self.f2(x)
        return o


class m1(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.overlap = 3
        self.in_conv = nn.Sequential(
            nn.Conv2d(self.overlap, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32)
        )

        kernel_size = 5
        d_kernel_size = 5
        padding = kernel_size // 2
        d_padding = d_kernel_size // 2
        self.down1 = DownSample(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, padding=padding)
        self.down2 = DownSample(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=2, padding=padding)
        self.down3 = DownSample(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=2, padding=padding)

        self.residualBlock = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=256, stride=1),
        )
        self.up1 = UpSampleLayer(in_channels=256, out_channels=128, kernel_size=d_kernel_size, stride=1,
                                 padding=d_padding, )
        self.up2 = UpSampleLayer(in_channels=128, out_channels=64, kernel_size=d_kernel_size, stride=1,
                                 padding=d_padding)
        self.up3 = UpSampleLayer(in_channels=64, out_channels=32, kernel_size=d_kernel_size, stride=1,
                                 padding=d_padding)
        self.out = PredictLayer(in_channel=32)

        self.reset_gates()

    def reset_gates(self):
        gates1 = [1.0, -2.0]
        gates2 = [1.5, -3.0]
        self.down1.conv.activation.g, self.down1.conv.activation.g_n = gates1[0], gates1[1]
        self.down2.conv.activation.g, self.down2.conv.activation.g_n = gates1[0], gates1[1]
        self.down3.conv.activation.g, self.down3.conv.activation.g_n = gates1[0], gates1[1]

        self.residualBlock[0].conv1.activation.g, self.residualBlock[0].conv1.activation.g_n = gates1[0], gates1[1]
        self.residualBlock[0].conv2.activation.g, self.residualBlock[0].conv2.activation.g_n = gates2[0], gates2[1]

        self.up1.conv.activation.g, self.up1.conv.activation.g_n = gates2[0], gates2[1]
        self.up2.conv.activation.g, self.up2.conv.activation.g_n = gates2[0], gates2[1]
        self.up3.conv.activation.g, self.up3.conv.activation.g_n = gates2[0], gates2[1]

    def reset_full_net(self):
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()

    def forward(self, on_img, pres):
        if pres is not None:
            pre_bins = pres['pre_bins']
        else:
            pre_bins = None

        if pre_bins is None:
            start = on_img[:, 0:1, ...]
            pre_bins = torch.cat([start for _ in range(self.overlap - 1)], dim=1)
        on_img = torch.cat([pre_bins, on_img], 1)
        pre_bins = on_img[:, 1 - self.overlap:, ...]

        on_img = torch.stack([on_img[:, i:i + self.overlap, ...] for i in range(5)], dim=0)  # t b 3 h w

        for t in range(5):
            x = self.in_conv(on_img[t, ...])
            x1 = self.down1(x)

            x2 = self.down2(x1)

            x3 = self.down3(x2)

            r1 = self.residualBlock(x3)

            u1 = self.up1(r1 + x3)
            u2 = self.up2(u1 + x2)
            u3 = self.up3(u2 + x1)
            o = u3 + x
            i = self.out(o)

        pres = {
            'pre_bins': pre_bins,
        }
        return {'image': i}, pres


class m2(nn.Module):
    def __init__(self):
        super().__init__()
        self.overlap = 3
        self.in_conv = nn.Sequential(
            nn.Conv2d(self.overlap, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32)
        )

        kernel_size = 5
        d_kernel_size = 5
        padding = kernel_size // 2
        d_padding = d_kernel_size // 2
        self.down1 = DownSample(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, padding=padding)
        self.down2 = DownSample(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=2, padding=padding)
        self.down3 = DownSample(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=2, padding=padding)

        self.residualBlock = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=256, stride=1),
        )
        self.up1 = UpSampleLayer(in_channels=256, out_channels=128, kernel_size=d_kernel_size, stride=1,
                                 padding=d_padding, )
        self.up2 = UpSampleLayer(in_channels=128, out_channels=64, kernel_size=d_kernel_size, stride=1,
                                 padding=d_padding)
        self.up3 = UpSampleLayer(in_channels=64, out_channels=32, kernel_size=d_kernel_size, stride=1,
                                 padding=d_padding)
        self.out = PredictLayer1(in_channel=32)

        self.head = TFC(32, 32, 32)
        self.reset_gates()

    def reset_gates(self):
        gates1 = [1.0, -2.0]
        gates2 = [1.5, -3.0]
        self.down1.conv.activation.g, self.down1.conv.activation.g_n = gates1[0], gates1[1]
        self.down2.conv.activation.g, self.down2.conv.activation.g_n = gates1[0], gates1[1]
        self.down3.conv.activation.g, self.down3.conv.activation.g_n = gates1[0], gates1[1]

        self.residualBlock[0].conv1.activation.g, self.residualBlock[0].conv1.activation.g_n = gates1[0], gates1[1]
        self.residualBlock[0].conv2.activation.g, self.residualBlock[0].conv2.activation.g_n = gates2[0], gates2[1]

        self.up1.conv.activation.g, self.up1.conv.activation.g_n = gates2[0], gates2[1]
        self.up2.conv.activation.g, self.up2.conv.activation.g_n = gates2[0], gates2[1]
        self.up3.conv.activation.g, self.up3.conv.activation.g_n = gates2[0], gates2[1]

    def reset_full_net(self):
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()

    def forward(self, on_img, pres):
        if pres is not None:
            pre_bins = pres['pre_bins']
        else:
            pre_bins = None

        if pre_bins is None:
            start = on_img[:, 0:1, ...]
            pre_bins = torch.cat([start for _ in range(self.overlap - 1)], dim=1)
        on_img = torch.cat([pre_bins, on_img], 1)
        pre_bins = on_img[:, 1 - self.overlap:, ...]
        on_img = torch.stack([on_img[:, i:i + self.overlap, ...] for i in range(5)], dim=0)

        for t in range(5):
            x = self.in_conv(on_img[t, ...])
            x = x + self.head(x)
            x1 = self.down1(x)

            x2 = self.down2(x1)

            x3 = self.down3(x2)

            r1 = self.residualBlock(x3)

            u1 = self.up1(r1 + x3)
            u2 = self.up2(u1 + x2)
            u3 = self.up3(u2 + x1)
            o = u3 + x
            i = self.out(o)
        pres = {
            'pre_bins': pre_bins,
        }
        return {'image': i}, pres


class M(nn.Module):

    def __init__(self, tfc=False):
        super().__init__()
        self.num_encoders = 3
        self.snn = m2() if tfc else m1()

    def forward(self, event_tensor, pre_bins):
        output_dict, pre_bins = self.snn.forward(event_tensor, pre_bins)
        return output_dict, pre_bins


if __name__ == '__main__':
    input = torch.randint(high=500, size=(1, 5, 112, 112), dtype=torch.float32)

    model = m2()

    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    print('Test Good!')

