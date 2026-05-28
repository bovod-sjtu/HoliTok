from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import pow, sin
from torch.nn import Parameter
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def high_order_resample_torch(x: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    return AF.resample(
        x,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=128,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
    )


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False, force_drop: bool = False, **kwargs):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.dropout(
            x,
            p=self.p,
            training=True if self.force_drop else self.training,
            inplace=self.inplace,
        )


class Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        w_init_gain: str | float | None = "linear",
        activation=None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = out_channels
        if w_init_gain is not None:
            gain = nn.init.calculate_gain(w_init_gain) if isinstance(w_init_gain, str) else w_init_gain
            nn.init.xavier_uniform_(self.weight, gain=gain)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.activation(super().forward(x))


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        bn: bool = False,
        activation=None,
        w_init_gain=None,
        input_transpose: bool = False,
        **kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super().forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, "kernel_size must be 2*stride in causal ConvTranspose1d."

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x.transpose(1, 2) if self.transpose else x


def sinc(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "sinc"):
        return torch.sinc(x)
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    a = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if a > 50.0:
        beta = 0.1102 * (a - 8.7)
    elif a >= 21.0:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
        channels: int = 1,
        causal: bool = True,
        fixed_filter: bool = False,
    ):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        if causal:
            self.pad_left = kernel_size - 1
            self.pad_right = 0
        else:
            even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.fixed_filter = fixed_filter
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        if fixed_filter:
            self.register_buffer("filter", filt)
        else:
            self.filter = nn.Parameter(filt.expand(channels, -1, -1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        if self.fixed_filter:
            return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return F.conv1d(x, self.filter, stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.causal = causal
        self.fixed_filter = fixed_filter
        if causal:
            self.pad = 0
        else:
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filt = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        if fixed_filter:
            self.register_buffer("filter", filt)
        else:
            self.filter = nn.Parameter(filt.expand(channels, -1, -1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        if self.fixed_filter:
            x = self.ratio * F.conv_transpose1d(
                x,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            )
        else:
            x = self.ratio * F.conv_transpose1d(x, self.filter, stride=self.stride, groups=channels)
        if self.causal:
            x = x[..., : -(self.kernel_size - self.stride)]
        else:
            x = x[..., self.pad_left : -self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            channels=channels,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal: bool = True,
        fixed_filter: bool = False,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(
            up_ratio,
            up_kernel_size,
            activation.in_features,
            causal=causal,
            fixed_filter=fixed_filter,
        )
        self.downsample = DownSample1d(
            down_ratio,
            down_kernel_size,
            activation.in_features,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class Conv1d_S(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        norm_type="weight_norm",
        init_type=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        pad = 0 if causal else dilation * (kernel_size - 1) // 2
        self.causal_pad = dilation * (kernel_size - 1) if causal else 0
        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
        )
        if init_type == "orthogonal":
            nn.init.orthogonal_(self.layer.weight)
        elif init_type == "normal":
            nn.init.normal_(self.layer.weight, mean=0.0, std=0.01)
        if norm_type == "weight_norm":
            self.layer = weight_norm(self.layer)
        elif norm_type == "spectral_norm":
            self.layer = spectral_norm(self.layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.causal and self.causal_pad > 0:
            inputs = F.pad(inputs, (self.causal_pad, 0))
        return self.layer(inputs)


class SLSTM(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.skip = skip
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=dimension,
            hidden_size=dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if self.bidirectional:
            self.proj_out = nn.Linear(dimension * 2, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        if self.bidirectional:
            y = self.proj_out(y)
        if self.skip:
            y = y + x
        return y


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4, causal=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(nums):
            dil = base**i
            pad1 = dil * (kernel_size - 1) if causal else dil
            pad2 = (kernel_size - 1) if causal else 1
            block: list[nn.Module] = [nn.LeakyReLU()]
            if causal and pad1 > 0:
                block.append(nn.ConstantPad1d((pad1, 0), 0.0))
            block.append(
                weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=dil,
                        padding=0 if causal else pad1,
                    )
                )
            )
            block.append(nn.LeakyReLU())
            if causal and pad2 > 0:
                block.append(nn.ConstantPad1d((pad2, 0), 0.0))
            block.append(
                weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=0 if causal else pad2,
                    )
                )
            )
            self.layers.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=100,
        base_channels=12,
        proj_kernel_size=3,
        stack_kernel_size=3,
        stack_dilation_base=2,
        stacks=6,
        channels=(12, 24, 48, 96, 192, 384, 768),
        down_sample_factors=(2, 2, 2, 2, 4, 4),
        causal=False,
        lookahead=0,
    ):
        super().__init__()
        act_slope = 0.2
        layers: list[nn.Module] = [
            Conv1d_S(in_channels, base_channels, kernel_size=proj_kernel_size, stride=1, causal=causal),
            nn.LeakyReLU(act_slope, True),
        ]
        for (in_c, out_c), down_f in zip(zip(channels[:-1], channels[1:]), down_sample_factors):
            layers += [
                Conv1d_S(in_c, out_c, kernel_size=down_f * 2, stride=down_f, causal=causal),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks, causal=causal),
                nn.LeakyReLU(act_slope, True),
            ]
        if lookahead > 0:
            layers.append(Conv1d_S(channels[-1], out_channels, kernel_size=lookahead * 2 + 1, stride=1))
        else:
            layers.append(Conv1d_S(channels[-1], out_channels, kernel_size=proj_kernel_size, stride=1, causal=causal))
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions: torch.Tensor, z_inputs=None) -> torch.Tensor:
        return self.generator(conditions)


class AMPBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None, causal=True):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, causal=causal))
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)) for _ in dilation]
        )
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = _make_activations(h, channels, activation, self.num_layers, causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = c1(a1(x))
            xt = c2(a2(xt))
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None, causal=True):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, causal=causal))
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)
        self.num_layers = len(self.convs)
        self.activations = _make_activations(h, channels, activation, self.num_layers, causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act in zip(self.convs, self.activations):
            x = conv(act(x)) + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


def _make_periodic_activation(name: str, channels: int, alpha_logscale: bool):
    if name == "snake":
        return Snake(channels, alpha_logscale=alpha_logscale)
    if name == "snakebeta":
        return SnakeBeta(channels, alpha_logscale=alpha_logscale)
    raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")


def _make_activations(h, channels: int, activation: str, count: int, causal: bool) -> nn.ModuleList:
    anti_alias_causal = h.get("anti_alias_causal", causal)
    fixed_filter = h.get("fixed_filter", False)
    return nn.ModuleList(
        [
            Activation1d(
                activation=_make_periodic_activation(activation, channels, h.snake_logscale),
                causal=anti_alias_causal,
                fixed_filter=fixed_filter,
            )
            for _ in range(count)
        ]
    )


class Decoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        causal = h.causal
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        num_decoder_lookahead = h.get("num_decoder_lookahead", 3)
        self.conv_pre = weight_norm(
            Conv1d(
                h.latent_dim,
                h.upsample_initial_channel,
                kernel_size=2 * num_decoder_lookahead + 1,
                stride=1,
                causal=False,
            )
        )
        resblock = AMPBlock1 if h.resblock == "1" else AMPBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation, causal=causal))

        activation_post = _make_periodic_activation(h.activation, ch, h.snake_logscale)
        self.activation_post = Activation1d(
            activation=activation_post,
            causal=h.get("anti_alias_causal", causal),
            fixed_filter=h.get("fixed_filter1", False),
        )
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, causal=causal, bias=h.get("use_bias_at_final", True))
        )

        for up in self.ups:
            up.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(z)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.h.get("use_tanh_at_final", True):
            return torch.tanh(x)
        return torch.clamp(x, min=-1.0, max=1.0)

    def remove_weight_norm(self) -> None:
        for up in self.ups:
            for layer in up:
                remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
