
import torch, warnings, math
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from ...ops.chunked.conv import Conv3dChunked
from ...ops.chunked.chunk_base import get_index
from ...ops.chunked.interpolate import UpsampleChunked
from torchseis.ops.chunked.norms import AdaptiveInstanceNorm3d, AdaptiveGroupNorm
import itertools



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于任意维度张量，不仅仅是2D ConvNet
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def tensor_normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-6)


def tensor_z_score_clip(data, clp_s=3.2):
    z = (data - torch.mean(data)) / torch.std(data)
    return tensor_normalization(torch.clip(z, min=-clp_s, max=clp_s))


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, Norm=nn.InstanceNorm3d, padding_mode='zeros',
                 use_checkpoint=False, use_scale=1):
        """
        :param Norm: 归一化层构造函数，其调用形式为 Norm(num_channels)
        """
        super(ConvBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.use_scale = use_scale
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False, padding_mode=padding_mode),
            Norm(out_dim),
            nn.SiLU(inplace=True),
        )

    def _forward_impl(self, x):
        if self.use_scale == 1:
            return self.conv(x)
        else:
            # eps = self.conv[1].eps
            # self.conv[1].eps = eps / self.use_scale**2
            # x = self.conv(x)
            # self.conv[1].eps = eps  # 恢复原来的 eps
            return self.conv(x/self.use_scale)

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
        
    def chunk_conv_forward(self, x, bsize=64):
        x = Conv3dChunked(self.conv[0], bsize)(x/self.use_scale)
        # x = instance_norm(x, None, None, self.conv[1].eps, self.conv[1].affine, self.conv[1].weight, self.conv[1].bias, True, True)
        # return self.conv[2](x)
        return self.conv[1:](x)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes,
                 stride=1, use_checkpoint=False, drop_path=0., Norm=nn.InstanceNorm3d, padding_mode='zeros', skip=True, use_scale=1):
        super(BasicBlock, self).__init__()
        self.skip = skip
        self.conv1 = ConvBlock(inplanes, planes, stride=stride, Norm=Norm, padding_mode=padding_mode, use_scale=use_scale)
        self.conv2 = ConvBlock(planes, planes, stride=1, Norm=Norm, padding_mode=padding_mode, use_scale=use_scale)
        self.res_conv = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                                  bias=False) if inplanes != planes or stride != 1 else nn.Identity()
        self.stride = stride
        self.use_checkpoint = use_checkpoint
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward_impl(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.skip:
            return self.drop_path(x) + self.res_conv(residual)
        else:
            return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
        
    def chunk_conv_forward(self, x, bsize=64):
        residual = x
        x = self.conv1.chunk_conv_forward(x, bsize)
        x = self.conv2.chunk_conv_forward(x, bsize)
        if self.skip:
            if isinstance(self.res_conv, nn.Identity):
                return self.drop_path(x) + self.res_conv(residual)
            else:
                return self.drop_path(x) + Conv3dChunked(self.res_conv, bsize)(residual)
        else:
            return x


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, Norm=nn.InstanceNorm3d, use_checkpoint=False, use_scale=1):
        super(Bottleneck, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = ConvBlock(inplanes, planes, kernel_size=1, Norm=Norm, use_scale=use_scale)
        self.conv2 = ConvBlock(planes, planes, kernel_size=3, Norm=Norm, use_scale=use_scale)
        self.conv3 = ConvBlock(planes, inplanes, kernel_size=1, Norm=Norm, use_scale=use_scale)

    def _forward_impl(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = x + residual
        return out

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def chunk_conv_forward(self, x, bsize=64):
        residual = x
        x = self.conv1.chunk_conv_forward(x, bsize=bsize)
        x = self.conv2.chunk_conv_forward(x, bsize=bsize)
        x = self.conv3.chunk_conv_forward(x, bsize=bsize)
        return x + residual


class CBAMBlock(nn.Module):
    def __init__(self, inplanes, rate=8, kernel_size=5, Norm=partial(nn.GroupNorm, 1), use_checkpoint=False, use_scale=1):
        super(CBAMBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.SEBlock = nn.Sequential(
            ConvBlock(inplanes, inplanes // rate, kernel_size=3, Norm=Norm, padding_mode='reflect', use_scale=use_scale),
            ConvBlock(inplanes // rate, inplanes // rate,
                      kernel_size=kernel_size, Norm=Norm, padding_mode='reflect', use_scale=use_scale),
            nn.Conv3d(inplanes // rate, inplanes, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid())

    def _forward_impl(self, x):
        se_att = self.SEBlock(x)
        return se_att * x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
        
    def chunk_conv_forward(self, x, bsize=64):
        se_att = self.SEBlock[0].chunk_conv_forward(x, bsize)
        se_att = self.SEBlock[1].chunk_conv_forward(se_att, bsize)
        se_att = Conv3dChunked(self.SEBlock[2], bsize)(se_att)
        se_att = self.SEBlock[3](se_att)
        return se_att * x


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, use_checkpoint=False, width_factors=None, lossless_fuse=False,
                 drop_path_rate=0.0, Norm=nn.InstanceNorm3d, CBAM=True, use_scale=1):
        """
        初始化多分支的 Stage 模块，其中每个分支可以使用不同的宽度缩放因子，同时在 fuse 时选择不同的卷积和上采样方式。

        Args:
            stage (int): 该 stage 的分支数量
            output_branches (int): 输出的分支数量
            c (int): 基础通道数（第一个分支）
            use_checkpoint (bool): 是否使用梯度检查点（节省内存）
            width_factors (List[int], optional): 每个分支的宽度因子。若为 None，则除第一分支外，其余分支因子均为 4
            lossless_fuse (bool): 是否使用无损融合策略
            drop_path_rate (float): Stochastic Depth 的丢弃概率
            Norm: 归一化层构造函数，其调用方式为 Norm(num_channels)
        """
        super(StageModule, self).__init__()
        self.use_scale = use_scale

        if width_factors is None:
            width_factors = [1] + [4] * (stage - 1)
        elif len(width_factors) < stage:
            width_factors = width_factors + [width_factors[-1]] * (stage - len(width_factors))
        self.base_channels = c
        self.branch_channels = [int(c * factor) for factor in width_factors]

        # 为每个分支中的块计算 drop path 概率（线性从 0 到 drop_path_rate）
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 4 * stage)]

        self.branches = nn.ModuleList()
        for i in range(stage):
            branch_blocks = []
            for j in range(4):  # 每个分支使用4个 BasicBlock
                dp_index = i * 4 + j
                branch_blocks.append(
                    BasicBlock(
                        self.branch_channels[i],
                        self.branch_channels[i],
                        use_checkpoint=use_checkpoint,
                        drop_path=dp_rates[dp_index],
                        Norm=Norm,
                        use_scale=use_scale,
                    )
                )
            if CBAM: branch_blocks.append(CBAMBlock(inplanes=self.branch_channels[i]))
            self.branches.append(nn.Sequential(*branch_blocks))

        self.fuse_layers = nn.ModuleList()
        for i in range(output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(stage):
                if i == j:
                    # 空操作，直接相加
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    if not lossless_fuse:
                        self.fuse_layers[-1].append(nn.Sequential(
                            nn.Conv3d(self.branch_channels[j], self.branch_channels[i], stride=1, kernel_size=1,
                                      bias=False),
                            Norm(self.branch_channels[i]),
                            nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                        ))
                    else:
                        self.fuse_layers[-1].append(nn.Sequential(
                            nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                            nn.Conv3d(self.branch_channels[j], self.branch_channels[i], stride=1, kernel_size=3,
                                      padding=1, bias=False),
                            Norm(self.branch_channels[i]),
                        ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(self.branch_channels[j],
                                      self.branch_channels[j],
                                      kernel_size=3, stride=2,
                                      padding=1,
                                      bias=False),
                            Norm(self.branch_channels[j]),
                            nn.SiLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(self.branch_channels[j],
                                  self.branch_channels[i],
                                  kernel_size=3, stride=2,
                                  padding=1,
                                  bias=False),
                        Norm(self.branch_channels[i]),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.SiLU = nn.SiLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            # 融合每个分支的特征
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
        for i in range(len(x_fused)):
            x_fused[i] = self.SiLU(x_fused[i])
        return x_fused
    
    def _branch_chunk(self, branch, x):
        for i in range(len(branch)):
            x = branch[i].chunk_conv_forward(x)
        return x

    def _sq_chunk(self, sq, x):
        assert isinstance(sq, nn.Sequential)
        for i in range(len(sq)):
            if isinstance(sq[i], nn.Conv3d):
                if sq[i].stride != (1, 1, 1):
                    scale = 1
                else:
                    scale = self.use_scale
                x = Conv3dChunked(sq[i], 64)(x/scale) # TODO
            elif isinstance(sq[i], nn.Upsample):
                x = UpsampleChunked(sq[i], 32)(x)
            # elif isinstance(sq[i], nn.Sequential):
            #     x = self._sq_chunk(sq[i], x)
            else:
                x = sq[i](x)

        return x

    def chunk_conv_forward(self, x):
        assert len(self.branches) == len(x)

        x = [self._branch_chunk(branch, b) for branch, b in zip(self.branches, x)]
        # print(x[0].max().item())

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self._sq_chunk(self.fuse_layers[i][0], x[0]))
                else:
                    x_fused[i] = x_fused[i] + self._sq_chunk(self.fuse_layers[i][j], x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.SiLU(x_fused[i])
        # print(x_fused[0].max().item())
        return x_fused


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups_enumeration=[32, 24, 16]):
        super(GroupNorm, self).__init__()
        found = False
        for num_groups in num_groups_enumeration:
            if num_channels % num_groups == 0:
                found = True
                break
        if not found:
            raise ValueError(
                f"No valid num_groups found in {num_groups_enumeration} for num_channels={num_channels}"
            )
        # print(num_groups)
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)
    



class FuseOps:
    def _concat_conv(self, x):
        """
        x = concat(x[0], up(x[1]), up(x[2]))
        x = self.decoder1[0](x)
        """

        b, c, d, h, w = x[2].shape
        pad = 1
        scale = 4
        oshape = (b, self.base*4, *x[0].shape[2:])
        out = torch.zeros(oshape, device=x[0].device, dtype=x[0].dtype)

        bsize = 32
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
            # indices = (slice(None), slice(None), slice(int(r_d0/scale), int(r_d1/scale)), slice(int(r_h0/scale), int(r_h1/scale)), slice(int(r_w0/scale), int(r_w1/scale)))
            # padlist2 = [f*2 for f in padlist]
            padlist3 = [f*4 for f in padlist]
            with torch.no_grad():
                x1 = x[0][:, :, p_d0*4:p_d1*4, p_h0*4:p_h1*4, p_w0*4:p_w1*4]
                x2 = x[1][:, :, p_d0*2:p_d1*2, p_h0*2:p_h1*2, p_w0*2:p_w1*2]
                x3 = x[2][:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                if x1.numel() == 0:
                    continue
                x1 = torch.cat([x1, F.interpolate(x2, scale_factor=2, mode='trilinear'), F.interpolate(x3, scale_factor=4, mode='trilinear')], dim=1)
                x1 = F.pad(x1, padlist3)
                s = self.decoder1[0].use_scale
                x1 = self.decoder1[0].conv[0](x1/s)
                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = x1[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        out = self.decoder1[0].conv[1:](out)
        return out
    
    def _concat_conv2(self, skip_x, x):
        """
        x = concat(x[0], up(x[1]), up(x[2]))
        x = self.decoder1[0](x)
        """

        b, c, d, h, w = x.shape
        pad = 1
        scale = 1
        oshape = (b, self.decoder2[0].conv[0].out_channels, *x.shape[2:])
        out = torch.zeros(oshape, device=x[0].device, dtype=x[0].dtype)

        bsize = 32
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
            # indices = (slice(None), slice(None), slice(int(r_d0/scale), int(r_d1/scale)), slice(int(r_h0/scale), int(r_h1/scale)), slice(int(r_w0/scale), int(r_w1/scale)))
            # padlist2 = [f*2 for f in padlist]
            # padlist3 = [f*4 for f in padlist]
            with torch.no_grad():
                x1 = skip_x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                x2 = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                if x1.numel() == 0:
                    continue
                x1 = torch.cat([x1, x2], dim=1)
                x1 = F.pad(x1, padlist)
                s = self.decoder2[0].use_scale
                x1 = self.decoder2[0].conv[0](x1/s)
                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = x1[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        out = self.decoder2[0].conv[1:](out)
        return out
    
    def _up_conv2(self, x, conv: ConvBlock):
        """
        x = concat(x[0], up(x[1]), up(x[2]))
        x = self.decoder1[0](x)
        """

        b, c, d, h, w = x.shape
        pad = 1
        scale = 2
        oshape = (b, conv.conv[0].out_channels, d*2, h*2, w*2)
        out = torch.zeros(oshape, device=x[0].device, dtype=x[0].dtype)

        bsize = 32
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
            # indices = (slice(None), slice(None), slice(int(r_d0/scale), int(r_d1/scale)), slice(int(r_h0/scale), int(r_h1/scale)), slice(int(r_w0/scale), int(r_w1/scale)))
            padlist2 = [f*2 for f in padlist]
            # padlist3 = [f*4 for f in padlist]
            with torch.no_grad():
                x1 = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                x1 = F.interpolate(x1, scale_factor=2, mode='trilinear')
                x1 = F.pad(x1, padlist2)
                x1 = conv.conv[0](x1)
                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = x1[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        out = conv.conv[1:](out)
        return out
    
    def _decoder_last(self, x):
        b, c, d, h, w = x.shape
        pad = 4
        scale = 2

        oshape = (b, 1, d*2, h*2, w*2)
        out = torch.zeros(oshape, device=x[0].device, dtype=x[0].dtype)
        self._replace_decoder_norm()
        nT = 6
        if isinstance(self.decoder2[5].conv[1], nn.BatchNorm3d):
            nT = 1

        bsize = 32
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        for t in range(nT):
            for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
                v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
                indices = (slice(None), slice(None), slice(int(r_d0), int(r_d1)), slice(int(r_h0), int(r_h1)), slice(int(r_w0), int(r_w1)))
                padlist2 = [f*2 for f in padlist]
                # padlist3 = [f*4 for f in padlist]
                with torch.no_grad():
                    patch = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    patch = F.interpolate(patch, scale_factor=2, mode='trilinear')
                    patch = F.pad(patch, padlist2)
                    if nT == 1:
                        patch = self.decoder2[5:-1](patch)
                    else:
                        for g in range(5, 10):
                            patch = self.decoder2[g].conv[0](patch)
                            if t == g-5:
                                self.decoder2[g].conv[1].stats.add_batch(patch[indices])
                            elif not self.decoder2[g].conv[1].stats.has_statistics():
                                self.decoder2[g].conv[1].stats.compute()
                            if t < g-4:
                                break
                            patch = self.decoder2[g].conv[1:](patch)

                        if t < nT-1:
                            continue
                        patch = self.decoder2[-1](patch)


                    out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patch[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out
    
    def _replace_decoder_norm(self):
        for i in range(5, 10):
            if isinstance(self.decoder2[i].conv[1], nn.InstanceNorm3d):
                self.decoder2[i].conv[1] = AdaptiveInstanceNorm3d.from_instance_norm(self.decoder2[i].conv[1])
            elif isinstance(self.decoder2[i].conv[1], AdaptiveInstanceNorm3d):
                self.decoder2[i].conv[1].stats.reset()
            elif isinstance(self.decoder2[i].conv[1], GroupNorm):
                self.decoder2[i].conv[1] = AdaptiveGroupNorm.from_group_norm(self.decoder2[i].conv[1].norm)
            elif isinstance(self.decoder2[i].conv[1], AdaptiveGroupNorm):
                self.decoder2[i].conv[1].stats.reset()