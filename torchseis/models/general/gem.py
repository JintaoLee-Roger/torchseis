import torch, warnings, math
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from ._gem_block import *
from ...ops.chunked.interpolate import interpolate3d_chunked



class Backbone(nn.Module, FuseOps):
    def __init__(self, 
                 in_channels=3,
                 base=64,
                 use_checkpoint=False,
                 width_factors=[1, 4, 8],
                 stage_num_blocks=[1, 8],
                 drop_path_rate=0.,
                 norm_layer=GroupNorm):
        """
        初始化 HRNet 模型，通过参数 norm_layer 可选择归一化方式。
        Args:
            in_channels (int): 输入通道数。默认 3
            base (int): 基础通道数。默认 80
            use_checkpoint (bool): 是否使用梯度检查点。默认 False
            width_factors (List[int], optional): 每个分支的通道宽度因子。默认 [1, 4, 8]
            drop_path_rate (float): Stochastic Depth 丢弃率。默认 0.2
        """
        super(Backbone, self).__init__()
        c = base
        self.base = base
        end_channels = base // 4 if base // 4 >= 16 else 16

        if len(width_factors) < 3:
            width_factors = width_factors + [width_factors[-1]] * (3 - len(width_factors))
        base_branch_channels = int(c * width_factors[0])
        second_branch_channels = int(c * width_factors[1])
        third_branch_channels = int(c * width_factors[2])

        self.input_layer = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2, Norm=norm_layer, use_checkpoint=use_checkpoint, use_scale=1),
            *[Bottleneck(32, 16, Norm=norm_layer, use_checkpoint=use_checkpoint, use_scale=1) for i in range(4)],
            CBAMBlock(inplanes=32, use_checkpoint=use_checkpoint)
        )

        self.stem_layer = nn.Sequential(
            nn.Identity(),
            ConvBlock(32, c * 4, stride=2, Norm=norm_layer, use_checkpoint=use_checkpoint, use_scale=1),
            *[Bottleneck(c * 4, c * 1, Norm=norm_layer, use_checkpoint=use_checkpoint, use_scale=1) for i in range(4)],
            CBAMBlock(inplanes=c * 4, use_checkpoint=use_checkpoint)
        )

        self.transition1 = nn.ModuleList([
            ConvBlock(c * 4, base_branch_channels, stride=1, Norm=norm_layer),
            ConvBlock(c * 4, second_branch_channels, stride=2, Norm=norm_layer),
        ])

        self.stage1 = nn.ModuleList([StageModule(stage=2, output_branches=2, c=base_branch_channels,
                                                 use_checkpoint=use_checkpoint,
                                                 width_factors=width_factors[:2],
                                                 drop_path_rate=0., Norm=norm_layer) for i in
                                     range(stage_num_blocks[0])])

        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            ConvBlock(second_branch_channels, third_branch_channels, stride=2, Norm=norm_layer),
        ])

        drop_paths = [x.item() for x in
                      torch.linspace(drop_path_rate * 0.5, drop_path_rate, stage_num_blocks[1])]

        self.stage2 = nn.ModuleList([StageModule(stage=3, output_branches=3, c=base_branch_channels,
                                                 use_checkpoint=use_checkpoint,
                                                 width_factors=width_factors[:3],
                                                 drop_path_rate=drop_paths[i], Norm=norm_layer, use_scale=1) for i in # TODO: 
                                     range(stage_num_blocks[1])])

        fuse_channels = base_branch_channels + second_branch_channels + third_branch_channels

        self.decoder1 = nn.Sequential(
            ConvBlock(fuse_channels, c * 4, Norm=norm_layer, use_checkpoint=use_checkpoint, use_scale=1024),
            *[Bottleneck(c * 4, c * 1, Norm=norm_layer, use_checkpoint=use_checkpoint) for i in range(4)],
            CBAMBlock(inplanes=c * 4, use_checkpoint=use_checkpoint),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            ConvBlock(c * 4, end_channels * 3, Norm=norm_layer, use_checkpoint=use_checkpoint),
        )

        self.decoder2 = nn.Sequential(
            ConvBlock(end_channels * 3 + 32, end_channels * 3, Norm=norm_layer, use_checkpoint=use_checkpoint, use_scale=1024),
            *[BasicBlock(end_channels * 3, end_channels * 3, Norm=norm_layer, use_checkpoint=use_checkpoint) for i in
              range(2)],
            CBAMBlock(inplanes=end_channels * 3, use_checkpoint=use_checkpoint),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            ConvBlock(end_channels * 3, end_channels, Norm=norm_layer, use_checkpoint=use_checkpoint),
            *[ConvBlock(end_channels, end_channels, Norm=norm_layer, padding_mode='reflect',
                        use_checkpoint=use_checkpoint) for i in range(4)],
            nn.Conv3d(end_channels, 1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect'),
        )

    def _input_layer(self, x: Tensor) -> Tensor:
        for i in range(len(self.input_layer)):
            x = self.input_layer[i].chunk_conv_forward(x)
        return x

    def _stem_layer(self, x: Tensor) -> Tensor:
        x = self.stem_layer[0](x)
        for i in range(1, len(self.stem_layer)):
            x = self.stem_layer[i].chunk_conv_forward(x)
        return x
    
    def _transition1(self, x):
        y1 = self.transition1[0].chunk_conv_forward(x)
        y2 = self.transition1[1].chunk_conv_forward(x)
        return [y1, y2]
    
    def _transition2(self, x):
        out = []
        for trans, i in zip(self.transition2, [0, 1, -1]):
            if isinstance(trans, ConvBlock):
                out.append(trans.chunk_conv_forward(x[i]))
            else:
                out.append(trans(x[i]))
        return out
    
    def _decoder1(self, x, skip_first=False):
        if not skip_first:
            x = self.decoder1[0].chunk_conv_forward(x)

        for i in range(1, len(self.decoder1)):
            if isinstance(self.decoder1[i], nn.Upsample):
                break
            else:
                x = self.decoder1[i].chunk_conv_forward(x)

        x = self._up_conv2(x, self.decoder1[-1])

        return x


    def _decoder2(self, x, skip_first=False, fuse=False):
        if not skip_first:
            x = self.decoder2[0].chunk_conv_forward(x)

        for i in range(1, len(self.decoder2)):
            if isinstance(self.decoder2[i], nn.Upsample):
                break
            else:
                x = self.decoder2[i].chunk_conv_forward(x)
                # print(f"decoder2-{i}: ", x.min().item(), x.max().item())

        if fuse:
            return self._decoder_last(x)
        k = i
        for i in range(k, len(self.decoder2)):
            if isinstance(self.decoder2[i], nn.Upsample):
                x = interpolate3d_chunked(x, 64, 2, 'trilinear')
            elif isinstance(self.decoder2[i], nn.Conv3d):
                x = Conv3dChunked(self.decoder2[i], 64)(x)
            else:
                x = self.decoder2[i].chunk_conv_forward(x)
                # print(f"decoder2-{i}: ", x.min().item(), x.max().item())

        return x


    def forward(self, x, prompt=None):
        # mask = (prompt != 0).to(prompt.dtype).to(prompt.device)
        # x = torch.cat([x, prompt, mask], dim=1)
        # x = torch.nn.ReflectionPad3d(32)(x)
        skip_x = self.input_layer(x) # 32, D/2
        x = self.stem_layer(skip_x) # 256, D/4

        x = [trans(x) for trans in self.transition1] # [(64, D/4), (32, D/8)]

        # stage 1
        for i, module in enumerate(self.stage1): x = module(x)

        # stage 2
        x = [trans(x[i]) for trans, i in zip(self.transition2, [0, 1, -1])] # [(64, D/4), (256, D/8), (512, D/16)]
        for i, module in enumerate(self.stage2):
            x = module(x)

        # decoder
        _, _, t, h, w = x[0].shape
        x = torch.cat([x[0], F.interpolate(x[1], size=(t, h, w), mode='trilinear'),
                              F.interpolate(x[2], size=(t, h, w), mode='trilinear')], dim=1) # 832, D/4
        # x_concat = torch.cat([x[0], interpolate3d_chunked(x[1], 32, 2, 'trilinear'), interpolate3d_chunked(x[2], 32, 4, 'trilinear')], dim=1)
        x = self.decoder1(x) # 48, D/2
        x = torch.cat([skip_x, x], dim=1)
        x = self.decoder2(x)
        # for i in range(len(self.decoder2)):
        #     x = self.decoder2[i](x)
        #     print(f"decoder2-{i}: ", x.min().item(), x.max().item())

        # x = x[:, :, 32:-32, 32:-32, 32:-32]
        x = torch.tanh(x)

        #x = tensor_z_score_clip(x) * 2 - 1
        return x

    def forward2(self, x, fuse=True, *args, **kwargs):
        # x = torch.nn.ReflectionPad3d(32)(x)
        skip_x = self._input_layer(x) # 32, D/2
        # print("skip_x: ", skip_x.min().item(), skip_x.max().item())
        x = self._stem_layer(skip_x) # 256, D/4
        # print("stem: ", x.min().item(), x.max().item())

        # stage 1
        x = self._transition1(x) # [(64, D/4), (32, D/8)]
        # print("trans1-x[0]: ", x[0].min().item(), x[0].max().item())
        # print("trans1-x[1]: ", x[1].min().item(), x[1].max().item())

        for i, module in enumerate(self.stage1): 
            module: StageModule
            x = module.chunk_conv_forward(x)
        # print("stage1-x[0]: ", x[0].min().item(), x[0].max().item())
        # print("stage1-x[1]: ", x[1].min().item(), x[1].max().item())

        # stage 2
        x = self._transition2(x) # [(64, D/4), (256, D/8), (512, D/16)]
        # print("trans1-x[0]: ", x[0].min().item(), x[0].max().item())
        # print("trans2-x[1]: ", x[1].min().item(), x[1].max().item())
        # print("trans2-x[2]: ", x[2].min().item(), x[2].max().item())

        for i, module in enumerate(self.stage2):
            module: StageModule
            x = module.chunk_conv_forward(x)
        # print("stage2-x[0]: ", x[0].min().item(), x[0].max().item())
        # print("stage2-x[1]: ", x[1].min().item(), x[1].max().item())
        # print("stage2-x[2]: ", x[2].min().item(), x[2].max().item())
        # decoder
        # _, _, t, h, w = x[0].shape
        # x = torch.cat([x[0], interpolate3d_chunked(x[1], 32, 2, 'trilinear'), interpolate3d_chunked(x[2], 32, 4, 'trilinear')], dim=1)
        x = self._concat_conv(x)
        # print("cat_conv: ", x.min().item(), x.max().item())
        x = self._decoder1(x, skip_first=True) # [b, 256, d/2, h/2, w/2]
        # print(f"decoder1: ", x.min().item(), x.max().item())
        # return x
        x = self._concat_conv2(skip_x, x)
        # print("decoder2-0: ", x.min().item(), x.max().item())
        # return x
        x = self._decoder2(x, skip_first=True, fuse=fuse)
        # print(f"decoder2: ", x.min().item(), x.max().item())

        # x = x[:, :, 32:-32, 32:-32, 32:-32]
        return torch.tanh(x)


if __name__ == '__main__':
    base = 32

    # 示例：使用默认的 Normalization 归一化
    print("\n=== 使用默认归一化 (InstanceNorm3d) 的 HRNet ===")
    model = Backbone(use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.InstanceNorm3d)
    model = model.cuda()
    with torch.no_grad():
        y = model(torch.ones(1, 3, 128, 128, 128).cuda())
    print(f"输出形状: {y.shape}")

    # 示例：使用 BatchNorm3d 作为归一化方式（注意：BatchNorm3d 接收的通道数参数和 InstanceNorm3d 参数类似）
    print("\n=== 使用 BatchNorm3d 的 HRNet ===")
    model_bn = Backbone(use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.BatchNorm3d)
    model_bn = model_bn.cuda()
    with torch.no_grad():
        y_bn = model_bn(torch.ones(1, 3, 128, 128, 128).cuda())
    print(f"输出形状: {y_bn.shape}")
