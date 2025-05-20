# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import torch
from torch import Tensor, nn
from torch.ao.quantization import fuse_modules
from copy import deepcopy

class ConvBatchNorm3d(nn.Conv3d):

    def __init__(self, conv: nn.Conv3d, bn: nn.BatchNorm3d):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation, conv.groups,
                         conv.bias is not None, conv.padding_mode)

        assert not bn.training, "BatchNorm must be in eval mode before fusion."
        device = conv.weight.device
        dtype = conv.weight.dtype
        self.to(device, dtype)

        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.bias.data.copy_(conv.bias.data)
        else:
            self.bias = nn.Parameter(torch.zeros_like(bn.running_mean, device=device, dtype=dtype))

        # HACK: need deepcopy?
        self.orig_conv = conv
        self.orig_bn = bn

        w_conv = self.weight.reshape(self.out_channels, -1)

        if bn.affine:
            gamma = bn.weight.data
            beta = bn.bias.data
        else:
            gamma = torch.ones_like(bn.running_mean, device=device, dtype=dtype)
            beta = torch.zeros_like(bn.running_mean, device=device, dtype=dtype)

        eps = bn.eps
        running_var = bn.running_var
        scale_factor = gamma / torch.sqrt(running_var + eps)

        w_conv = w_conv * scale_factor.reshape(-1, 1)
        self.weight.data = w_conv.reshape(self.weight.shape)

        b_conv = self.bias.data
        running_mean = bn.running_mean
        b_conv = b_conv - running_mean * scale_factor + beta
        self.bias.data = b_conv

    def unfuse(self) -> tuple[nn.Conv3d, nn.BatchNorm3d]:
        return self.orig_conv, self.orig_bn



class ConvInstanceNorm3d(nn.Conv3d):
    def __init__(self, conv: nn.Conv3d, instnorm: nn.InstanceNorm3d, mean: torch.Tensor, var: torch.Tensor):
        super().__init__(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            conv.bias is not None, conv.padding_mode
        )
        
        mean = torch.squeeze(mean)
        var = torch.squeeze(var)
        assert mean.shape == (conv.out_channels,), "Mean shape mismatch"
        assert var.shape == (conv.out_channels,), "Variance shape mismatch"

        device = conv.weight.device
        dtype = conv.weight.dtype
        self.to(device, dtype)        

        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.bias.data.copy_(conv.bias.data)
        else:
            self.bias = nn.Parameter(torch.zeros_like(mean, device=device, dtype=dtype))
        
        self.orig_conv = conv
        self.orig_instnorm = instnorm
        
        if instnorm.affine:
            gamma = instnorm.weight.data
            beta = instnorm.bias.data
        else:
            gamma = torch.ones_like(mean, device=device, dtype=dtype)
            beta = torch.zeros_like(mean, device=device, dtype=dtype)
        
        eps = instnorm.eps
        scale_factor = gamma / torch.sqrt(var + eps)
        
        # [out_channels, in_channels/groups, *kernel_size]
        w_conv = self.weight.view(self.out_channels, -1)
        w_conv = w_conv * scale_factor.view(-1, 1)
        self.weight.data = w_conv.view(self.weight.shape)
        
        # b_new = (b_conv - mean) * scale + beta
        b_conv = self.bias.data
        b_conv = (b_conv - mean) * scale_factor + beta
        self.bias.data = b_conv

    def unfuse(self) -> tuple[nn.Conv3d, nn.InstanceNorm3d]:
        return self.orig_conv, self.orig_instnorm




class ConvBatch(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv3d(3, 24, 3, 1, 1)
        self.norm = nn.BatchNorm3d(24)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


def test_conv_batchnorm3d():
    torch.manual_seed(42)
    
    # 创建输入数据
    x = torch.randn(2, 3, 16, 16, 16).cuda()
    
    # 原始卷积和BN层
    conv = nn.Conv3d(3, 6, kernel_size=3, padding=1, bias=False).eval().cuda()
    bn = nn.BatchNorm3d(6)
    bn.eval().cuda()  # BN必须处于评估模式
    
    # 融合层
    fused_conv = ConvBatchNorm3d(conv, bn)
    
    # 原始输出
    with torch.no_grad():
        original_out = bn(conv(x))
    
    # 融合输出
    with torch.no_grad():
        fused_out = fused_conv(x)
    
    # 验证输出一致性
    assert torch.allclose(original_out, fused_out, atol=1e-5), "Conv+BN融合失败"

def test_conv_instancenorm3d():
    torch.manual_seed(42)
    
    # 创建输入数据
    x = torch.randn(2, 3, 16, 16, 16)
    
    # 原始卷积和InstanceNorm层
    conv = nn.Conv3d(3, 6, kernel_size=3, padding=1, bias=True)
    instnorm = nn.InstanceNorm3d(6, affine=True)
    
    # 计算当前输入的统计量
    with torch.no_grad():
        conv_out = conv(x)
    
    # 计算每个通道的均值和方差
    mean = conv_out.mean(dim=(0,2,3,4))          # 形状 [6]
    var = conv_out.var(dim=(0,2,3,4), unbiased=True) # 无偏方差
    
    # 融合层
    fused_conv = ConvInstanceNorm3d(conv, instnorm, mean, var)
    
    # 原始输出
    with torch.no_grad():
        original_out = instnorm(conv_out)
    
    # 融合输出
    with torch.no_grad():
        fused_out = fused_conv(x)
    
    # 验证输出一致性
    diff = torch.abs(original_out - fused_out)
    print(diff.max().item(), diff.mean().item())
    # assert torch.allclose(original_out, fused_out, atol=1e-5), "Conv+IN融合失败"

if __name__ == "__main__":
    # test_conv_batchnorm3d()
    # print("test_conv_batchnorm3d passed")
    test_conv_instancenorm3d()
    # print("test_conv_instancenorm3d passed")
    # print("所有测试通过！")