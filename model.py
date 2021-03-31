import torch
import torch.nn as nn
import torch.nn.functional as F
from block import fusions

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # class 2로 수정 ##################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # x = self.fc(x) ##

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



class ClickbaitDetector(nn.Module):
    def __init__(self, opt):
        super(ClickbaitDetector, self).__init__()
        self.opt = opt

        # image encoder
        self.thumbnail_encoder = resnet50(pretrained=True)
        
        # text encoder
        self.text_encoder = nn.Linear(opt.word_vec_dim, opt.text_dim)
        
        # fusion layer
        if opt.fusion_layer == 'concat':
            self.img_fuse_layer = nn.Sequential(
                nn.Conv2d(2048, opt.fusion_dim//2, 1, 1, 0), # 14*14*fusion_dim//2
                nn.BatchNorm2d(opt.fusion_dim//2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)) # 1*1*fusion_dim//2
            )
            self.text_fuse_layer = nn.Sequential(
                nn.Linear(opt.text_dim, opt.fusion_dim//2),
                nn.BatchNorm1d(opt.fusion_dim//2),
                nn.ReLU()
            )
        elif opt.fusion_layer == 'LinearSum':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.LinearSum([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'ConcatMLP':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.ConcatMLP([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'Block':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.Block([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'MLB':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.MLB([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'Tucker':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.Tucker([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'Mutan':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.Mutan([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'BlockTucker':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.BlockTucker([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'MFB':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.MFB([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'MFH':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.MFH([opt.text_dim, 2048], opt.fusion_dim)

        elif opt.fusion_layer == 'MCB':
            self.img_fuse_layer =  nn.AdaptiveAvgPool2d((1,1))
            self._fusion_layer = fusions.MCB([opt.text_dim, 2048], opt.fusion_dim)
        
        elif opt.fusion_layer == 'MCB.att':
            self.text_fuse_layer = nn.Linear(opt.text_dim, opt.text_dim)
            self._att_fusion_layer = fusions.MCB([opt.text_dim, 2048], opt.att_dim)
            self._att_map_layer = nn.Conv2d(opt.att_dim, 1, 1, 1, 0)
            self._fc_layer = nn.Sequential(
                nn.Linear(2048, opt.fusion_dim),
                nn.BatchNorm1d(opt.fusion_dim),
                nn.ReLU(inplace=True),
            )

        elif opt.fusion_layer == 'Mutan.att':
            self.text_fuse_layer = nn.Linear(opt.text_dim, opt.text_dim)
            self._att_fusion_layer = fusions.Mutan([opt.text_dim, 2048], opt.att_dim)
            self._att_map_layer = nn.Conv2d(opt.att_dim, 1, 1, 1, 0)
            self._fc_layer = nn.Sequential(
                nn.Linear(2048, opt.fusion_dim),
                nn.BatchNorm1d(opt.fusion_dim),
                nn.ReLU(inplace=True),
            )
        
        else:
            raise ValueError('{} not supported for fusion_layer'.format(opt.fusion_layer))

        # classify
        self.classify = nn.Sequential(
            nn.Linear(opt.fusion_dim, opt.fusion_dim),
            nn.BatchNorm1d(opt.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.fusion_dim, 2),
        )

    def _fusion(self, text_code, img_code):
        if self.opt.fusion_layer == 'concat':
            b, _, _, _ = img_code.shape
            img_fusion_code = self.img_fuse_layer(img_code)
            text_fusion_code = self.text_fuse_layer(text_code)
            img_fusion_code = img_fusion_code.view(b, -1)
            code = torch.cat([img_fusion_code, text_fusion_code], dim=1)

        elif self.opt.fusion_layer.endswith('att'):
            b, c, h, w = img_code.shape
            text_code = self.text_fuse_layer(text_code)
            img_code2 = img_code.view(b,c,-1).transpose(1,2).reshape(b*h*w,-1) # b*h*w x c_img
            text_code2 = text_code.unsqueeze(dim=1).expand(-1,h*w,-1).reshape(b*h*w,-1) # b*h*w x c_text

            # calculate attention
            att_map = self._att_fusion_layer([text_code2, img_code2]).view(b,h*w,-1)# b x h*w x att_dim
            att_map = att_map.transpose(1,2) # b x att_dim x h*w
            att_map = att_map.view(b, -1, h, w)
            att_map = self._att_map_layer(att_map) # b x 1 x h x w
            att_map = F.softmax(att_map.view(b,-1), dim=1).view(b, -1, 1) # b x h*w x 1
            img_att_code = torch.bmm(img_code.view(b,c,-1), att_map).view(b, -1) # b x c_img

            # Linear Layer
            code = self._fc_layer(img_att_code)

        else:
            b, _, _, _ = img_code.shape
            img_fusion_code = self.img_fuse_layer(img_code)
            img_fusion_code = img_fusion_code.view(b, -1)
            code = self._fusion_layer([text_code, img_fusion_code])
        
        return code

    def forward(self, text, img):
        text_code = self.text_encoder(text)
        img_code = self.thumbnail_encoder(img)

        code = self._fusion(text_code, img_code)

        y_hat = self.classify(code)

        return y_hat