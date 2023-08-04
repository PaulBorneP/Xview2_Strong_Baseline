import abc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet34_Weights

# from .senet import se_resnext50_32x4d, senet154
# from .dpn import dpn92


class ConvReluBN(nn.Module):
    """ Convolution -> BatchNorm -> ReLU

        Args:
            in_channels : number of input channels
            out_channels : number of output channels
            kernel_size : size of convolution kernel

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ConvRelu(nn.Module):
    """ Convolution -> ReLU.

        Args:   
            in_channels : number of input channels
            out_channels : number of output channels
            kernel_size : size of convolution kernel
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    """ Squeeze-and-Excitation Module, according to https://arxiv.org/pdf/1709.01507.pdf concat is better.

        Args:
            channels : equals to input_channels and output_channels*2
            reduction : reduction ratio int the squeeze layer
            concat : if True, concat spatial and channel se, else add them

    """

    def __init__(self, channels: int, reduction: int = 16, concat: bool = False) -> None:

        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se


class LocModel(nn.Module, metaclass=abc.ABCMeta):
    """ Base class for all localization models."""
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.forward_once(x))

    def _initialize_weights(self) -> None:
        """ Initialize weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Res34_Unet_Loc(LocModel):
    """Unet model with a resnet34 encoder used for localization. 
        
        Args:
            pretrained : if True, use pretrained resnet34 weights.

    """
    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__()

        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(
            decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(
            decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(
            decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(
            decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()
        # pretrained argument was deprecated so we changed to weights

        # throw error if pretrained is not a bool
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained argument should be a bool")
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        if weights is not None:
            print(f"using weights from {weights}")
        encoder = torchvision.models.resnet34(weights=weights)
        self.conv1 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu)
        self.conv2 = nn.Sequential(
            encoder.maxpool,
            encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        return dec10


class SeResNext50_Unet_Loc(LocModel):
    """ ResNext Unet model with se blocks used for localization tasks.

        Args:
            pretrained : name of pretrained model, Default: 'imagenet'
    """

    def __init__(self, pretrained: str = 'imagenet', **kwargs) -> None:
        super(SeResNext50_Unet_Loc, self).__init__()

        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(
            decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(
            decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(
            decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(
            decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1,
                                   encoder.layer0.relu1)  # encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the model for one image."""

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        return dec10


class Dpn92_Unet_Loc(LocModel):
    def __init__(self, pretrained: str = 'imagenet+5k', **kwargs) -> None:
        """Dual path network Unet model for localization tasks.
        
            Args:
                pretrained : name of pretrained model, Default: 'imagenet+5k'
        """
        super(Dpn92_Unet_Loc, self).__init__()

        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1]),
                                     SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2]),
                                     SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3]),
                                     SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4]),
                                     SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)


        self.conv1 = nn.Sequential(
            encoder.blocks['conv1_1'].conv,  # conv
            encoder.blocks['conv1_1'].bn,  # bn
            encoder.blocks['conv1_1'].act,  # relu
        )
        self.conv2 = nn.Sequential(
            encoder.blocks['conv1_1'].pool,  # maxpool
            *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
        )
        self.conv3 = nn.Sequential(
            *[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(
            *[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(
            *[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the model for one image."""

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9,
                                       enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


class SeNet154_Unet_Loc(LocModel):
    """Squeeze-and-excitation Unet model for localization tasks."""
    def __init__(self, pretrained: str = 'imagenet', **kwargs) -> None:
        super(SeNet154_Unet_Loc, self).__init__()
        encoder_filters = [128, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(
            decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(
            decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(
            decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(
            decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = senet154(pretrained=pretrained)

        # conv1_new = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1, encoder.layer0.conv2,
                                   encoder.layer0.bn2, encoder.layer0.relu2, encoder.layer0.conv3, encoder.layer0.bn3,
                                   encoder.layer0.relu3)
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model for one input image."""

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        return dec10


class Siamese(nn.Module,metaclass=abc.ABCMeta,):
    """ Abstract class for siamese networks. To create a 
    siamese network, inherit from this class and the class of the localization model.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.res = nn.Conv2d(
            decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass  for two inputs (that have been concateneted chanelwise to one). """
        output1 = self.forward_once(x[:, :3, :, :])
        output2 = self.forward_once(x[:, 3:, :, :])
        return self.res(torch.cat([output1, output2], 1))


class Res34_Unet_Double(Siamese, Res34_Unet_Loc):
    """ ResNet34 Unet model for classification tasks."""
    def encode_once(self, x: torch.Tensor) -> torch.Tensor:
        """ Encode one image with the encoder part of the model."""
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return enc5
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """ Get the embeddings of the images."""
        encoded1 = self.encode_once(x[:, :3, :, :])
        encoded2 = self.encode_once(x[:, 3:, :, :])
        encoded = torch.cat([encoded1, encoded2], 1)
        return F.adaptive_avg_pool2d(encoded, 1).view(encoded.shape[0], -1)

class SeNet154_Unet_Double(Siamese, SeNet154_Unet_Loc):
    """ Squeeze-and-excitation Unet model for classification tasks."""
    pass


class SeResNext50_Unet_Double(Siamese, SeResNext50_Unet_Loc):
    """ ResNext Unet model with SE blocks for classification tasks."""
    pass


class Dpn92_Unet_Double(Siamese, Dpn92_Unet_Loc):
    """ DPN Unet model for classification tasks."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2
        self.res = self.res = nn.Conv2d(
            decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)
    pass

if __name__ == "__main__":
    import torchsummary
    model = Res34_Unet_Double().to('cuda')
    torchsummary.summary(model, (6, 608, 608))