from torch import nn

nonlinearity = nn.ReLU

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlockV2(nn.Module):
    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 is_deconv=True):
        
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)    
    
class DecoderBlockLinkNet(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv = False,
                ):
        super().__init__()

        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0
            
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding = 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)
        
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding = conv_padding)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DecoderBlockLinkNetV2(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=4,
                 is_deconv=False,
                 is_upsample=True,
                 ):
        super().__init__()

        
        self.is_upsample = is_upsample
        
        if kernel_size == 3:
            conv_stride = 1
        elif kernel_size == 1:
            conv_stride = 1
        elif kernel_size == 4:
            conv_stride = 2

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               3,
                               padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              kernel_size,
                                              stride=conv_stride,
                                              padding=1)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               3,
                               padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        if self.is_upsample:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = self.relu3(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = self.relu3(x)
        return x

class DecoderBlockLinkNetInceptionV2(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 n_filters=256,
                 last_padding=0,
                 kernel_size=3,
                 is_deconv = False
                ):
        super().__init__()

        if kernel_size == 3:
            conv_stride = 1              
        elif kernel_size == 1:
            conv_stride = 1              
        elif kernel_size == 4:
            conv_stride = 2              
        
        # B, C, H, W -> B, out_channels, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               3,
                               padding = 2)
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nonlinearity(inplace=True)

        # B, out_channels, H, W -> B, out_channels, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(out_channels,
                                              out_channels,
                                              kernel_size,
                                              stride=conv_stride,
                                              padding=1)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)
            
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nonlinearity(inplace=True)

        # B, out_channels, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(out_channels,
                               n_filters,
                               3,
                               padding = 1+last_padding)
        
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x    