import torch.nn as nn
import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=True)

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        #Defining U-Net Generator Layers
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride = 1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride = 1, padding=1)
        self.pool1 =  nn.MaxPool2d(2, stride=2, padding = 0, ceil_mode = False)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride = 1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride = 1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding = 0, ceil_mode = False)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride = 1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride = 1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride = 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding = 0, ceil_mode = False)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride = 1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, padding = 0, ceil_mode = False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv6_1 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.upsample6 = nn.ConvTranspose2d(512, 512, 2, stride = 2, bias = False)
        self.conv7_1 = nn.Conv2d(1024, 512, 3, stride = 1, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, stride = 1, padding=1)
        self.upsample7 = nn.ConvTranspose2d(512, 256, 2, stride = 2, bias = False)
        self.conv8_1 = nn.Conv2d(512, 256, 3, stride = 1, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, stride = 1, padding=1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, stride = 1, padding=1)
        self.upsample8 = nn.ConvTranspose2d(256, 128, 2, stride = 2, bias = False)
        self.conv9_1 = nn.Conv2d(256, 128, 3, stride = 1, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, stride = 1, padding=1)
        self.upsample9 = nn.ConvTranspose2d(128, 64, 2, stride = 2, bias = False)
        self.conv10_1 = nn.Conv2d(128, 64, 3, stride = 1, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, 3, stride = 1, padding=1)
        self.output = nn.Conv2d(64, 1, 1, stride = 1, padding=0)
        #activation layers
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


        #initialize encoder layer weights using VGG16 pretrained weights
        self.copy_params_from_vgg16(vgg16)
        #initialize decoder layer weights randomly using Xavier Initialization
        self._initialize_weights()


    def _initialize_weights(self):
        #only randomly initialize weights of decoder layers we use Xavier Initialization
        decoder_layers = [self.conv6_1, self.conv6_2, self.conv6_3, self.upsample6,
                         self.conv7_1, self.conv7_2, self.conv7_3, self.upsample7,
                         self.conv8_1, self.conv8_2, self.conv8_3, self.upsample8,
                         self.conv9_1, self.conv9_2, self.upsample9,
                         self.conv10_1, self.conv10_2, self.output]
        for layer in decoder_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()


    def forward(self, x):

        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        #Save Layer 1 Output
        conv1_fmap = x
        x = self.pool1(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        #Save Layer 2 Output
        conv2_fmap = x
        x = self.pool2(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        #Save Layer 3 Output
        conv3_fmap = x
        x = self.pool3(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        #Save Layer 4 Output
        conv4_fmap = x
        x = self.pool4(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))

        x = self.relu(self.conv6_1(x))
        x = self.relu(self.conv6_2(x))
        x = self.relu(self.conv6_3(x))
        x = self.upsample6(x)

        #Conactenate upsampled fmap with conv4_fmap
        x = torch.cat([conv4_fmap,x], dim = 1)
        x = self.relu(self.conv7_1(x))
        x = self.relu(self.conv7_2(x))
        x = self.relu(self.conv7_3(x))
        x = self.upsample7(x)


        #Conactenate upsampled fmap with conv3_fmap
        x = torch.cat([conv3_fmap,x], dim = 1)
        x = self.relu(self.conv8_1(x))
        x = self.relu(self.conv8_2(x))
        x = self.relu(self.conv8_3(x))
        x = self.upsample8(x)

        #Conactenate upsampled fmap with conv2_fmap
        x = torch.cat([conv2_fmap,x], dim = 1)
        x = self.relu(self.conv9_1(x))
        x = self.relu(self.conv9_2(x))
        x = self.upsample9(x)

        #Conactenate upsampled fmap with conv1_fmap
        x = torch.cat([conv1_fmap,x], dim = 1)
        x = self.relu(self.conv10_1(x))
        x = self.relu(self.conv10_2(x))
        x = self.sigmoid(self.output(x))

        return x


    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1,
            self.conv1_2,
            self.pool1,
            self.conv2_1,
            self.conv2_2,
            self.pool2,
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.pool3,
            self.conv4_1,
            self.conv4_2,
            self.conv4_3,
            self.pool4,
            self.conv5_1,
            self.conv5_2,
            self.conv5_3,
        ]


        vgg_blocks = [l for l in vgg16.features.children() if isinstance(l, nn.Conv2d)]
        self_blocks = [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, self.conv3_1, self.conv3_2, self.conv3_3, self.conv4_1, self.conv4_2, self.conv4_3, self.conv5_1, self.conv5_2, self.conv5_3]

        for l1, l2 in zip(vgg_blocks, self_blocks):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

        #fix layer weights before last two convolutional layers of VGG16
        fixed_weight_layers = [self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3]
        for layer in fixed_weight_layers:
            for param in layer.parameters():
                param.requires_grad = False
