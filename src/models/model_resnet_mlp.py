import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResnetMLP(nn.Module):
    def __init__(self,
                 pretrained=True,
                 n_input_channels=4,
                 n_variables_out=4,
                 n_flat_input=1000,
                 activation_last='sigmoid'
                 ):
        super(ResnetMLP, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet18(weights=weights)
        if n_input_channels > 3:
            resnet.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        self.resnet_linear = nn.Linear(512, n_flat_input)

        # FC Layer
        self.fc_regression = nn.Linear(n_flat_input, n_variables_out)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.activation_last = activation_last

    def forward(self, input_f):
        x = self.resnet18(input_f)
        x = x.view(x.size(0), -1)
        x = self.resnet_linear(x)
        x = self.fc_regression(x)

        if self.activation_last == 'sigmoid':
            x = self.sigmoid(x)
            output = x * 2
        elif self.activation_last == 'softplus':
            output = self.softplus(x)
        else:
            output = x
        return output


if __name__ == '__main__':
    TEST_MODEL = True

    if TEST_MODEL:
        INPUT_CHANNELS = 4
        N_VARIABLES_OUT = 4
        IMAGE_SIZE = 128
        N_FLAT_INPUT = 1000

        net = ResnetMLP(
            pretrained=True,
            n_input_channels=INPUT_CHANNELS,
            n_variables_out=N_VARIABLES_OUT,
            n_flat_input=N_FLAT_INPUT,
            activation_last='sigmoid'
        )
        print(net)

        params = list(net.parameters())
        print(len(params))
        print(params[0].size())

        input_vector = torch.randn(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        out = net(input_vector)
        print(out)
