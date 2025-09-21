import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnMlpNet(nn.Module):
    def __init__(self,
                 n_variables_out=4, n_input_channels=7, kernel=5,
                 n_filters_c1=32, n_filters_c2=64, n_fc1=2600, n_fc2=84,
                 n_flat_input=9216, activation_last='sigmoid', dropout_bool=False,
                 dropout_perc=0.05):
        super(CnnMlpNet, self).__init__()

        self.conv1 = nn.Conv2d(n_input_channels, n_filters_c1, kernel)
        self.conv2 = nn.Conv2d(n_filters_c1, n_filters_c2, kernel)

        self.dropout1 = nn.Dropout(dropout_perc)
        self.dropout2 = nn.Dropout(dropout_perc)
        self.dropout3 = nn.Dropout(dropout_perc)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(n_flat_input, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.fc3 = nn.Linear(n_fc2, n_variables_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.activation_last = activation_last
        self.dropout_bool = dropout_bool

    def forward(self, input):
        x = self.conv1(input)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.dropout_bool:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.dropout_bool:
            x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.activation_last == 'relu':
            output = self.relu(x)
        elif self.activation_last == 'silu':
            output = self.silu(x)
        elif self.activation_last == 'sigmoid':
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
        INPUT_CHANNELS = 7
        N_VARIABLES_OUT = 4
        IMAGE_SIZE = 128  # 32, 64, 128
        N_FLAT_INPUT = 53824  # // image 128->53824

        net = CnnMlpNet(
            n_variables_out=N_VARIABLES_OUT,
            n_input_channels=INPUT_CHANNELS,
            kernel=5,
            n_filters_c1=32,
            n_filters_c2=64,
            n_fc1=2600,
            n_fc2=1200,
            n_flat_input=N_FLAT_INPUT,
        )
        print(net)

        params = list(net.parameters())
        print(len(params))
        print(params[0].size())

        input = torch.randn(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        out = net(input)
        print(out)
