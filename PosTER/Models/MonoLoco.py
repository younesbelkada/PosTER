import torch
import torch.nn as nn

class MonoLoco(nn.Module):
    """
    Class definition for the Looking Model.
    """
    def __init__(self, input_size, p_dropout=0.2, output_size=1, linear_size=256, num_stage=3):
        """[summary]
        Args:
            input_size (int): Input size for the model. If the whole pose needs to be used, the value should be 51.
            p_dropout (float, optional): Dropout rate in the linear blocks. Defaults to 0.2.
            output_size (int, optional): Output number of nodes. Defaults to 1.
            linear_size (int, optional): Size of the fully connected layers in the Linear blocks. Defaults to 256.
            num_stage (int, optional): Number of stages to use in the Linear Block. Defaults to 3.
            bce (bool, optional): Make use of the BCE Loss. Defaults to False.
        """
        super(MonoLoco, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


class Linear(nn.Module):
    """
    Class definition of the Linear block
    """
    def __init__(self, linear_size=256, p_dropout=0.2):
        """
        Args:
            linear_size (int, optional): Size of the FC layers inside the block. Defaults to 256.
            p_dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(Linear, self).__init__()
        ###

        self.linear_size = linear_size
        self.p_dropout = p_dropout

        ###

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(self.p_dropout)

        ###
        self.l1 = nn.Linear(self.linear_size, self.linear_size)
        self.bn1 = nn.BatchNorm1d(self.linear_size)

        self.l2 = nn.Linear(self.linear_size, self.linear_size)
        self.bn2 = nn.BatchNorm1d(self.linear_size)
        
    def forward(self, x):
        # stage I

        y = self.l1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # stage II

        
        y = self.l2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout(y)

        # concatenation

        out = x+y

        return out