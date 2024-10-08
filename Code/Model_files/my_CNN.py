import torch.nn as nn
import torch
import math


class Residual_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate=0.5
    ):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(out_channels)
        self.batchnorm2 = nn.BatchNorm1d(out_channels)
        self.ELU = nn.ELU()
        if in_channels != out_channels:
            self.adjust_dims = nn.Conv1d(in_channels, out_channels, 1, stride, 0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.ELU(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out += self.adjust_dims(residual) if hasattr(self, "adjust_dims") else residual
        out = self.ELU(out)
        return out


class myCNN(nn.Module):
    def __init__(
        self,
        sequence_length,
        n_labels,
        n_ressidual_blocks,
        in_channels,
        out_channels,
        kernel_size,
        max_pooling_kernel_size,
        dropout_rate,
        ffn_size_1,
        ffn_size_2,
        padding="same",
        stride=1,
    ):
        super(myCNN, self).__init__()

        # first assert that sequence length after the max pooling operations is greater than 100
        # assert sequence_length // (max_pooling_kernel_size ** n_ressidual_blocks/2) > 1
        # kernel size, channels, stride and padding are shared among all the residual blocks
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.residual_blocks = nn.ModuleList()
        assert (
            n_ressidual_blocks % 2 == 0
        ), "The number of residual blocks must be even!"
        self.max_pooling = nn.MaxPool1d(
            max_pooling_kernel_size
        )  # stride default to kernel size

        for i in range(n_ressidual_blocks):
            self.residual_blocks.append(
                Residual_Block(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dropout_rate,
                )
            )
            in_channels = out_channels

        self.inp_length = (
            sequence_length // (max_pooling_kernel_size ** int(n_ressidual_blocks / 2))
        ) * out_channels

        # 1d batch norm
        # import pdb; pdb.set_trace()
        self.bn1 = nn.BatchNorm1d(self.inp_length)
        print("inp_length", self.inp_length)
        self.ffn = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.inp_length, ffn_size_1),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_size_1, ffn_size_2),
            nn.ELU(),
            nn.Linear(ffn_size_2, n_labels),
        )

    def forward(self, z):
        for j, block in enumerate(self.residual_blocks):
            z = block(z)
            if j % 2 == 0:
                z = self.max_pooling(z)

        z = z.reshape(z.size(0), -1)
        z = self.bn1(z)
        logits = self.ffn(z)

        return logits  # this is batch x labels. The loss function applies a sigmoid elementwise.


if __name__ == "__main__":
    # let's try out this with gene lengths
    dna = torch.rand(2, 4, 3020)
    design_matrix = torch.rand(1, 5, 30)
    gene_lengths = torch.rand(2, 1)

    model = myCNN(3020, 5, 2, 4, 16, 3, 3, 0.5, 128, 32)
    outputs = model(dna)
    print(outputs.shape)
