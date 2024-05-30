"""
Author：Xie Lei
Data：2023/06/27
"""
import torch
from torch.nn import Linear, BatchNorm1d
import numpy as np
import sparsemax
import torch.nn as nn
import torch.nn.functional as F


#initialize the FC layer
def initialize_FC(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

#initialize the self-attention layer
def initialize_sa(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(8*input_dim))
    torch.nn.init.xavier_normal_(module.query.weight, gain=gain_value)
    torch.nn.init.xavier_normal_(module.key.weight, gain=gain_value)
    torch.nn.init.xavier_normal_(module.value.weight, gain=gain_value)
    return

#classical self-attention module
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = F.softmax(
            torch.bmm(Q.unsqueeze(2), K.unsqueeze(1)) / torch.sqrt(torch.tensor(Q.shape[-1]).float()), dim=-1)
        output = torch.matmul(attention_weights, V.unsqueeze(-1))
        output = torch.sum(output, dim=-1)
        return output

# Ghost Batch Normalization  module
class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)

#  Gated Unit Layer
class GLU_Layer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        virtual_batch_size=128,
        momentum=0.02
    ):
        super(GLU_Layer, self).__init__()
        self.Self = SelfAttention(input_dim=input_dim, hidden_dim=2*output_dim)
        self.fc = nn.Linear(input_dim, 2*output_dim)
        initialize_sa(self.Self, input_dim, 2*output_dim)
        initialize_FC(self.fc, input_dim, 2 * output_dim)
        self.bn = GBN(
            2*output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )
        self.output_dim = output_dim
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(x)
        # according to different datasets,
        # attention mechanisms can be added or traditional FC can be used for dimension expansion in this part.
        # x = self.Self(x)
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out


class SelfAttenTransformer(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            virtual_batch_size=128,
            momentum=0.02,
            mask_type="sparsemax",
    ):
        super(SelfAttenTransformer, self).__init__()
        self.Self = SelfAttention(input_dim=input_dim, hidden_dim=output_dim)
        initialize_sa(self.Self, input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        if mask_type == "sparsemax":
            self.selector = sparsemax.Sparsemax(dim=-1)
        else:
            self.selector = sparsemax.Entmax15(dim=-1)

    def forward(self,x):
        x = self.Self(x)
        x = self.bn(x)
        x = self.selector(x)
        return x


class Covfeaturetransformer(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim = 256 ,
            virtual_batch_size=128,
            momentum=0.02,
    ):
        super(Covfeaturetransformer,self).__init__()
        self.fc = Linear(input_dim, hidden_dim, bias=False)
        self.cov = nn.Conv1d(1,1,kernel_size=2,stride=1)
        self.bn = GBN(hidden_dim-1, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.cglu = GLU_Layer(hidden_dim-1,output_dim,virtual_batch_size=128,momentum=0.02)

    def forward(self,x):
        x = self.fc(x)
        x=x.unsqueeze(1)
        x = self.cov(x)
        x = x.squeeze(1)
        x = self.bn(x)
        x = self.cglu(x)
        return x


class FeatureBlock(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            output_dim,
    ):
        super(FeatureBlock, self).__init__()
        self.SAT = SelfAttenTransformer(input_dim,feature_dim,
                                        virtual_batch_size=128,
                                        momentum=0.02,
                                        mask_type="Entmax15")
        self.CFT = Covfeaturetransformer(feature_dim,output_dim)

    def forward(self,input,x):
        M = self.SAT(input)
        Selected_feature = torch.mul(M,x)
        output = self.CFT(Selected_feature)
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        output = torch.add(input,output)
        output = output*scale
        return output


class Firstprocess(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            output_dim
    ):
        super(Firstprocess, self).__init__()
        self.SAT = SelfAttenTransformer(input_dim,feature_dim,
                                        virtual_batch_size=128,
                                        momentum=0.02,
                                        mask_type="Entmax15")
        self.CFT = Covfeaturetransformer(feature_dim,output_dim)
        self.feature_dim = feature_dim

    def forward(self,x):
        M = self.SAT(x)
        selected_feature = torch.mul(M,x)
        output = self.CFT(selected_feature)
        return output


class singlelink(torch.nn.Module):
    def __init__(
            self,
            n_layer,
            input_dim,
            feature_dim,
            output_dim
    ):
        super(singlelink, self).__init__()
        self.layers = n_layer
        self.Firstprocess = Firstprocess(input_dim,feature_dim,output_dim)
        self.featureBlock = torch.nn.ModuleList()
        for step in range(self.layers):
            featureBlock = FeatureBlock(input_dim=output_dim,feature_dim=feature_dim,output_dim=output_dim)
            self.featureBlock.append(featureBlock)

    def forward(self,x):
        input = self.Firstprocess(x)
        for step in range(self.layers):
            input = self.featureBlock[step](input,x)
        return input


class  NET(torch.nn.Module):
    def __init__(
            self,
            hy_num,
            n_layer,
            input_dim,
            feature_dim,
            output_dim
    ):
        super(NET,self).__init__()
        self.hynum = hy_num
        self.n_layer = n_layer
        self.input = input_dim
        self.feature = feature_dim
        self.output = output_dim
        self.hynet = torch.nn.ModuleList()
        self.register_parameter('weight', nn.Parameter(torch.ones(1,hy_num)/hy_num))
        for step in range(self.hynum):
            singlelink_module = singlelink(n_layer=self.n_layer,input_dim=self.input,feature_dim=self.feature,output_dim=self.output)
            self.hynet.append(singlelink_module)
    def forward(self,x):
        output =[]
        for step in range(self.hynum):
            output_step = self.hynet[step](x)
            output.append(output_step)
        mask = self.weight.T
        data_tensor = torch.stack(output)
        data_tensor = data_tensor*mask.unsqueeze(2)
        mean = torch.sum(data_tensor, dim=0)
        return mean
















