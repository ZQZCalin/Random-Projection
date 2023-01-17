import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchsummary import summary


class ApproximateFC(nn.Module):
    '''
    Composition of random projection and approximate linear layer.
    '''
    def __init__(self, reduced_dim):
        super(ApproximateFC, self).__init__()
        # non-trainable random projection, initialized with standard normal
        self.projection = nn.Linear(in_features=2048, out_features=reduced_dim, bias=True)
        self.projection.weight.requires_grad = False
        self.projection.bias.requires_grad = False
        self.reset_projection(mean=0., std=1.)
        # approximate linear layer
        self.linear = nn.Linear(in_features=reduced_dim, out_features=100, bias=True)
        # store original layer for reset

    def forward(self, x):
        return self.linear(self.projection(x))

    def reset_projection(self, mean, std):
        nn.init.normal_(self.projection.weight, mean, std)
        nn.init.normal_(self.projection.bias, mean, std)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
        

def get_pretrained_resnet50(no_last=False):
    '''
    Returns a Resnet-50 model pretrained on Imagenet, with last layer randomly initialized.
    '''
    net = torchvision.models.resnet50(pretrained=True)
    if no_last:
        net.fc = Identity()
    else:
        net.fc = nn.Linear(in_features=2048, out_features=100, bias=True)

    return net


def get_pretrained_resnet50_approximate(reduced_dim=100):
    '''
    Returns a Resnet-50 model pretrained on Imagenet, with last layer substituted by
    composition of random projection and approximated linear layer.
    '''
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = ApproximateFC(reduced_dim)

    return net


# testing code
if __name__ == "__main__":
    net = get_pretrained_resnet50_approximate(reduced_dim=1000)
    # print(net.fc)
    net.fc.reset_projection(mean=0., std=5.)

    summary(net.cuda(), input_size=(3,224,224))


# to do: 
# - (done) load partial pretrained resnets
# - (done) add random projection layer before last layer
