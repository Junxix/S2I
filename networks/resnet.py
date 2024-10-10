import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, pretrained=True):
        super(SupConResNet, self).__init__()
        if name == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            dim_in = 2048  
        elif name == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            dim_in = 512  
        else:
            raise ValueError(f"Model {name} not supported")
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


