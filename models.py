from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo   


model_pretrained_path={
    'resnet34':"/home/czd-2019/Downloads/resnet34-333f7ec4.pth"
}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()

        self.resnet34 = models.resnet34()
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-1])
        self.resnet34.cuda()
    
    def forward(self,x):
        y = self.resnet34(x)
        return y

class SubTask(nn.Module):
    def __init__(self, output_dims):
        super(SubTask,self).__init__()
        # self.fc1 = nn.Linear(512,512).cuda()
        # self.fc2 = nn.Linear(512,128).cuda()
        # self.fc3 = nn.Linear(128,output_dims).cuda()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dims),
        )
        self.fc.cuda()

    def forward(self,x):
        x = x.view(x.size(0), -1) 
        y = self.fc(x)
        return y

class TailBlock(nn.Module):
    def __init__(self,output_dims):
        super(TailBlock,self).__init__()
        pass

class MultiTaskNetwork(nn.Module):
    def __init__(self):
        super(MultiTaskNetwork,self).__init__()
        self.FeatureExtraction = FeatureExtraction()

        init_pretrained_weights(self.FeatureExtraction,model_pretrained_path['resnet34'])
        self.HairPart = []
        self.EyesPart = []
        self.HairColoePart = []
        

        self.HolisticPart = []

    def forward(self):
        pass

class StructureCheck(nn.Module):
    def __init__(self):
        super(StructureCheck,self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        init_pretrained_weights(self.FeatureExtraction,model_pretrained_path['resnet34'])
        self.task1 = SubTask(output_dims=10)

    def forward(self,x):
        x = self.FeatureExtraction(x)
        x = self.task1(x)
        return x


def init_pretrained_weights(model, pretrain_dict_path):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = torch.load(pretrain_dict_path)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_pretrained_path))
