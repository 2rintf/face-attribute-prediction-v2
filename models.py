from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo   


model_pretrained_path={
    'resnet18':'/home/czd-2019/Downloads/resnet18-5c106cde.pth',
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
        # self.resnet34.cuda()
    
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
        # self.fc.cuda()

    def forward(self,x):
        x = x.view(x.size(0), -1) 
        y = self.fc(x)
        return y

class TailBlock(nn.Module):
    def __init__(self,output_dims):
        super(TailBlock,self).__init__()
        pass


'''
8 subtasks.
Holistic    : 3/11/14/19/21/26/27/32/40   [9]
Hair        : 5/6/9/10/12/18/29/33/34/36  [10]
Eyes        : 2/4/13/16/24  [5]        
Nose        : 8/28/         [2]
Cheek       : 20/30/ (31 sideburns /35 wearing_earrings)   [4]
Mouth(beard): 1/7/22/23/37  [5]
Chin        : 15/17/25      [3]
Neck        : 38/39         [2]     
'''
class MultiTaskNetwork(nn.Module):
    '''
    Completed network.
    '''
    def __init__(self,isPretrained=True):
        super(MultiTaskNetwork,self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        if isPretrained:
            init_pretrained_weights(self.FeatureExtraction,model_pretrained_path['resnet34'])
        
        self.HairPart = SubTask(10)
        self.EyesPart = SubTask(5)
        self.NosePart = SubTask(2)
        self.CheekPart = SubTask(4)
        self.MouthPart = SubTask(5)
        self.ChinPart = SubTask(3)
        self.NeckPart = SubTask(2)
        self.HolisticPart = SubTask(9)

    def forward(self,x):
        x = self.FeatureExtraction(x)
        hair = self.HairPart(x)
        eyes = self.EyesPart(x)
        nose = self.NosePart(x)
        cheek = self.CheekPart(x)
        mouth = self.MouthPart(x)
        chin = self.ChinPart(x)
        neck = self.NeckPart(x)

        holistic = self.HolisticPart(x)
        return hair,eyes,nose,cheek,mouth,chin,neck,holistic

class StructureCheck(nn.Module):
    '''
    Just for test on cifar-10.
    '''
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
