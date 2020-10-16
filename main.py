from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

import torch.utils.model_zoo as model_zoo   
import torch.utils.data as Data
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt

from models import MultiTaskNetwork,StructureCheck

from PIL import Image,ImageShow,ImageTransform
import os


def train(net,train_loader,test_loader):
    step = 0
    loss_output_step = 200.0
    optimizer = torch.optim.Adam(net.parameters(),LR)
    loss_func = nn.CrossEntropyLoss().cuda()
    print(optimizer)
    for epoch in range(EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0

        # choose any frequncy of output you like.
        if epoch%1==0 and epoch != 0:
            # train_data test
            correct_t = 0
            total_t = 0
            for data in train_loader:
                images_t, labels_t = data
                outputs_t = net(images_t.cuda())


                _, predicted_t = torch.max(outputs_t.data, 1)
                total_t += labels_t.size(0)
                correct_t += (predicted_t == labels_t.cuda()).sum().item()  # item() -> Tensor转标量

            print('epoch: %d | Train dataset accuracy: %.3f %%' % (epoch, 100 * correct_t*1.0 / total_t))
            train_set_acc.append(100 * correct_t*1.0 / total_t)

            # test
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                outputs = net(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()  # item() -> Tensor转标量

            print('epoch: %d | Test accuracy: %.3f %%' % (epoch, 100 * correct * 1.0/ total))
            test_set_acc.append(100 * correct * 1.0/ total)

        for i, data in enumerate(train_loader):
            step+=1
            # get the inputs
            inputs, labels = data
        
            # print(labels.shape)
            # print(labels)
            # exit(0)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = net(inputs.cuda())
            loss = loss_func(output, labels.cuda())
            loss.backward()
            optimizer.step()



            # print statistics
            running_loss += loss.item()

            if i % int(loss_output_step) == 0 and i != 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / loss_output_step))
                loss_val.append(running_loss / loss_output_step)
                real_loss_step.append(step)
                running_loss = 0.0

        # scheduler.step()
        # print(scheduler.get_lr())
    print('Finished Training')

    torch.save(net.state_dict(), "resnet34_cifar10.pth")


    correct_t = 0
    total_t = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images_t, labels_t = data
            outputs_t = net(images_t.cuda())
            _, predicted_t = torch.max(outputs_t.data, 1)
            total_t += labels_t.size(0)
            correct_t += (predicted_t == labels_t.cuda()).sum().item()  # item() -> Tensor转标量

        print('FINAL | Train dataset accuracy: %.3f %%' % ( 100 * correct_t*1.0 / total_t))
        train_set_acc.append(100 * correct_t * 1.0 / total_t)

        for data in test_loader:
            images, labels = data
            outputs = net(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item() # item() -> Tensor转标量
        print('FINAL | Test dataset accuracy: %.3f %%' % (100 * correct / total))
        test_set_acc.append(100 * correct * 1.0 / total)



def main():
    net = StructureCheck()
    print(net)


    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    train_data= torchvision.datasets.CIFAR10(
        root="/home/czd-2019/torch_foobar/cifar-10",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD
    )
    print(train_data.data.shape)
    # img, target= train_data.__getitem__(0)
    # img.save("./first.jpg")
    # print(target)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_data= torchvision.datasets.CIFAR10(
        root="/home/czd-2019/torch_foobar/cifar-10",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    print(test_data.data.shape)
    

    net.train()
    train(net,train_loader,test_loader)

    x1 = range(EPOCH)
    x2 = range(EPOCH)
    x3 = real_loss_step # batch_size 和 train_dataset_size 不是整数倍

    y1 = train_set_acc
    y2 = test_set_acc
    y3 = loss_val


    plt.title("ResNet34-CIFAR10-dataset_accuracy")
    # plt.title("LeNet-FashionMNIST-dataset_accuracy")
    plt.plot(x1,y1,'o-b',label='train_acc')
    plt.plot(x2,y2,'o-r',label='test_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("./acc.png")
    plt.cla()
    plt.title("ResNet-CIFAR10-Loss")
    # plt.title("LeNet-FashionMNIST-Loss")
    plt.plot(x3,y3,'o-y',label='loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig("./loss.png")
    '''
    Finished Training
    FINAL | Train dataset accuracy: 93.804 %
    FINAL | Test dataset accuracy: 76.160 %
    '''


def save_cifar10_pic():
    train_data= torchvision.datasets.CIFAR10(
        root="/home/czd-2019/torch_foobar/cifar-10",
        train=True,
        # transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD
    )
    print(train_data.data.shape)

    for i in range(100):
        img, target= train_data.__getitem__(i)
        path = os.path.join("./test_pic_cifar10",str(target))
        if not os.path.isdir(path):
            os.makedirs(path)
        img.save(path+"/"+str(i)+".jpg")       

    # img, target= train_data.__getitem__(0)
    # img.save("./0.jpg")
    # print(target)

    # img, target= train_data.__getitem__(100)
    # img.save("./100.jpg")
    # print(target)


def make_hook(name, flag):
     if flag == 'forward':
         def hook(m, input, output):
             inter_feature[name] = output
         return hook
     elif flag == 'backward':
         def hook(m, input, output):
             inter_gradient[name] = output
         return hook
     else:
         assert False


def model_check():


    model = StructureCheck()
    model.cuda()
    checkpoint = torch.load("/home/czd-2019/Projects/face-attribute-prediction-v2/resnet34_cifar10.pth")
    model.load_state_dict(checkpoint)

    model.eval()
    #! 做一个hook，获取最后一层fc中的128维vector
    for name,m in model.named_modules():
        if name == "task1.fc.3":
            m.register_forward_hook(make_hook("task1.fc.3",'forward'))
    

    # classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # print(model)
    # img = Image.open("/home/czd-2019/Downloads/dog.png")
    img = Image.open("./0.jpg")
    img = img.resize((32,32))
    img = img.convert('RGB')
    print(img.size)

    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                  std=[0.5, 0.5, 0.5])
    trans = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    pred = model(trans(img).unsqueeze(0).cuda())

    print(pred)
    output_s = [torch.sigmoid(i) for i in pred]
    print(output_s)

    print(inter_feature['task1.fc.3'][0].shape)



if __name__ == "__main__":
    LR = 0.001
    BATCH_SIZE = 128
    EPOCH = 15
    DOWNLOAD = False
    train_set_acc=[]
    test_set_acc=[]
    loss_val=[]
    real_loss_step=[]

    inter_feature = {}
    inter_gradient = {}


    # main()
    model_check()
    # save_cifar10_pic()