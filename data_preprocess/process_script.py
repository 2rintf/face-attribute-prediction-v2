import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path


raw_file = "/home/czd-2019/Projects/celebA_dataset/Anno/list_attr_celeba.txt"

holistic = [3,11,14,19,21,26,27,32,40]
hair = [5,6,9,10,12,18,29,33,34,36]
eyes = [2,4,13,16,24]
nose = [8,28]
cheek = [20,30,31,35]
mouth = [1,7,22,23,37]
chin = [15,17,25]
neck = [38,39]

total_label = holistic+hair+eyes+nose+cheek+mouth+chin+neck
print(total_label)



train_img_list = []
train_label_list = []

val_img_list = []
val_label_list = []

test_img_list = []
test_label_list = []

count = 1
for line in open(raw_file,'r'):
    sample = line.split()
    if len(sample)!=41:
        print("Not sample line.")
        continue
    
    img_t = sample[0]
    # Change -1 to 0.
    label_t = [1 if int(i)==1 else 0 for i in sample[1:]]
    # Change order of label.
    label_t = [label_t[i-1] for i in total_label]

    if count <= 162770:
        train_img_list.append(img_t)
        train_label_list.append(label_t)
    elif count>162770 and count <= 182637:
        val_img_list.append(img_t)
        val_label_list.append(label_t)
    elif count>182637 and count <= 202599:
        test_img_list.append(img_t)
        test_label_list.append(label_t)
    count+=1

print(count)
print(len(train_img_list))
print(len(val_img_list))
print(len(test_img_list))

dic = {
    'train_part.txt':[train_img_list,train_label_list],
    'val_part.txt':[val_img_list,val_label_list],
    'test_part.txt':[test_img_list,test_label_list]
}

print(len(dic['train_part.txt'][0]))

for fn in ['train_part.txt','val_part.txt','test_part.txt']:
    file = open(os.path.join("./data_preprocess",fn),'w')
    for img,label in zip(dic[fn][0],dic[fn][1]):
        label = [str(i) for i in label]
        file.write(img+" ")
        for l in label:
            file.write(l+" ")
        file.write("\n")
    file.close()

print('done.')

