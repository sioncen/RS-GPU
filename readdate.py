import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


# 自定义图片图片读取方式，可以自行增加resize、数据增强等操作
def MyLoader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 构造函数设置默认参数
    def __init__(self, torv="train", transform=None, target_transform=None, loader=MyLoader):
        # rootdir = r'F:\Data\UCMerced_LandUse\images\\'+torv
        # rootdir2 = r'F:\Data\UCMerced_LandUse\labels\\'+torv
        rootdir = r'D:\NWPU-RESISC45\images\\' + torv
        rootdir2 = r'D:\NWPU-RESISC45\labels\\' + torv
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        list2 = os.listdir(rootdir2)  # 列出文件夹下所有的目录与文件
        imgs = []
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            path2 = os.path.join(rootdir2, list2[i])
            with open(path2, 'r') as fl:
                for line in fl:
                    words = line.split()
                    label = words[0]
            imgs.append((path, int(label)))
            # print(path, int(label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# train_data = MyDataset(txt=root + '\\' + 'train.txt', transform=transforms.ToTensor())
# test_data = MyDataset(txt=root + '\\' + 'test.txt', transform=transforms.ToTensor())
#
# # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_data, batch_size=64)

# print('加载成功！')
