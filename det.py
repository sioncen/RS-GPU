import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# from restnet18.restnet18 import RestNet18
# from resnet50 import ResNet50
import torch.backends.cudnn as cudnn
from readdate import MyDataset
from torchvision import models
import os
import shutil
from datetime import datetime
import time


def main():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    test_data = MyDataset('test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device('cuda')
    print(device)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 45)
    checkpoint = torch.load(r'D:\Project\RS\GRN-SNDL-master\2.pth')
    # try:
    #     checkpoint.eval()
    # except AttributeError as error:
    #     print(error)
    ### 'dict' object has no attribute 'eval'

    model.load_state_dict(checkpoint)  # (checkpoint['state_dict'])
    model = model.to(device)
    # model_CKPT = torch.load(checkpoint_PATH)
    # model.load_state_dict(model_CKPT['state_dict'])
    # print('loading checkpoint!')
    # optimizer.load_state_dict(model_CKPT['optimizer'])

    begin = time.time()
    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        index = 1
        wrong_num = 0
        for x, label in test_loader:
            inputs, label = x.to(device), label.to(device)

            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            if correct == 1:
                print(index, label, pred, '===')
            else:
                wrong_num += 1
                print(index, label, pred, '&&&&&&&&&&&&&')
            total_correct += correct
            total_num += inputs.size(0)
            index += 1
            # print(correct)

        acc = total_correct / total_num

    end = time.time()
    time_used = (end - begin) / 60
    print('test acc:{},right:{},wrong:{},time:{}'.format(acc, total_correct, wrong_num, time_used))


if __name__ == '__main__':
    main()
