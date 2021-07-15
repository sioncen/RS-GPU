import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# from restnet18.restnet18 import RestNet18
from resnet50 import ResNet50
import torch.backends.cudnn as cudnn
from readdate import MyDataset
from torchvision import models
import os
import shutil
from datetime import datetime
import time


def save_checkpoint(state, is_best, name, savefilename):
    filename = os.path.join(r'.\run\{}'.format(savefilename), name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(r'.\run\{}'.format(savefilename), name + '_model_best.pth.tar'))


def main():
    batchsz = 16
    best_acc = 0
    star = time.time()
    use_cuda = torch.cuda.is_available()
    savefilename = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    os.mkdir(r'.\run\{}'.format(savefilename))
    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    train_data = MyDataset('train', transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]))
    test_data = MyDataset('val', transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]))
    train_loader = DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsz)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    # model = ResNet50().to(device)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 45)
    model = model.to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(300):
        begin = time.time()
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(torch.device("cuda")), label.to(torch.device("cuda"))

            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num

        sv_name = '%.5f' % acc#datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        is_best_acc = acc > best_acc
        best_acc = max(best_acc, acc)
        if epoch>200:
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best_acc, sv_name, savefilename)

        end = time.time()
        time_used = (end - begin) / 60
        print(epoch, 'test acc:{},time:{}'.format(acc, time_used))
    finish = time.time()
    time_used2 = (finish - star) / 60
    print("总用时：{}".format(time_used2))

if __name__ == '__main__':
    main()
