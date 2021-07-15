# *_*coding: utf-8 *_*
# author --liming--

import os
import shutil
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from readdate import MyDataset
from datetime import datetime

os.environ['CUDA_VISION_DEVICES'] = '0'
project_path = os.getcwd()

classes_num = 45  # 21
batchsz = 32
resize = 256
crop_size = 224
epochs = 400  # 100
init_lr = 0.01  # 0.001
momentum = 0.9
log_interval = 10
stop_accuracy = 85
adjust_lr_epoch = 120  # 60

savefilename = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
os.mkdir(r'/content/RS-GPU/run/{}'.format(savefilename))
img_save_path = str('/content/RS-GPU/run/{}/'.format(savefilename))


# 训练函数
def model_train(model, train_data_load, optimizer, loss_func, epoch, log_interval):
    model.train()

    correct = 0
    train_loss = 0
    total = len(train_data_load.dataset)

    for i, (img, label) in enumerate(train_data_load, 0):
        begin = time.time()
        img, label = img.cuda(), label.cuda()

        optimizer.zero_grad()
        outputs = model(img)
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            traind_total = (i + 1) * len(label)
            acc = 100. * correct / traind_total
            end = time.time()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t lr: {}\t Train_Acc: {:.6f}\t Speed: {}'.format(
                epoch,
                i * len(img),
                total,
                100. * i / len(train_data_load),
                loss.data.item(),
                optimizer.param_groups[0]['lr'],
                acc,
                (end - begin)))

            global_train_acc.append(acc)
    filename = os.path.join(r'/content/RS-GPU/run/{}'.format(savefilename),
                                    str(epoch) + str('_{}'.format(correct)) + '.pth')
    torch.save(model.state_dict(), filename)

def model_test(model, test_data_load, epoch, kk):
    model.eval()
    with torch.no_grad():

        correct = 0
        total = len(test_data_load.dataset)

        for i, (img, label) in enumerate(test_data_load):
            img, label = img.cuda(), label.cuda()

            outputs = model(img)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == label).sum()

        acc = correct.item() * 100. / (len(test_data_load.dataset))
        # 记录最佳分类精度
        global best_acc, best_epoch
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        print('Test Set: Accuracy: {}/{}, ({:.6f}%)\nBest_Acc: {}(Epoch:{})'.format(correct, total, acc, best_acc,
                                                                                    best_epoch))
        global_test_acc.append(acc)
        if acc > stop_accuracy:
            filename = os.path.join(r'.\run\{}'.format(savefilename),
                                    str(epoch) + str('_{}'.format(acc)) + '.pth')
            # torch.save(model.state_dict(), str(kk + 1) + '_ResNet34_BestScore_' + str(best_acc) + '.pth')
            torch.save(model.state_dict(), filename)


def show_acc_curv(ratio, kk):
    # 训练准确率曲线的x、y
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc

    # 测试准确率曲线的x、y
    # 每ratio个训练准确率对应一个测试准确率
    test_x = train_x[ratio - 1::ratio]
    test_y = global_test_acc

    plt.title('M ResNet34 ACC')
    plt.plot(train_x, train_y, color='green', label='training accuracy')
    plt.plot(test_x, test_y, color='red', label='testing accuracy')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')
    plt.savefig(img_save_path + 'acc_curv_' + str(kk + 1) + '.jpg')
    # plt.show()


def adjust_learning_rate(optimizer, epoch):
    if (epoch+1) % adjust_lr_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__ == '__main__':
    # 按设定的划分比例,得到随机数据集,分别进行10次训练,然后计算其均值和方差.
    # for k in range(10):

    best_acc = 0
    global_train_acc = []
    global_test_acc = []

    train_data = MyDataset('train', transform=transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.CenterCrop(size=224),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]))
    test_data = MyDataset('val', transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]))
    train_loader = DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True)
    #test_loader = DataLoader(dataset=test_data, batch_size=batchsz, shuffle=True)
    train_data_size = len(train_data)
    #valid_data_size = len(test_data)
    cudnn.benchmark = True

    """加载模型
    # VGG16
    model = models.vgg16(pretrained=True)
    model.classifier[-1].out_features = config.classes_num
    model = model.cuda()
    """
    # ResNet
    model = models.resnet50(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, classes_num)
    model = model.cuda()

    # 优化器与损失
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum)
    loss_func = nn.CrossEntropyLoss().cuda()

    start_time = time.time()

    # 训练
    for epoch in range(0, epochs):
        begin = time.time()
        print('----------------------第%s轮----------------------------' % epoch)
        model_train(model, train_loader, optimizer, loss_func, epoch, log_interval)
        torch.cuda.empty_cache()
        #model_test(model, test_loader, epoch, 1)
        end = time.time()
        time_used = (end - begin) / 60
        print('一轮用时:{}'.format(time_used))
        adjust_learning_rate(optimizer, epoch)

    end_time = time.time()
    print('Train Speed Time:', end_time - start_time)

    # 显示训练和测试曲线
    #ratio = int(train_data_size / batchsz / log_interval)
    #show_acc_curv(ratio, 0)

    torch.save(model.state_dict(), 'resnet_fs_' + str(1) + '.pth')

    print('训练结束, 最佳分类精度为:{}'.format(best_acc))
    print('--------------------------------------------------\n')
