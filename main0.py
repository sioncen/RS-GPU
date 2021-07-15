import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from readdate import MyDataset
from torchvision import models
import os
from datetime import datetime
import time
import shutil
import cProfile

def save_checkpoint(state, is_best, name, savefilename):
    filename = os.path.join(r'.\run\{}'.format(savefilename), str(name) + '_checkpoint.pth')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(r'.\run\{}'.format(savefilename), str(name) + '_model_best.pth.tar'))

def main():
    batchsz = 16
    best_acc = 0
    best_epoch = 0
    history = []
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
    test_loader = DataLoader(dataset=test_data, batch_size=batchsz, shuffle=True)
    train_data_size = len(train_data)
    valid_data_size = len(test_data)

    device = torch.device('cuda')
    print(device)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 21)
    model = model.to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(300):
        begin = time.time()
        train_loss, train_acc, valid_loss, valid_acc = 0.0, 0.0, 0.0, 0.0
        if epoch == 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
        elif epoch == 250:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            inputs, labels = x.to(torch.device("cuda")), label.to(torch.device("cuda"))
            outputs = model(inputs)
            loss = criteon(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        model.eval()
        with torch.no_grad():
            for x, label in test_loader:
                inputs, labels = x.to(device), label.to(device)
                outputs = model(inputs)
                loss = criteon(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch
            filename = os.path.join(r'.\run\{}'.format(savefilename), str('best') + '.pth')
            torch.save(model.state_dict(), filename)
        if epoch > 200:
            filename = os.path.join(r'.\run\{}'.format(savefilename),
                                    str(epoch) + str('_{}'.format(avg_valid_acc)) + '.pth')
            torch.save(model.state_dict(), filename)


        end = time.time()
        time_used = (end - begin) / 60
        print("【Epoch: {:03d}】Training Loss: {:.4f}, Accuracy: {:.4f}%, Validation Loss: {:.4f},Accuracy: {:.4f}"
              .format(epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))
        print("time:{},Best Accuracy for validation : {:.4f} at epoch {:03d}".format(time_used, best_acc, best_epoch))

    finish = time.time()
    time_used2 = (finish - star) / 60
    print("总用时：{}".format(time_used2))


if __name__ == '__main__':
    main()
