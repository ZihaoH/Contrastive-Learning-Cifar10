from torchvision import datasets, transforms
import numpy as np
import torch
from models import PyramidNet



def kNN(epoch, net, trainloader, testloader, feat_dim, K, ):
    net.eval()
    total = 0
    trainsize = trainloader.dataset.__len__()
    testsize = testloader.dataset.__len__()

    trainFeatures = torch.rand(trainsize,feat_dim, ).cuda()
    trainLabels = torch.LongTensor([0]*trainsize).cuda()

    C = 10
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batchSize = inputs.size(0)
            inputs = inputs.cuda()
            targets = targets.cuda()
            trainLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = targets.data

            features = net.extract_feat(inputs)
            trainFeatures[ batch_idx * batchSize:batch_idx * batchSize + batchSize,:] = features.data

    top1 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(batchSize * K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.cuda()
            inputs = inputs.cuda()
            batchSize = inputs.size(0)
            features = net.extract_feat(inputs)

            m = features.size(0)
            n = trainFeatures.size(0)
            xx = (features ** 2).sum(dim=1, keepdim=True).expand(m, n)
            yy = (trainFeatures ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)
            dist = xx + yy - 2 * features.matmul(trainFeatures.transpose(0, 1))

            _, yi = dist.topk(K, dim=1, largest=False, sorted=True)

            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            retrieval_one_hot = retrieval_one_hot.view(batchSize,K,C)
            res = torch.sum(retrieval_one_hot,dim=1)
            res = torch.argmax(res,dim=1)
            correct = torch.sum(targets==res)


            top1 = top1 + correct.item()

            total += targets.size(0)
    #         print('Test [{}/{}]\t'
    #               'Top1: {:.2f} '.format(
    #             total, testsize, top1 * 100. / total))
    # print(top1 * 100. / total)

    return top1 * 100. / total


if __name__ == "__main__":
    ckpt='simclr'

    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root="../../data", transform=transform_all, train=True)
    test_dataset = datasets.CIFAR10(root="../../data", transform=transform_all, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False)

    model = PyramidNet(dataset='cifar10', depth=50, alpha=200, num_classes=10,
                       bottleneck=True, ).cuda()
    if ckpt=='moco':
        checkpoint = torch.load('../../ckeckpoint/MoCo/MoCo_PyramidNet_4/PyramidNet_4_last.pth.tar',map_location=torch.device('cuda', 0))
        state_dict = checkpoint['model_state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('encoder_q.'):
                if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
            del state_dict[k]
    elif ckpt=='simclr':
        checkpoint = torch.load('../../ckeckpoint/SimCLR/SimCLR_PyramidNet_2/PyramidNet_2_last.pth.tar', map_location=torch.device('cuda', 0))
        state_dict = checkpoint['model_state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
    elif ckpt=='supervised':
        checkpoint = torch.load('../../ckeckpoint/cifar10/MoCo_PyramidNet_5/PyramidNet_5_last.pth.tar', map_location=torch.device('cuda', 0))
        state_dict = checkpoint['model_state_dict']
    log = model.load_state_dict(state_dict, strict=False)

    acc = kNN(epoch=1, net=model, trainloader=train_loader, testloader=test_loader, feat_dim=864, K=7)
    print(acc)