import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
import os


"""
Several and general utils file.

Fully developed by Alejandro López-Cifuentes
"""


def readSemanticClasses(DatasetGlobalPath):
    # Semantic Classes
    sem_classes = list()
    with open(os.path.join(DatasetGlobalPath, "objectInfo150.txt")) as f:
        for i, line in enumerate(f):
            if i > 0:
                name = line.split()
                if len(name) == 2:
                    name = name[1]
                else:
                    name = name[1] + ' ' + name[2]
                sem_classes.append(name)

    return sem_classes


def unNormalizeImage(image, mean=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]):
    """
    Unnormalizes a numpy array given mean and STD
    :param image: Image to unormalize
    :param mean: Mean
    :param STD: Standard Deviation
    :return: Unnormalize image
    """
    for i in range(0, image.shape[0]):
        image[i, :, :] = (image[i, :, :] * STD[i]) + mean[i]
    return image


def plotTensorImage(image, label="GT Label"):
    """
    Function to plot a PyTorch Tensor image
    :param image: Image to display in Tensor format
    :param mean: Mean of the normalization
    :param STD: Standard Deviation of the normalization
    :param label: (Optional) Ground-truth label
    :return:
    """
    # Convert PyTorch Tensor to Numpy array
    npimg = image.numpy()
    # # Unnormalize image
    unNormalizeImage(npimg)
    # Change from (chns, rows, cols) to (rows, cols, chns)
    npimg = np.transpose(npimg, (1, 2, 0))

    # Convert to RGB if gray-scale
    if npimg.shape[2] is 1:
        rgbArray = np.zeros((npimg.shape[0], npimg.shape[1], 3), 'float32')
        rgbArray[:, :, 0] = npimg[:, :, 0]
        rgbArray[:, :, 1] = npimg[:, :, 0]
        rgbArray[:, :, 2] = npimg[:, :, 0]
        npimg = rgbArray

    # Display image
    plt.figure()
    plt.imshow(npimg)
    plt.title(label)


class AverageMeter(object):
    """
    Class to store instant values, accumulated and average of measures
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += np.power(val,2) * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.sqrt((self.sum2 / self.count) - np.power(self.avg, 2))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves check point
    :param state: Dictionary to save. Constains models state_dictionary
    :param is_best: Boolean variable to check if is the best model
    :param filename: Saving filename
    :return:
    """
    torch.save(state, 'Files/' + filename + '_latest.pth.tar')
    if is_best:
        print('Best model updated.')
        shutil.copyfile('Files/' + filename + '_latest.pth.tar', 'Files/' + filename + '_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy between output and target.
    :param output: output vector from the network
    :param target: ground-truth
    :param topk: Top-k results desired, i.e. top1, top5, top10
    :return: vector with accuracy values
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # output = output.long()
    # target = target.long()

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def getclassAccuracy(output, target, nclasses, topk=(1,)):
    """
    Computes the top-k accuracy between output and target and aggregates it by class
    :param output: output vector from the network
    :param target: ground-truth
    :param nclasses: nclasses in the problem
    :param topk: Top-k results desired, i.e. top1, top2, top5
    :return: topk vectors aggregated by class
    """
    maxk = max(topk)

    score, label_index = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    correct = label_index.eq(torch.unsqueeze(target, 1))

    ClassAccuracyRes = []
    for k in topk:
        ClassAccuracy = torch.zeros([1, nclasses], dtype=torch.uint8).cuda()
        correct_k = correct[:, :k].sum(1)
        for n in range(target.shape[0]):
            ClassAccuracy[0, target[n]] += correct_k[n].byte()
        ClassAccuracyRes.append(ClassAccuracy)

    return ClassAccuracyRes


def getHistogramOfClasses(dataloader, classes):
    """
    Computes the histogram of classes for the given dataloader
    :param dataloader: Pytorch dataloader to compute the histogram
    :param classes: Classes names
    :return: Histogram of classes
    """
    ClassesHist = [0] * len(classes)
    images = dataloader.dataset.labelsindex
    for item in images:
        ClassesHist[item] += 1
    ClassesHist = np.asarray(ClassesHist)

    return ClassesHist


def obtainPredictedClasses(outputSceneLabel):
    """
    Fucntion to obtain the indices for the 10 most-scored scene labels
    :param outputSceneLabel: Tensor obtain from the network
    :return: numpy array 1x10 with scene labels indices
    """

    # Obtain the predicted class by obtaining the maximum score.
    score, label_index = outputSceneLabel.topk(k=1, dim=1, largest=True, sorted=True)
    label_index = np.squeeze(label_index.cpu().numpy())

    return label_index

