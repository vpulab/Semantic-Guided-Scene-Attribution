import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from SceneRecognitionCNN import SceneRecognitionCNN
from Libs.Datasets.Places365Dataset import Places365Dataset
from Libs.Utils import utils
import numpy as np
import yaml
import pickle

"""
Evaluation file to obtain all the necessary Scene Recognition
statistics for the attribution method.
  
Fully developed by Alejandro López-Cifuentes

"""

parser = argparse.ArgumentParser(description='Semantic-Aware Scene Recognition Evaluation')
parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path', required=True)


def evaluationDataLoader(dataloader, model, set):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()

    Predictions = list()
    SceneGTLabels = list()

    # Extract batch size
    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['TEST']

    # Enable the extract of semantic occlusions
    sem_oclusions = CONFIG['VALIDATION']['SEM_OCCLUSIONS']
    SampleCounter = 1

    # Start data time
    data_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                semantic_mask = mini_batch['Semantic'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            # Model Forward
            outputSceneLabel = model(RGB_image)

            # Get predictions
            batch_predictions = utils.obtainPredictedClasses(outputSceneLabel)

            # Save Predictions and Ground-Truth
            Predictions.extend(batch_predictions.tolist())
            SceneGTLabels.extend(sceneLabelGT.cpu().numpy().tolist())

            # Analyze Semantic Occlusions
            if sem_oclusions:
                # Mean ImageNet value
                mean = [0.485, 0.456, 0.406]

                # Get Top1 Semantic Labels
                semantic_mask = semantic_mask[:, 2, :, :]

                # Intermediate array to save results
                results_array = np.zeros([CONFIG['DATASET']['N_CLASSES_SEM'], CONFIG['DATASET']['N_CLASSES_SCENE'], batch_size])

                # Original Scene Prediction goes in the first row of the matrix
                results_array[0, :, :] = np.transpose(torch.nn.functional.softmax(outputSceneLabel, dim=1).cpu().numpy())

                # There is no semantic label = 0 (or it is not-annotated label) so we start in label = 1
                for j in range(1, CONFIG['DATASET']['N_CLASSES_SEM']):
                    RGB_image_occluded = RGB_image.clone()

                    # Select areas of the image corresponding to semantic class j
                    indices = (semantic_mask == j)

                    # Change RGB Images in those indices pixels. Original paper puts RGB values to the mean ImageNet Distribution value
                    R = RGB_image_occluded[:, 0, :, :]
                    G = RGB_image_occluded[:, 1, :, :]
                    B = RGB_image_occluded[:, 2, :, :]
                    R[indices] = mean[0]
                    G[indices] = mean[1]
                    B[indices] = mean[2]
                    R = torch.unsqueeze(R, dim=1)
                    G = torch.unsqueeze(G, dim=1)
                    B = torch.unsqueeze(B, dim=1)

                    # Reconstruct again the images
                    RGB_image_occluded = torch.cat((R, G, B), dim=1)

                    # Obtained new predictions with the occluded RGB images
                    outputSceneLabel_occluded = model(RGB_image_occluded)

                    # Save the correspondent results
                    results_array[j, :, :] = np.transpose(torch.nn.functional.softmax(outputSceneLabel_occluded, dim=1).cpu().numpy())

                for k in range(batch_size):
                    # Save each matrix in an indeppendent file
                    # sio.savemat('Occlusion Matrices Results/Mat Files/RGB_matrix_pred_' + str(SampleCounter).zfill(5) + '.mat',
                    #             {'image_' + str(SampleCounter): results_array[:, :, k]})

                    with open(os.path.join(ResultPathMatrices, 'RGB_matrix_pred_' + str(SampleCounter).zfill(5) + '.pkl'), 'wb') as filehandle:
                        # store the data as binary data stream
                        pickle.dump(results_array[:, :, k], filehandle)

                    SampleCounter += 1

            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model.loss(outputSceneLabel, sceneLabelGT)

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            # Update values
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top2.update(prec2.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # Measure batch elapsed time
            batch_time.update(time.time() - start_time)

            # Print information
            if i % CONFIG['VALIDATION']['PRINT_FREQ'] == 0:
                print('Testing {} set batch: [{}/{}] '
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}) '
                      'Loss {loss.val:.3f} (avg: {loss.avg:.3f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Prec@2 {top2.val:.3f} (avg: {top2.avg:.3f}) '
                      'Prec@5 {top5.val:.3f} (avg: {top5.avg:.3f})'.
                      format(set, i, len(dataloader), set, batch_time=batch_time, loss=losses,
                             top1=top1, top2=top2, top5=top5))

        ClassTPDic = {'Top1': ClassTPs_Top1.cpu().numpy(),
                      'Top2': ClassTPs_Top2.cpu().numpy(), 'Top5': ClassTPs_Top5.cpu().numpy()}

        print('Elapsed time for {} set evaluation {time:.3f} seconds'.format(set, time=time.time() - data_time_start))
        print("")

        # Save predictions and Scene GT in pickle files
        with open(os.path.join(ResultPath, set + '_Predictions.pkl'), 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(Predictions, filehandle)

        with open(os.path.join(ResultPath, set + 'SceneGTLabels.pkl'), 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(SceneGTLabels, filehandle)

        return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic


global USE_CUDA, classes, CONFIG

# ----------------------------- #
#         Configuration         #
# ----------------------------- #

# Decode CONFIG file information
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))
USE_CUDA = torch.cuda.is_available()

print('-' * 65)
print("Evaluation starting...")
print('-' * 65)

ResultPath = os.path.join(CONFIG['RESULTS']['OUTPUT_DIR'], CONFIG['DATASET']['NAME'])
ResultPathMatrices = os.path.join(ResultPath, 'Occlusion Matrices')

if not os.path.isdir(ResultPathMatrices):
    os.makedirs(ResultPathMatrices)

# ----------------------------- #
#             Model             #
# ----------------------------- #

print('Evaluating Scene Recognition model.')
print('Selected Scene Recognition architecture: ' + CONFIG['MODEL']['ARCH'])
model = SceneRecognitionCNN(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'])

# Load the trained model
completePath = CONFIG['MODEL']['PATH'] + CONFIG['MODEL']['NAME'] + '.pth.tar'
if os.path.isfile(completePath):
    print("Loading model {} from path {}...".format(CONFIG['MODEL']['NAME'], completePath))
    checkpoint = torch.load(completePath)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded model {} from path {}.".format(CONFIG['MODEL']['NAME'], completePath))
    print("     Epochs {}".format(checkpoint['epoch']))
    print("     Single crop reported precision {}".format(best_prec1))
else:
    print("No checkpoint found at '{}'. Check configuration file MODEL field".format(completePath))
    quit()

# Move Model to GPU an set it to evaluation mode
if USE_CUDA:
    model.cuda()
cudnn.benchmark = USE_CUDA
model.eval()

# Model Parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])


# ----------------------------- #
#            Dataset            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

traindir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])
valdir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])

val_dataset = Places365Dataset(valdir, "val")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'],
                                         shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

classes = val_dataset.classes

# Get Histogram of class samples
ValHist = utils.getHistogramOfClasses(val_loader, classes)


# ----------------------------- #
#         Printing Info         #
# ----------------------------- #

# Print dataset information
print('Dataset loaded!')
print('Dataset Information:')
print('Validation set. Size {}. Batch size {}. Nbatches {}'
      .format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], len(val_loader)))
print('Number of scenes: {}' .format(len(classes)))
print('-' * 65)
print('Computing histogram of scene classes...')
print('-' * 65)
print('Number of params: {}'. format(params))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)


# ----------------------------- #
#          Evaluation           #
# ----------------------------- #

print('Evaluating dataset ...')

# Evaluate model on validation set
val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = evaluationDataLoader(val_loader, model, set='Validation')

# Save Validation Class Accuracy
val_ClassAcc_top1 = (val_ClassTPDic['Top1'] / (ValHist + 0.0001)) * 100

# Print complete evaluation information
print('-' * 65)
print('Evaluation statistics:')
print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
      'Mean Class Accuracy {MCA:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5, MCA=np.mean(val_ClassAcc_top1)))
