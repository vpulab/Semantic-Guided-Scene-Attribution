import numpy as np
import pickle
import os
# import scipy.io as sio
import utils
from PIL import Image
from Libs.Datasets.Places365Dataset import Places365Dataset

# Define dataset
dataset_name = 'places365_standard'

# Paths
dataset_dir = os.path.join('Data', 'Datasets', dataset_name)
StatisticsPath = os.path.join('Attribution Results', dataset_name)
MatricesPath = os.path.join(StatisticsPath, 'Occlusion Matrices')
ResultsPath = os.path.join(StatisticsPath, 'Statistics')

# Intialize Dataset
dataset = Places365Dataset(dataset_dir, "val")

# Files names
dataset_filenames = dataset.filenames

# Scene Classes
scene_classes = dataset.classes
n_scenes = len(scene_classes)

# Semantic Classes
sem_classes = utils.readSemanticClasses(dataset_dir)
n_semantic = len(sem_classes)

# Ground Truth
labels = dataset.labels
labels_index = dataset.labelsindex
labels_index = np.asarray(labels_index)

# Predictions
with open(os.path.join(StatisticsPath, 'Validation_Predictions.pkl'), 'rb') as f:
    predictions = pickle.load(f)
predictions = np.squeeze(np.asarray(predictions))

for scene_index, scene in enumerate(scene_classes):
    print('Extracting statistics from scene {} ({}/{})'.format(scene, scene_index, n_scenes - 1))

    # Create folder for scene
    scene_folder = os.path.join(ResultsPath, scene)

    if not os.path.isdir(scene_folder):
        os.makedirs(scene_folder)

    # Index which detections are predicted as scene index
    Det = np.squeeze(np.argwhere(predictions == scene_index))
    n_det = len(Det)

    # Index those GT frames that are scene index
    GT = np.squeeze(np.argwhere(labels_index == scene_index))
    n_gt = len(GT)

    # Compute True Positives and False Negatives
    tp = np.asarray([x for x in Det if x in GT])
    fn = np.asarray([x for x in GT if x not in Det])

    # Number of correct and non correct predictions
    n_correct_predictions = tp.shape[0]
    n_error_predictions = fn.shape[0]

    # Define matrices
    dec_aciertos = {'Score Deviation': [], 'Sample': [], 'Pred': [], 'GT': []}
    dec_distractors = {'Score Deviation': [], 'Sample': [], 'Pred': [], 'GT': []}
    dec_predictions = {'Score Deviation': [], 'Sample': [], 'Pred': [], 'GT': []}

    sem_hist_gt = []
    sem_hist_pred = []

    # Global statistics for ground truth. Only correct images for Score Deviation Maps
    for sample_GT in GT:
        # Load Matrix of predictions.
        with open(os.path.join(MatricesPath, 'RGB_matrix_pred_' + str(sample_GT+1).zfill(5) + '.pkl'), 'rb') as f:
            sample_mat = pickle.load(f)

        # Read Semantic Image and compute histogram of labels
        sem_path = os.path.join(dataset_dir, 'noisy_annotations_RGB', 'val', labels[sample_GT], dataset_filenames[sample_GT].split('.')[0] + '.png')
        sem = np.asarray(Image.open(sem_path))

        # Top@1 Semantic Labels are encoded in the last channel
        sem = sem[:, :, 2]
        # Histogram
        sem_hist_gt.append(np.histogram(sem, bins=np.arange(n_semantic + 1))[0])

        if sample_GT in tp:
            # GT sample is in True Positives
            # Subtraction of original probability againts the obtained by occluding semantic classes
            dec_aciertos['Score Deviation'].append(sample_mat[0, scene_index] - sample_mat[1:, scene_index])
            dec_aciertos['Sample'].append(sample_GT)
            dec_aciertos['Pred'].append(predictions[sample_GT])
            dec_aciertos['GT'].append(labels_index[sample_GT])
        else:
            # GT Sample is False negative
            # Get which semantic class inhibition gets the max scene value
            row, col = np.where(sample_mat[1:, :] == np.max(sample_mat[1:, :]))

            # If is the same as scene_index is a distractor
            if scene_index in col:
                dec_distractors['Score Deviation'].append(row)
                dec_distractors['Sample'].append(sample_GT)
                dec_distractors['Pred'].append(predictions[sample_GT])
                dec_distractors['GT'].append(labels_index[sample_GT])

    # Global statistics for detections. Correct and non correct images
    for sample_det in Det:
         # Load Matrix of predictions.
        with open(os.path.join(MatricesPath, 'RGB_matrix_pred_' + str(sample_det+1).zfill(5) + '.pkl'), 'rb') as f:
            sample_mat = pickle.load(f)

        # Read Semantic Image and compute histogram of labels
        sem_path = os.path.join(dataset_dir, 'noisy_annotations_RGB', 'val', labels[sample_det], dataset_filenames[sample_det].split('.')[0] + '.png')
        sem = np.asarray(Image.open(sem_path))

        # Top@1 Semantic Labels are encoded in the last channel
        sem = sem[:, :, 2]

        # Histogram
        sem_hist_pred.append(np.histogram(sem, bins=np.arange(n_semantic + 1))[0])

        # Subtraction of original probability against the obtained by occluding semantic classes
        dec_predictions['Score Deviation'].append(sample_mat[0, scene_index] - sample_mat[1:, scene_index])
        dec_predictions['Sample'].append(sample_det)
        dec_predictions['Pred'].append(predictions[sample_det])
        dec_predictions['GT'].append(labels_index[sample_det])

    # Aggregate histograms of semantic masks and obtain probability distributions
    sem_hist_gt = np.sum(np.asarray(sem_hist_gt), axis=0)
    sem_hist_gt = sem_hist_gt / np.sum(sem_hist_gt, axis=0)
    sem_hist_pred = np.sum(np.asarray(sem_hist_pred), axis=0)
    sem_hist_pred = sem_hist_pred / np.sum(sem_hist_pred, axis=0)

    # Save results in scene_folder
    with open(os.path.join(scene_folder, 'dec_aciertos.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dec_aciertos, filehandle)
    with open(os.path.join(scene_folder, 'dec_distractors.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dec_distractors, filehandle)
    with open(os.path.join(scene_folder, 'dec_predictions.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dec_predictions, filehandle)
    with open(os.path.join(scene_folder, 'sem_hist_gt.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(sem_hist_gt, filehandle)
    with open(os.path.join(scene_folder, 'sem_hist_pred.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(sem_hist_pred, filehandle)

