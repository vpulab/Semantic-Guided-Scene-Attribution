import numpy as np
import pickle
import os
# import scipy.io as sio
import utils
from PIL import Image
from Libs.Datasets.Places365Dataset import Places365Dataset
import matplotlib.pyplot as plt
import matplotlib


# Define dataset
dataset_name = 'places365_standard'

# Paths
DatasetGlobalPath = os.path.join('Data', 'Datasets')
dataset_dir = os.path.join(DatasetGlobalPath, dataset_name)
StatsPath = os.path.join('Attribution Results', dataset_name, 'Statistics')
ResultsPath = os.path.join('Attribution Results', dataset_name, 'Score Deviation Maps')

# Intialize Dataset
dataset = Places365Dataset(dataset_dir, "val")

# Files names
dataset_filenames = dataset.filenames

# Labels
dataset_labels = dataset.labels

# Scene Classes
scene_classes = dataset.classes
n_scenes = len(scene_classes)

# Semantic Classes
sem_classes = utils.readSemanticClasses(dataset_dir)
n_semantic = len(sem_classes)

# Define Colormap
n_levels = 255
plasma_cm = matplotlib.cm.get_cmap('plasma', n_levels)

for scene_index, scene in enumerate(scene_classes):
    print('Extracting SDMs for scene {} ({}/{})'.format(scene, scene_index, n_scenes - 1))

    # Scene folder
    scene_folder = os.path.join(ResultsPath, scene)
    # Statistics folder
    scene_stats_folder = os.path.join(StatsPath, scene)

    # SCORE DEVIATION MAPS FOR CORRECT PREDICTIONS
    maps_folder_correct = os.path.join(scene_folder, 'Correct Predictions')
    if not os.path.isdir(maps_folder_correct):
        os.makedirs(maps_folder_correct)

    with open(os.path.join(scene_stats_folder, 'dec_aciertos.pkl'), 'rb') as f:
        Deviations = pickle.load(f)

    for Dev, Sample, Pred, GT in zip(Deviations['Score Deviation'], Deviations['Sample'], Deviations['Pred'], Deviations['GT']):

        # Load Semantic Image
        sem_path = os.path.join(dataset_dir, 'noisy_annotations_RGB', 'val', scene, dataset_filenames[Sample].split('.')[0] + '.png')
        sem = np.asarray(Image.open(sem_path))

        # Top@1 Semantic Labels are encoded in the last channel
        sem = sem[:, :, 2]

        # Extract unique labels in the image
        labels = np.unique(sem)

        map = np.zeros_like(sem)

        for label in labels:
            map = np.where(sem == label, Dev[label], map)

        plt.figure()
        plt.imshow(np.round(map * n_levels), cmap='plasma')
        plt.axis('off')
        plt.savefig(os.path.join(maps_folder_correct, dataset_filenames[Sample]), bbox_inches='tight', dpi=300)
        plt.close()

    # SCORE DEVIATION MAPS FOR ALL THE PREDICTIONS
    maps_folder_all = os.path.join(scene_folder, 'All Predictions')
    if not os.path.isdir(maps_folder_all):
        os.makedirs(maps_folder_all)

    with open(os.path.join(scene_stats_folder, 'dec_predictions.pkl'), 'rb') as f:
        Deviations = pickle.load(f)

    for Dev, Sample, Pred, GT in zip(Deviations['Score Deviation'], Deviations['Sample'], Deviations['Pred'], Deviations['GT']):

        # Load Semantic Image
        sem_path = os.path.join(dataset_dir, 'noisy_annotations_RGB', 'val', dataset_labels[Sample], dataset_filenames[Sample].split('.')[0] + '.png')
        sem = np.asarray(Image.open(sem_path))

        # Top@1 Semantic Labels are encoded in the last channel
        sem = sem[:, :, 2]

        # Extract unique labels in the image
        labels = np.unique(sem)

        map = np.zeros_like(sem)

        for label in labels:
            map = np.where(sem == label, Dev[label], map)

        plt.figure()
        plt.imshow(np.round(map * n_levels), cmap='plasma')
        plt.axis('off')
        plt.savefig(os.path.join(maps_folder_all, dataset_filenames[Sample]), bbox_inches='tight', dpi=300)
        plt.close()
