from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
# from imgaug import augmenters as iaa


class Places365Dataset(Dataset):
    """Class for Places 365 dataset."""

    def __init__(self, root_dir, set):
        """
        Initialize the dataset. Read scene categories, get number of classes, create filename and ground-truth labels
        lists, create ImAug and PyTorch transformations

        :param root_dir: Root directory to the dataset
        :param set: Dataset set: Training or Validation
        """
        # Extract main path and set (Train or Val).
        self.image_dir = root_dir
        self.set = set

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(root_dir, "categories_places365.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                line = line.split()[0]
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Check if there a class with a subclass inside (outdoor, indoor)
                if len(split_indices) > 2:
                    line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]

                self.classes.append(line[split_indices[1] + 1:])

        # Get number of classes
        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        filenames_file = os.path.join(root_dir, (set + ".txt"))

        # Fill filenames list and ground-truth labels list
        with open(filenames_file) as class_file:
            for line in class_file:
                # if random.random() > 0.6 or (self.set is "val"):
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Obtain name and label
                name = line[split_indices[1] + 1:-1]
                label = line[split_indices[0] + 1: split_indices[1]]

                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

        # Control Statements for data loading
        assert len(self.filenames) == len(self.labels)

        # ----------------------------- #
        #    Pytorch Transformations    #
        # ----------------------------- #
        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.outputSize = 224

        # Train Set Transformation
        self.train_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.STD)
        ])
        self.train_transforms_scores = transforms.ToTensor()

        self.train_transforms_sem = transforms.Lambda(lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))

        # Transformations for validation set
        self.val_transforms_img = transforms.Compose([
            transforms.CenterCrop(self.outputSize),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.STD)
        ])
        self.val_transforms_sem = transforms.Compose([
            transforms.CenterCrop(self.outputSize),
            transforms.Lambda(lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))
        ])

    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Function to get a sample from the dataset. First both RGB and Semantic images are read in PIL format. Then
        transformations are applied from PIL to Numpy arrays to Tensors.

        For regular usage:
            - Images should be outputed with dimensions (3, W, H)
            - Semantic Images should be outputed with dimensions (1, W, H)

        :param idx: Index
        :return: Dictionary containing {RGB image, semantic segmentation mask, scene category index}
        """

        # Get RGB image path and load it
        img_name = os.path.join(self.image_dir, self.set, self.labels[idx], self.filenames[idx])
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        filename_sem = os.path.join(self.image_dir, 'noisy_annotations_RGB', self.set, self.labels[idx], self.filenames[idx].split('.')[0] + '.png')
        sem = Image.open(filename_sem)

        # Apply transformations depending on the set (train, val)
        if self.set is "train":
            # # Extract Random Crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.outputSize, self.outputSize))

            # Apply Random Crop parameters
            img = TF.crop(img, i, j, h, w)
            sem = TF.crop(sem, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                sem = TF.hflip(sem)

            # Apply not random transforms. To tensor and normalization for RGB. To tensor for semantic segmentation.
            img = self.train_transforms_img(img)
            sem = self.train_transforms_sem(sem)
        else:
            img = self.val_transforms_img(img)
            sem = self.val_transforms_sem(sem)

        # Create dictionary
        self.sample = {'Image': img, 'Semantic': sem, 'Scene Index': self.classes.index(self.labels[idx])}

        return self.sample
