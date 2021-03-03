import numpy as np
import pickle
import os
# import scipy.io as sio
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from Libs.Datasets.Places365Dataset import Places365Dataset
from wordcloud import (WordCloud, get_single_color_func)
from PIL import ImageColor
import utils


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        # self.word_to_color = {word: color
        #                       for (color, words) in color_to_words.items()
        #                       for word in words}
        self.word_to_color = color_to_words
        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


# Define dataset
dataset_name = 'places365_standard'

# Dataset dir
DatasetGlobalPath = os.path.join('Data', 'Datasets')
dataset_dir = os.path.join(DatasetGlobalPath, 'places365_standard')

# Path from saved statistics
statistics_path = os.path.join('Attribution Results', dataset_name, 'Statistics')

# Path to save results
results_path = os.path.join('Attribution Results', dataset_name, 'WordClouds')

if not os.path.isdir(results_path):
    os.makedirs(results_path)

# Intialize Dataset
dataset = Places365Dataset(dataset_dir, "val")

# Scene Classes
scene_classes = dataset.classes
n_scenes = len(scene_classes)

# Semantic Classes
sem_classes = utils.readSemanticClasses(dataset_dir)
n_semantic = len(sem_classes)

# Mask for WordCloud
mask = np.array(Image.open(os.path.join('Data', "mask.png")))

for scene_index, scene in enumerate(scene_classes):
    print('Extracting wordclouds from scene {} ({}/{})'.format(scene, scene_index, n_scenes - 1))

    # Create results folder for the scene
    scene_folder = os.path.join(results_path, scene)

    if not os.path.isdir(scene_folder):
        os.makedirs(scene_folder)

    # CORRECT PREDICTIONS
    # Load statistics
    scene_statistics_path = os.path.join(statistics_path, scene)

    with open(os.path.join(scene_statistics_path, 'dec_aciertos.pkl'), 'rb') as f:
        Deviations = pickle.load(f)

    # Get relevant semantics
    th = 0.01

    # Convert list to numpy
    ScoreDeviations = np.asarray(Deviations['Score Deviation'])

    # Threshold those semantic classes with a higher deviation than th
    selected_semantics = np.mean(ScoreDeviations, axis=0) > th

    selected_semantics_names = np.asarray(sem_classes)[selected_semantics].tolist()
    selected_semantics_names.insert(0, '')

    plt.figure()
    plt.boxplot(ScoreDeviations[:, selected_semantics])
    plt.xticks(np.arange(np.sum(selected_semantics) + 1), selected_semantics_names, rotation=45)
    plt.savefig(os.path.join(scene_folder, 'Relevant Semantics GT ' + str(th) + '.jpg'), bbox_inches='tight', dpi=300)
    plt.close()

    # Wordcloud
    selected_semantics = np.mean(ScoreDeviations, axis=0) > 0
    selected_semantics_names = np.asarray(sem_classes)[selected_semantics].tolist()

    # Get mu
    mu = np.mean(ScoreDeviations[:, selected_semantics], axis=0)
    numOccurrences = np.ceil(mu / np.min(mu))
    numOccurrences[numOccurrences < np.percentile(numOccurrences, 45)] = np.percentile(numOccurrences, 45)

    with open(os.path.join('Data', 'Colormaps', 'Distinguishable_colors_colormap.pkl'), 'rb') as f:
        ColorMap = pickle.load(f)
        ColorMap = ColorMap[1:]

    temp_dict = {}
    color_to_words = {}
    for i, sem in enumerate(sem_classes):
        if sem in selected_semantics_names:
            temp_dict[sem] = numOccurrences[selected_semantics_names.index(sem)]
        else:
            temp_dict[sem] = 0
        color_to_words[sem] = tuple([int(ColorMap[i, 0] * 255), int(ColorMap[i, 1] * 255), int(ColorMap[i, 2] * 255)])

    grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color='grey')

    wc = WordCloud(mask=mask, font_path=os.path.join('Data', 'Font', 'ArialUnicodeMS.ttf'), background_color="white",
                   prefer_horizontal=1, margin=15, scale=5, relative_scaling=0.35)
    wc.generate_from_frequencies(temp_dict)
    wc.recolor(color_func=grouped_color_func)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(scene_folder, 'Relevant Semantics GT WordCloud.png'), bbox_inches='tight', dpi=600)
    plt.close()

    # Semantic Distributions for Correct Predictions
    with open(os.path.join(scene_statistics_path, 'sem_hist_gt.pkl'), 'rb') as f:
        Sem_Dist_GT = pickle.load(f)

    selected_semantics = Sem_Dist_GT > th
    selected_semantics_names = np.asarray(sem_classes)[selected_semantics].tolist()

    plt.figure()
    plt.bar(np.arange(np.sum(selected_semantics)), Sem_Dist_GT[selected_semantics])
    plt.xticks(np.arange(np.sum(selected_semantics)), selected_semantics_names, rotation=45)
    plt.savefig(os.path.join(scene_folder, 'Semantics Distribution ' + str(th) + '.jpg'), bbox_inches='tight', dpi=300)
    plt.close()