# Visualizing the Effect of Semantic Classes in the Attribution of Scene Recognition Models
Official Pytorch Implementation of [Visualizing the Effect of Semantic Classes in the Attribution of Scene Recognition Models](https://link.springer.com/chapter/10.1007%2F978-3-030-68796-0_9) by Alejandro López-Cifuentes, Marcos Escudero-Viñolo, Andrija Gajić and Jesús Bescós (ICPR 2020 EDL-AI Workshop).

Project Website with supplementary material: [Visualizing the Effect of Semantic Categories in the Attribution of Scene Recognition Models](http://www-vpu.eps.uam.es/publications/SemanticEffectSceneRecognition/)

## Summary
Our method inhibits specific image areas according to their assigned semantic label.Perturbations are link up with a semantic meaning and a complete attribution map is obtained for all image pixels. In addition, we propose a particularization of the proposed method to the scene recognition task which, differently than image classification, requires multi-focus attribution models. The proposed semantic-guided attribution method enables us to delve deeper into scene recognition interpretability by obtaining for each scene class the sets of relevant, irrelevant and distracting semantic labels.

<p align="center">
	<img alt="Method" src="/Docs/Method.png">
</p>

## Setup

### Requirements
The repository has been tested in the following software versions.
 - Ubuntu 16.04
 - Python 3.6
 - Anaconda 4.9
 - PyTorch 1.7
 - CUDA 10.1
 
 **Important**: Support for different requirements than the ones stated above will not be provided in the issues section, specially for those regarding lower versions of libraries or CUDA.
 
 ### Clone Repository
Clone repository running the following command:

	$ git clone https://github.com/vpulab/Semantic-Guided-Scene-Attribution.git

### Anaconda Enviroment
To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:

    $ conda env create -f Config/Conda_Env.yml
    $ conda activate SemGuided-Attribution
 

### Model Zoo and Dataset

In order to obtain the results presented in the paper we provide one Scene Recognition Model trained over Places-365 for the extraction of the statistics. The model should be placed by default on `Data/Model Zoo/places365_standard/` and can be downloaded from the following link:

 - [Places 365 ResNet-18](http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/Model_Zoo/Places_365/RGB_ResNet18_Places.pth.tar)

Places-365 validation set must be completly downloaded before extracting any statistics. We provide the following script to automatically download the images from the original author's website and the needed semantic segmentations. To download and extract the dataset run:

    $ ./Scripts/download_Places365.sh [PATH]
    
where `[PATH]` is where you want to save dataset. By default the dataset must be placed in folder `Data/Datasets/places365_standard` so run:

    $ ./Scripts/download_Places365.sh ./Data/Datasets/places365_standard
    
## Results

### Scene Recognition Analisys

In order to evaluate all the dataset for the Scene Recognition task and extract the initial statistics run:

    $ python evaluation.py --ConfigPath "Config/config_Places365.yaml"
  
Any path related to the model or the dataset, as well as inference parameters, can be freely changed in the configuration file. Feel free to adapt it to you computer requirements.

This script will save the different Scene Recognition results when inhibiting different image areas according to the semantic segmentation. All the different probability distributions will are saved by default in `Attribution Results/places365_standard/Occlusion Matrices`.
Scene Recognition results in `evaluation.py` should be:
|       Model    | Loss  | Top@1 | Top@2 | Top@5 | Mean Class Accuracy
|--         | :--:    | :--:  | :--:| :--: | :--:|
| ResNet-50 |    1.76    | 53.69| 68.73 | 83.77 | 53.69

Then run

    $ python store_statistics.py

to process all the probability distributions extracted before to obtain Score Deviations for the validation set. Results will be saved in `Attribution Results/places365_standard/Statistics/`
 
### Single Image Score Deviation Maps
Once the evaluation has finished, to obtain Score Deviation Maps for all the images run:

    $ python obtainSDM.py

The Score Deviation Maps are saved by default in `Attribution Results/places365_standard/Score Deviation Maps`.   For a given class, Score Deviation Maps will be divided into correct predicted images and all images predicted as the class.
You should get similar results to the ones in the paper (colormaps may slightly differ):
<p align="center">
	<img alt="SDM" src="/Docs/Results SDM.png">
</p>

### Scene Recogntion Per Class Statistics
To obtain boxplot and wordclouds on a per scene basis run:

    $ python sceneWordClouds.py
    
You should get similar results to the ones in the paper:  
<p align="center">
	<img alt="Results Bedroom" src="/Docs/Results Bedroom.png">
</p>

### Additional Information
Any other model different than the one provided can be used straight away, just plug your model definition and model weigths and repetea all the process changing the configuration file.

## Citation
If you find this code and work useful, please consider citing:
```
@InProceedings{Lopez2020Visualizing,
  author="L{\'o}pez-Cifuentes, Alejandro and Escudero-Vi{\~{n}}olo, Marcos and Gaji{\'{c}}, Andrija and Besc{\'o}s, Jes{\'u}s",
  title="Visualizing the Effect of Semantic Classes in the Attribution of Scene Recognition Models",
  booktitle="Pattern Recognition. ICPR International Workshops and Challenges",
  year="2021",
  pages="115--129",
}
```
## Acknowledgment
This study has been partially supported by the Spanish Government through its TEC2017-88169-R MobiNetVideo project.

<p align="center">
	<img alt="LogoMinisterio" src="/Docs/LogoMinisterio.png">
</p>