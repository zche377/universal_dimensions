# Contents

- [Contents](#contents)
- [Overview](#overview)
- [System requirements](#system-requirements)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
    - [OS requirements](#os-requirements)
    - [Python version](#python-version)
    - [Python dependencies](#python-dependencies)
- [Installation guide](#installation-guide)
  - [Running the demo](#running-the-demo)
  - [Reproducing the results](#reproducing-the-results)
- [Information about the datasets](#information-about-the-datasets)
  - [The Natural Scenes Dataset](#the-natural-scenes-dataset)
- [License](#license)
- [References](#references)


# Overview

In this work, we characterized the universality of hundreds of thousands of representational dimensions from visual neural networks with varied construction. We found that networks with varied architectures and task objectives learn to represent natural images using a shared set of latent dimensions, despite appearing highly distinct at a surface level. Next, by comparing these networks with human brain representations measured with fMRI, we found that the most brain-aligned representations in neural networks are those that are universal and independent of a network’s specific characteristics.

Here, we demonstrate how to:
 - [Install the code and libraries for analyses](#installation-guide;) 
 - [Compute scores and generate figures with a subset of data](#running-the-demo)
 - [Reproduce all results in the manuscipt](#reproducing-results)


# System requirements

## Hardware requirements
<!-- update gpu ram -->
The code requires a standard computer with enough CPU and GPU compute power to support all operations. The scripts for replicating the main results use about ~24 GB GPU RAM at peak but also work with CPU only.


## Software requirements

### OS requirements
The code has been tested on RHEL 9.3.

### Python version
The code has been tested on Python==3.10.14.

### Python dependencies
The following is a list of python libraries to run all scripts, with more detail in ```requirements.txt```:
```
bonner-libraries==0.0.0
huggingface-hub==0.23.4
lmfit==1.3.1
matplotlib==3.9.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
scipy==1.13.1
seaborn==0.13.2
timm==1.0.3
torch==2.3.1
torchdata==0.7.1
torchmetrics==1.4.0.post0
torchvision==0.18.1
umap-learn==0.5.6
xarray==2024.6.0
zenodo-get==1.6.1
python-dotenv==1.0.1
```

# Installation guide

Clone this repository and navigate to the repository folder.
```
git clone https://github.com/zche377/universal_dimensions.git
cd universal_dimensions
```

In the root directory, open ```.env``` and set the paths for where the data, models, intermediate and final results are saved.

Install required packages and activate the environment. (~3 minutes)
```
conda env create -f environment.yml
conda activate universal_dimensions
```

## Running the demo

In this demo, we computed the universality and brain similarity of 3 ResNet-18 models trained for image classification with different random seeds (Schürholt et al., 2022).

Brain similarity was computed with one subject's data from the Natural Scenes Dataset (NSD; Allen et al., 2022). 

For saving time, obtain pre-cached, pre-processed feature maps of the models and NSD data from OSF: (~3 min)

```
python scripts/download_demo_cache.py
```

To compute universality and brain similarity: (~5 min)
```
python scripts/compute_score.py --yielder demo_resnet18_varied_seeds --score universality_index
python scripts/compute_score.py --yielder demo_resnet18_varied_seeds --score brain_similarity --subj 0
```

To organize the results for plotting: (~1 min)
```
python scripts/summarize_results.py --yielder demo_resnet18_varied_seeds --score universality_index
python scripts/summarize_results.py --yielder demo_resnet18_varied_seeds --score brain_similarity --subj 0
```

See ```notebooks/demo.ipynb``` for generating the relevant plotts with these results.

<!-- To generate the UMAP plot which reduces the 10 most universal features from these models' penultimate ReLU layers to 2 dimensions: (~6 min)
```
python scripts/compute_score.py --yielder demo_resnet18_varied_seeds --score pc_umap --ntop 10 --node layer4.0.relu
``` -->

The UMAP plot will be available in ```BONNER_CACHING_HOME / "pc_umap"```

## Reproducing the results

Note, to reproduce the complete results in the manuscript, one needs to first request for permission from the NSD group for data access (See [The Natural Scenes Dataset](#the-natural-scenes-dataset)).

To cache the full NSD data:

```
python scripts/cache_betas.py --roi general
```

Each model set has a corresponding ```yielder``` identifier: 
 - ```resnet18_classification_imagenet1k_varied_seeds```
 - ```classification_imagenet1k_varied_architectures```
 - ```resnet50_imagenet1k_varied_tasks```
 - ```untrained_resnet18_varied_seeds```
 - ```varied_visual_diets```

Each metric or analysis has a corresponding ```score``` identifier: 
 - ```universality_index```
 - ```universality_index --cb``` (between-subject reliability)
 - ```brain_similarity```
 - ```rsa```
 - ```pc_umap```

To cache the model feature maps and compute the results:
```
python scripts/cache_model_features.py --yielder {yielder}
python scripts/compute_score.py --yielder {yielder} --score {score}
```

To compute the scores necessary for Figure 5 in the manuscript:
```
python scripts/compute_score.py --yielder {yielder} --score rsa
python scripts/compute_score.py --yielder {yielder} --score rsa --sortby within_basis_ui --ntop {ntop}
```
for ```ntop``` in ```[1,5,10]```.

To compute the scores necessary for Supplement Figure 3, use option ```--roi default_list```.


Note: to speed up by manually computing scores in parallel, use options ```--sidx``` - the starting index of models among the list -  and ```--nyield``` - the number of models computed. For example:

```
for i in {0..20..4}; do
    python --yielder {yielder} --score {score} --sidx "$i" --nyield 4
done
```

To organize the results for plotting, except for ```pc_umap```:
```
python scripts/summarize_results.py --yielder {yielder} --score {score}
python scripts/summarize_results.py --yielder {yielder} --score rsa --sortby within_basis_ui --ntop {ntop}
```

See ```notebooks/figures.ipynb``` for generating all plots relevant to these scores in the manuscript.

To generate all UMAP plots and stats in the manuscript:
```
python scripts/compute_score.py --score pc_umap --yielder resnet50_imagenet1k_varied_tasks --node layer4.1.relu_2 --sortby within_basis_ui --ntop 100
python scripts/compute_score.py --score pc_umap --yielder resnet50_imagenet1k_varied_tasks --node layer4.1.relu_2 -ntop -100 --sortby within_basis_ui --ntop 100
python scripts/compute_score.py --score pc_umap --yielder untrained_resnet18_varied_seeds --node layer4.0.relu --sortby within_basis_ui --ntop 100
python scripts/compute_score.py --score pc_umap --yielder resnet50_imagenet1k_varied_tasks --node layer4.1.relu_2 --sortby within_basis_ui --ntop 100 --stim all --noplot
python scripts/compute_score.py --score pc_umap --yielder resnet50_imagenet1k_varied_tasks --node layer4.1.relu_2 -ntop -100 --sortby within_basis_ui --ntop 100 --stim all --noplot
python scripts/compute_score.py --score pc_umap --yielder untrained_resnet18_varied_seeds --node layer4.0.relu --sortby within_basis_ui --ntop 100 --stim all --noplot
```

# Information about the datasets

## The Natural Scenes Dataset

The Natural Scenes Dataset  (Allen et al., 2022) can be downloaded [here](https://naturalscenesdataset.org/) after obtaining [permission](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform) from the NSD group for data access.

We utilized the NSD single-trial betas, preprocessed in 1.8-mm volume space and denoised with the GLMdenoise technique (version 3; betas\_fithrf\_GLMdenoiseRR). Subsequently, the betas were transformed to z-scores within each individual scanning session. For all metrics in our analysis, we used the averaged betas across repetitions.

# License

This project is covered under the MIT License.

# References

Schürholt, K., Taskiran, D., Knyazev, B., Giró-i-Nieto, X., & Borth, D. (2022). Model zoos: A dataset of diverse populations of neural network models. Advances in Neural Information Processing Systems, 35, 38134-38148. 

Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J. B., Naselaris, T., & Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience, 25(1), 116–126.

