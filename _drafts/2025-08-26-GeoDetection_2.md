---
title: 'Challenges & Methods in Detecting Objects from UAV-Data (2)'
date: 2025-08-11
permalink: /posts/2025/08/Detecting-Objects/
tags:
  - Environmental Science
  - Solar
  - Detection
  - Segmentation
  - Uni
  - DeepLearning
--- 

This article is about a project I did at the [ISP](https://www.ki.uni-stuttgart.de/departments/isp/) at the University of Stuttgart. 
The second part is examining the transferability of our framework to other objects. We choose solar panel detection as such an other class, as there is a concrete demand to evaluate the development stages of renewable energies. Secondly solar panels exhibit a homogenous structure and are clearly visible and seperable, so an easy candidate for inference. In this case the goal is not to achieve the best inference results, but to test the transferability of our network.

# Solar Detection
We can conceptually separate two cases of solar panels: Rooftop PV and Solar Parks. (For our case we do not distinguish between PV and Solar.) Since there is no data to efficiently segment the cases early in the predictions we need to make a model that covers both of these areas. 

## Using publically available training data
Since we already have the training and inference logic from the TreeDetection, we mainly lack datasets to train a new model. Here we use 4 major datasets and adapt their labeling to COCO-Format to train with them, the datasets are:
- **MultiRes Dataset** by Hou et al. available at [Zenodo](zenodo.org/records/5171712) and the corresponding peer reviewed [article in Copernicus](https://essd.copernicus.org/articles/13/5389/2021/).
- **Crowdsource Dataset** by Kasmi et al. available at [Zenodo](https://zenodo.org/records/7358126) and the corresponding peer reviewed [article in nature](https://www.nature.com/articles/s41597-023-01951-4).
- **SolarDK Dataset** by Khomiakov et al. available at the OSF-Storgae website [https://osf.io/aj539/](https://osf.io/aj539/) and [their paper available at Arxiv](https://arxiv.org/pdf/2212.01260).
"OSFstorage Dataset"
- **Germany Dataset** by Clark et Pacifici with images available [here](https://resources.maxar.com/product-samples/15-cm-hd-and-30-cm-view-ready-solar-panels-germany) by Maxar and annotations in a figshare [here](https://figshare.com/ndownloader/files/39255599). The corresponding peer reviewed [article in Nature](https://www.nature.com/articles/s41597-023-02539-8).

These Datasets vary in different qualities and instances there are roof-pv, water-pv, and ground-mounted-pv. Together with a bit of filtering we have 22.407 images with 71043 training instances and 8622 test instances.

Evaluating the training we can see the precision and recieve the results as:
| AP      | AP50    | AP75     |
| :---    |   :---  |     :--- |
| 63.3    | 84.4    | 74.0     |

applying inference to the trained model we get the following images:
" TODO ADD Images Good + Bad "

So we can see the Precision is okay, but on a landwide scale there are several problems, especially since, many complicated structures especially in big cities are present in the real life inference data, that are not represented adequately in the training data. This leads to high confidence false positives. In order to mitigate this we have to further annotate real data. 

## Finetuning with our own data sets

## Adding features to enrich the data

## Results 