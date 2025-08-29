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
We can conceptually separate two cases of solar panels: Rooftop PV and Solar Parks. (For our case we do not distinguish between PV and Solar.) Since there is no data to efficiently segment the cases early in the predictions we need to make a model that covers both of these areas. In this area we decided to not start with 

## Using publically available training data

## Finetuning with our own data set

## Results 