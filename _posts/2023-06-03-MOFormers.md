---
title: 'MOFormer - A Review'
date: 2012-08-14
permalink: /posts/2023/06/MOFormers/
tags:
  - ML4Sci
  - Seminar
  - Uni
  - DeepLearning
  - Transformer
---

# How can we find Metal-Organic-Frameworks with the desired properties we want using Transformers ? 

Just include a short intro to Material sciences and why there still is a problem
## The problem

Property Prediction and ...
### Why we need new MOFs right now

what are fields mofs are used and why do we need a lot with different properites
## What actually are MOFs?

Typical building blocks and sbus, how are they obtained
### How is discovery done right now? 

Chemist Knowledge, Mass testing, Htihg thorughput compuational screening, postsynthetic modification, Synthesis
### How are MOFs represented in chemics and what data is there? 
Exact Systems: Systre, Topos Pro
3D Coordinate Systems, Topology Graphs, SMILES, MOF Key, MOF Id

## What can we do with machine learning
Short summary of requirements for a machine learning solution
### State-of-the-Arts right now
Conventional Algorithms and CGCNNs (what problems are there and how can we approach them in a logical convention)
### Transformers are expected to bring something to the Table ...
How do we expect them to overcome some of the issues posed

### Traning of our Transformers
Self-Supervised Training Methods

## Evaluation
What the researchers took from it ?
- t-SNE
- Head weights 
- Speedup

### My Commentary

## References 

[1] Zhonglin Cao, Rishikesh Magar, Yuyang Wang, and Amir Barati Farimani. "MOFormer: Self-Supervised Transformer Model for Metal–Organic Framework Property Prediction." Journal of the American Chemical Society, 145(5):2958-2967, 2023. DOI: [10.1021/jacs.2c11420](https://doi.org/10.1021/jacs.2c11420)

[2] Benjamin J. Bucior, Andrew S. Rosen, Maciej Haranczyk, Zhenpeng Yao, Michael E. Ziebel, Omar K. Farha, Joseph T. Hupp, J. Ilja Siepmann, Alán Aspuru-Guzik, and Randall Q. Snurr. "Identification Schemes for Metal–Organic Frameworks To Enable Rapid Search and Cheminformatics Analysis." Crystal Growth & Design, 19(11):6682-6697, 2019. DOI: [10.1021/acs.cgd.9b01050](https://doi.org/10.1021/acs.cgd.9b01050)

[3] Yongchul G. Chung, Emmanuel Haldoupis, Benjamin J. Bucior, Maciej Haranczyk, Seulchan Lee, Hongda Zhang, Konstantinos D. Vogiatzis, Marija Milisavljevic, Sanliang Ling, Jeffrey S. Camp, Ben Slater, J. Ilja Siepmann, David S. Sholl, and Randall Q. Snurr. "Advances, Updates, and Analytics for the Computation-Ready, Experimental Metal–Organic Framework Database: CoRE MOF 2019." Journal of Chemical & Engineering Data, 64(12):5985-5998, 2019. DOI: [10.1021/acs.jced.9b00835](https://doi.org/10.1021/acs.jced.9b00835)

[4] Christopher E. Wilmer, Michael Leaf, Chang Yeon Lee, Omar K. Farha, Brad G. Hauser, Joseph T. Hupp, and Randall Q. Snurr. "Large-scale screening of hypothetical metal--organic frameworks." Nature chemistry, 4(2):83-89, 2012.

[5] Tian Xie and Jeffrey C. Grossman. "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." Physical Review Letters, 120(14), 2018. DOI: [10.1103/physrevlett.120.145301](https://doi.org/10.1103/physrevlett.120.145301)

[6] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention Is All You Need." CoRR, abs/1706.03762, 2017. DOI: [arXiv:1706.03762](http://arxiv.org/abs/1706.03762)

[7] VV Butova, MA Soldatov, AA Guda, KA Lomachenko, and C Lamberti. "Metal-organic frameworks: Structure, properties, methods of synthesis and characterization." Russian Chemical Reviews, 85(3):280-307, 2016.

[8] Andrew S. Rosen, Shaelyn M. Iyer, Debmalya Ray, Zhenpeng Yao, Alan Aspuru-Guzik, Laura Gagliardi, Justin M. Notestein, and Randall Q. Snurr. "Machine learning the quantum-chemical properties of metal--organic frameworks for accelerated materials discovery." Matter, 4(5):1578-1597, 2021.
