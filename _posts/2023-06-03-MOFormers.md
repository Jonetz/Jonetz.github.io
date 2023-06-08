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

# How can we find Metal-Organic-Frameworks with the desired properties using Transformers ? 
Metal-Organic-Frameworks (MOFs) are materials that are highly tuneable to fullfil certain properties in the interaction with gases, this can be useful for different application fields (see later). This provides us with a multiplicity of potential materials from which we have to select the ones that fullfil our needs the best. Since this multiplicity is magnitudes greater than what conventional material science researchers could probe, there is a call for newer more scalable synthesis functionalities, that give researchers a closely preselected set of material combinations that fullfil certain desired properties. 

Material science is a interdisciplinary field using mainly physics and chemics (we consider our work to be at quantum chemics level), with the goal to analyze, describe and syntetizise materials.

### Why we need new Methods to find specialized MOFs right now? 
TODO Add sources 

The main usages of MOFs constitutes the separation and storage of gas atoms, this makes it a good material for air-/pollution filters, drug delivery, water harvesting, Hydrogen Storage, CO2-cleansing, and many more applications. So can for instance MOF elements be added to conventional active coal filters in order to filter certain pollutants more targeted depending on the factory type and pollutant output. Also MOFs became a contender for the more recently intensified search for alternative fuel carriers, as the can store a large amount of hydrogen at subatmospheric pressure levels.

These properties are mainly due to the microsporous nature of MOFs, as they are build as periodic net of metal units and organic connectors they include highly regular structures with a lot of holes in them. A good analogy given by experts is the one of an atomic sponge, where we have to tune the atomic stickiness specified in order to achieve the wanted result. Of course we do not only have to look at the physical size of holes in this net, but rather at a lot of different indicators such as charge distribution and different changes in the fields given some interactions.

We will later see that MOFs are build like legos that only have to be assembled with out of the box parts, for this we have a whole catalogue of building units that we can combine hierarchically and tune very precise. This simplifes the problem from finding a stable configuration to predicting whether a stable configuration provides the properties we want. The synthesis workflow we propose to optimize includes first of all extracting possible MOFs either from academic databases or construct a large number of self-made Frameworks from a building unit catalogue, the we want to filter them with high throughput analysis and only take the few best scoring ones to practice, where they will be generated and iteratively modified to recieve the best possible material for the given requirements.
## What actually are MOFs?
Metal-Organic-Frameworks have two special components, as the name suggests one of them is a metal also referred to as node. This metal-node takes the function of a vertex in our cyclic net and the different nodes are connected with organic linkers, also referred to as lignants. These blocks are also referred to as secondary-building-units (SBUs). Moreover SBUs can include some more complex parts from MOF structures (this is due to the possiblity of structuring MOFs hierachically).

From this description we know the parts that can be used to assemble a basic MOF, but this is still not enough as the same materials often can have different variants, introduced by differences in geometry as well as topology. Depending on their method of synthesis and possible environments the geometry (i.e. how the links and nodes are placed in the space relative to each other) may change, so a MOF can change the geometry for instance if it stores a certain material. If only the geometry differs we generally consider it to be the same MOFs, as long as the topology is homomorphic. Other than that if the topologies differ but the same materials are included we consider this to be another MOF.  The detection and removal of duplicates is a still non trivial task in practice, but we do not consider this a problem we have to deal with from a machine learning perspective.  
### How is discovery done right now? 
The structured syntesis of new MOFs with specific properties is also referred to as discovery. Here the chemists have developed a suite of different methods to facilitate this process:
-Chemist Knowledge
-Mass testing
- High thorughput compuational screening
- postsynthetic modification

Write about how the actual synthesis is done here...
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
