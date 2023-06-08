---
title: 'MOFormer - A Review'
date: 2023-07-31
permalink: /posts/2023/06/MOFormers/
tags:
  - ML4Sci
  - Seminar
  - Uni
  - DeepLearning
  - Transformer
---


I did not do any research on this topic on my own, but only literature review, there is no content directly from me but only summary. If not directly cited the information is either considered common knowledge or from my references at the end, our main source that is reviewed is [1].

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
The structured synthesis of new MOFs with specific properties is also referred to as discovery. Here the chemists have developed a suite of different methods to facilitate this process:
- *Chemist Knowledge* This process just includes an expert that has gathered some experience in this field using intuition and knowledge what the effects of SBUs are or could be to test them in a structured manner. This is not State of the Art right now, but can achieve good results in some 
- *Mass testing* Create different MOFs in practice and test them for the desired properties. This is not scalable and very (time) expensive, so we can only do this for a limited set of materials. 
- *High throughput computational screening* predict the properties of different materials using simulations and approximations, the most accurate simulation is given by the computation of the density functionality theorem, which allows to approximate the properties using a quantum chemical model that simplifies the schrödinger's equations. The approximation given by machine learning also count in this category as we mostly approximate the results of simulations as they are to expensive.
- *postsynthetic modification* Given a MOF that is already close to fulfilling our requirements we can selectively change lignants, nodes, inject impurities or combine different structures to combine some property components. This helps with fine tuning MOFs, but requires an already very good researched property and material field.

A practical synthesis can be done under different conditions, this depends on the components, the MOF, and the expected Material. There are as a short selection solvothermal and non-solvothermal methods (depending on temperature), microwave, electrochemical, mechanochemical or sonochemical synthesis. We do not consider different synthesis methods.
### How are MOFs represented in chemics and what data is there? 
The representation of MOFs diverges in different databases depending on the methods the MOFs were obtained and what they are used for. 

There are mainly to systems that are proven to be exact descriptors of the materials, Systre and Topos Pro, both developed by mathematicians include complicated geometry constructions to achieve a unique representation. These systems are not really useful for machine learning as they are not very interpretable.
Also there is the representation in 3D Coordinate systems, here we again have to extract topology as geometry is mostly not considered in material discovery and deal with duplicates. Although a 3D representation can lead to a better runtime of DFT Simulations as we have a good intital guess. This leads to a representation as topology graphs, up until now these graphs provide the best descriptions for machine learning (see later). Lastly a less concise representation is given by textual descriptors, these should improve searchability and give some information to researchers, as such they aim to improve interpretability. The presentation that we will later use to train our transformers is MOF-IDs: Derived from the SIMLES descriptor MOF-IDs are a textual descriptor of MOFs that include the different building blocks, as well a basic information of the net topology (still we consider this a topology unaware presentation of the net). One example of how the MOF-ID is created is given in *Figure-1*.

![](/images/MOFs/mofid.png)
**Figure-1** Structure of MOF-ID and MOF-Key Identifiers TODO Add source 

## What can we do with machine learning
As already indicated the most accurate data comes from density functionality theorem simulations, so we cannot expect to get a better result than that (we consider DFT data to be the golden standard, even if it is not technically correct). Thus we can only provide horizontal scalability, this means we are able to test a lot more MOFs for different properties and still be efficient, vertical scalability (this means our MOFs can be a lot more complex and we can enforce even complex hierarchical structures and a lot more smaller variations), and interpretability to help researchers find better MOFs contenders, that can be further analyzed in practice.

### State-of-the-Arts right now
There are different sorts of algorithms with different kinds of sophistication and requirements. 
A naive approach is given by conventional machine learning algorithms such as support-vector-machines, decision trees, or other model driven algorithms, here usually property descriptors will be given and only superficial relations between already known Frameworks can be analyzed.  

The most accurate algorithm is given by *crystal graph convolutional neural networks* (CGCNN), they take as input a graph that represents the topology of one cycle of the MOF as shown in *Figure-2*. Then they combine the local effects these links and nodes have with convolutional layers and by increasing these effect windows in a given cascade of convolutional layers we extract the properties we trained for.  This configuration may have problems with the generalization of new building units (as they have to be covered extensively in the training data) and vertical scalability (as the required computation power increases enourmously if we want to train on larger MOF-configurations). This is a reason for Zhonglin et al. to propose a new architecture that aims to solve some of these issues.

![](/images/MOFs/cgcnn.pdf)
**Figure-1** Input preparation of a CGCNN 

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
