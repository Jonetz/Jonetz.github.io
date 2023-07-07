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

<div style="text-align:center">
    <a href="https://de.wikipedia.org/wiki/MOF-5#/media/Datei:MOF-5.png">
        <img src="/images/MOFs/mof5.png" alt="MOF5" style="width:30%;height:30%">
    </a>
    <br>
    <i>Figure 1: Periodic elements from MOF5, pores are illustrated by spheres (via wikipedia from Tony Boehle)</i>
</div>

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

<div style="text-align:center">
    <a href="https://pubs.acs.org/doi/10.1021/acs.cgd.9b01050">
        <img src="/images/MOFs/mofid.png" alt="MOFID" style="width:80%;height:80%">
    </a>
    <br>
    <i>Figure 1: Structure of MOF-ID and MOF-Key Identifiers, taken from [2]</i>
</div>

## What can we do with machine learning
As already indicated the most accurate data comes from density functionality theorem simulations, so we cannot expect to get a better result than that (we consider DFT data to be the golden standard, even if it is not technically correct). Thus we can only provide horizontal scalability, this means we are able to test a lot more MOFs for different properties and still be efficient, vertical scalability (this means our MOFs can be a lot more complex and we can enforce even complex hierarchical structures and a lot more smaller variations), and interpretability to help researchers find better MOFs contenders, that can be further analyzed in practice.

### State-of-the-Arts right now
There are different sorts of algorithms with different kinds of sophistication and requirements. 
A naive approach is given by conventional machine learning algorithms such as support-vector-machines, decision trees, or other model driven algorithms, here usually property descriptors will be given and only superficial relations between already known Frameworks can be analyzed.  

The most accurate algorithm is given by *crystal graph convolutional neural networks* (CGCNN), they take as input a graph that represents the topology of one cycle of the MOF as shown in *Figure-2*. Then they combine the local effects these links and nodes have with convolutional layers and by increasing these effect windows in a given cascade of convolutional layers we extract the properties we trained for.  This configuration may have problems with the generalization of new building units (as they have to be covered extensively in the training data) and vertical scalability (as the required computation power increases enourmously if we want to train on larger MOF-configurations). This is a reason for Zhonglin et al. to propose a new architecture that aims to solve some of these issues.

<div style="text-align:center">
    <a href="https://doi.org/10.1103/physrevlett.120.145301" >
        <img src="/images/MOFs/cgcnn.png" alt="CGCNN" style="width:70%;height:70%">
    </a>
    <br>
    <i>Figure 3: Input + Architecture of a CGCNN (from [5])</i>
</div>

### The Transformer architecture
Now finally we come to the new paper that we review. Zhonglin et al. are using transformers for material property prediction. To accomplish this, the authors utilize the MOFID format, which provides string representations that can be naturally tokenized. These tokenzied strings provide an input instance, that includes information about the secondary building units (as arguably the most important data for material behaviour), as well as some hints on the topology of the network. These strings will be tokenized into single tokens and transformed into an embedded vector representing this token. After this attention maps are computed.
Attention Maps can either refer to self-attention or to cross-attention, both are used in transformer models. Self-attention enables us to capture relations between the instance and cross-attention enables us to relate those semantics to other instances. With employing both types of attention we are enabled to relate the internal structure of the input to possible output instances. A attention head will compute the degree of relevance between an input or output token and itself or other tokens at the given position.
Generally multiple attention heads are used and their output combined and normalized (in our case it will be 8 heads).
The most simplified explanation of what a transformer does is: given a part of the input, at which other parts do you have to look to understand the semantics of this part in the overall instance?.
These building blocks will be followed by a shallow network to relate these semantic relations to a desired material property (for instance: C02 adsorption).

<div style="text-align:center">
    <a href="">
        <img src="/images/MOFs/transformer.png" alt="TRANS" style="width:70%;height:70%">
    </a>
    <br>
    <i>Figure 3: The MOFormer architecture proposed in [1].</i>
</div>

Transformers as such provide very good results, given that certain criteria are fullfilled. Firstly they need a lot more training data then deep neural networks or traditional machine learning approaches, this becomes a problem since data availability is still sparse in the context of MOFs and their porperties. Moreover the data is very inhomgenous, which means there are a lot of properties of which we only have certain values given and a quality that is not consistent over all datasets. Secondly we still have to deal with the input format, which is topology agnostic and as such only has limited capabilites of mapping the structure onto the special properties. This provides us with an issue, as we need certain considerations about topology to achive the accuracy we desire, since as explained previously, only considering the building units opens up the representation to a lot of different invariances that come with different material properties.

### Traning of the Transformers

The authors aim to overcome these issues by self-supervised training using the already existent and sufficiently accurate CGCNN solution. Here we use this available model to learn a suitable compressed representation of the input instance. This has several advantages: We can learn the transformer model parameters more data efficient, as we now only need the MOFID and the Graph now to train a representation, rather than using different properties and resetting the deeper layers severeal times in the training process (as you would probably to in a multi task training). 
Also we now hope to learn a latent representation that includes the topology considerations, which is included in the representation given by the CGCNN as here we have the complete topology as input. So we try to get the transformer to firstly learn a representation that includes topology that is missing from the input data, in order to later use this representation to predict material properties more correctly later on. 

<div style="text-align:center">
    <a href="">
        <img src="/images/MOFs/training.png" alt="TRAIN" style="width:70%;height:70%">
    </a>
    <br>
    <i>Figure 3: The self-supervised training proposed in [1].</i>
</div>

So now that we have a satisfying representation we need to do the original property prediction, this is done via the standart training algorithms. The authors include data from several material databases:
- The [`Reticular Chemistry Structure Resource`](http://rcsr.anu.edu.au/) provides us with 3D representations as well as Systre-Notations of possible Secondary Building Units (especially nodes). We can use this data to automatically synthetizise data. Organic structures that function as lignants are available at many resources.
- The [`Cambridge Structural Database`](https://www.ccdc.cam.ac.uk/solutions/software/csd/) contains a lot of different Metal-Organic Compounds, that is also a lot of MOFs and organic linkers.
- The qMOF-Dataset as in [7] used high-throughput Density Functionality Theory quantum simulations to obtain material properties that include optimized geometries, absolute energies, band gap density of states, charge densities, and other attributes to provide a massive datasets on theoretical MOFs.
- The hMOF-Dataset [4] provides us with nearly 140 thousand hypothetical MOFs with pore size, methane storage, and surface size. It was automatically generated from secondary building units with templating, as it is normal in reticular chemistry 
- Lastly in [3] the Authors provide a collection of datasets named CoRe MOF (Computation-Ready, Experimental Metal–Organic Framework Database), that collected instances from different databases with extensive duplicate elimination and therefore considers a manifold in attributes that can be used for training.

All these datasets have been used to train the transformer once simply supervised and once with pretraining as described above. The self-supervised training increases accuracy, which makes us confident that the approach of the authors works conceptually.

## Evaluation

What the researchers took from it ?
- t-SNE

- Head weight interpretation

- Training data speedup

- Accuracy evaluation

### My Commentary

## References 

[1] Zhonglin Cao, Rishikesh Magar, Yuyang Wang, and Amir Barati Farimani. "MOFormer: Self-Supervised Transformer Model for Metal–Organic Framework Property Prediction." Journal of the American Chemical Society, 145(5):2958-2967, 2023. DOI: [10.1021/jacs.2c11420](https://doi.org/10.1021/jacs.2c11420)

[2] Benjamin J. Bucior, Andrew S. Rosen, Maciej Haranczyk, Zhenpeng Yao, Michael E. Ziebel, Omar K. Farha, Joseph T. Hupp, J. Ilja Siepmann, Alán Aspuru-Guzik, and Randall Q. Snurr. "Identification Schemes for Metal–Organic Frameworks To Enable Rapid Search and Cheminformatics Analysis." Crystal Growth & Design, 19(11):6682-6697, 2019. DOI: [10.1021/acs.cgd.9b01050](https://doi.org/10.1021/acs.cgd.9b01050)

[3] Yongchul G. Chung, Emmanuel Haldoupis, Benjamin J. Bucior, Maciej Haranczyk, Seulchan Lee, Hongda Zhang, Konstantinos D. Vogiatzis, Marija Milisavljevic, Sanliang Ling, Jeffrey S. Camp, Ben Slater, J. Ilja Siepmann, David S. Sholl, and Randall Q. Snurr. "Advances, Updates, and Analytics for the Computation-Ready, Experimental Metal–Organic Framework Database: CoRE MOF 2019." Journal of Chemical & Engineering Data, 64(12):5985-5998, 2019. DOI: [10.1021/acs.jced.9b00835](https://doi.org/10.1021/acs.jced.9b00835)

[4] Christopher E. Wilmer, Michael Leaf, Chang Yeon Lee, Omar K. Farha, Brad G. Hauser, Joseph T. Hupp, and Randall Q. Snurr. "Large-scale screening of hypothetical metal--organic frameworks." Nature chemistry, 4(2):83-89, 2012.

[5] Tian Xie and Jeffrey C. Grossman. "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." Physical Review Letters, 120(14), 2018. DOI: [10.1103/physrevlett.120.145301](https://doi.org/10.1103/physrevlett.120.145301)

[6] VV Butova, MA Soldatov, AA Guda, KA Lomachenko, and C Lamberti. "Metal-organic frameworks: Structure, properties, methods of synthesis and characterization." Russian Chemical Reviews, 85(3):280-307, 2016.

[7] Andrew S. Rosen, Shaelyn M. Iyer, Debmalya Ray, Zhenpeng Yao, Alan Aspuru-Guzik, Laura Gagliardi, Justin M. Notestein, and Randall Q. Snurr. "Machine learning the quantum-chemical properties of metal--organic frameworks for accelerated materials discovery." Matter, 4(5):1578-1597, 2021.
