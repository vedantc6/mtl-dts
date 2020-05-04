# mtl-dts
Implementation for "Deeper Task-Specificity Improves Joint Entity and Relation Extraction"

This is the final project for CS533: Natural Language taken at Rutgers University under Prof. Karl Stratos.
For the actual paper, please go to https://arxiv.org/abs/2002.06424.

The data can be downloaded by running the script taken from https://github.com/markus-eberts/spert. Also, make a model/ folder to save models at every epoch

Our code is an implementation for it with some extensions, particularly a novel transformer architecture change (for this model) and tested on experimental batch.

To run the actual paper, all that has to be done is python main.py. Please check the argument options for more detail.
To run the transformer model, go to vedant_ade branch (branch name has to be changed later on). 

In future, everything will be merged to a single branch and there will be options to run with biRNNs or transformers. 
