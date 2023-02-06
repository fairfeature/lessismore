# Less is More: Feature Engineering for Fairness and Performance of Machine Learning Software


This repository stores our experimental codes for the paper “Less is More: Feature Engineering for Fairness and Performance of Machine Learning Software”, HEFR is short for the method we propose in this paper: *Hibrid_importance and Early_validation based Feature Ranking*. The experimental part of our work is divided into two parts - empirical research and evaluation:



## Empirical Study

### Datasets

We use 7 datasets, all of which are widely used in fairness research: **Adult, COMPAS, German Credit, Bank Marketing, MEPS, Default, Home** (where Adult, COMPAS and German all contain two protected features). The first five datasets can be loaded through python's aif360 package, for example:

```python
from aif360.datasets import AdultDataset
```
The other two datasets (Default and Home) are placed in the dataset folder.

### Experimental Settings

We use the **decision tree** as the main classifier in the experiment. Besides, we split 60% of the dataset into the training set, 20% into the validation set and 20% into the test set. To minimize the effects of randomness, we repeat the experiments 50 times and report the mean.

### Code
RQ1 is a replication of the paper [*"Ignorance and Prejudice" in Software Fairness*](https://ieeexplore.ieee.org/document/9402057). We supplement the results with two datasets on the basis of the original paper. For the implemented code, we refer to the code repository of the original paper.

Our empirical research on removal of protected features and the performance and fairness trends of enlarging features without protected features is placed in the "RQ2-3" folder.







## Evaluation for HEFR

### Datasets

We conduct experiments on ten scenarios on seven datasets, where Adult, COMPAS and German all contain two protected features. Just same as the datasets used in empirical study.

### Experimental Settings

We used **decision tree** as the main classifier, in addition to this, we also discuss the results of our method on RF and GBDT in RQ4a. Other settings are consistent with empirical study.

### Code

You can easily reproduce our method, we provide its code in RQ4 and RQ5. 

The filter.py in the two folders records the implementation of **hybrid importances to combine performance and fairness** process of HEFR, and the **early validation for hybrid importances** part of the algorithm is shown in the file feature_HEFR.py. 




### Baseline methods

We compare HEFR to random selection and four feature selection methods (RFE, ChiSquare, ReliefF and SFM) to observe how many features each method needs to select to reach the First Equivalent Point (**FEP**). For the baselines ChiSquare and ReliefF, we apply the chi_square and reliefF function in the skfeature library. For the baseline SFM, we apply the  SelectFromModel function in the sklearn library. As for the RFE algorithm, it is the representation of our method when $\alpha$ is equal to 0.

## Acknowledgment
Many thanks to the code contribution of [*"Ignorance and Prejudice" in Software Fairness*](https://ieeexplore.ieee.org/document/9402057), our experiments are based on the work of this paper.
