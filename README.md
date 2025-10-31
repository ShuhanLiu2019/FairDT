# Fairness-Aware Graph Representation Learning through Bias Disentanglement
This repository contains a PyTorch implementation of paper "Fairness-Aware Graph Representation Learning through Bias Disentanglement".


## Environment Settings

- Python: 3.8  
- CUDA: 11.3  

```
pytorch==1.11.0
dgl-cu113==0.8.2
matplotlib==3.7.5
networkx==3.1
pandas==2.0.3
PyWavelets==1.4.1
PyYAML==6.0.2
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.11.2
torch_cluster==1.6.3
torch_geometric==1.7.2
torch_scatter==2.1.2
torch_sparse==0.6.13
torch_spline_conv==1.2.2
torchvision==0.12.0
```

See 'requirements.txt' for more details.


## Datasets
We provide the six public datasets we used in the folder "data".   
The original dataset is in the "raw" folder in each dataset folder.   
The processed dataset is in the "processed" folder in each dataset folder.   


## Reproduce the results
Run the following commands directly.

`sh exp_Fair.sh`

For additional fairness metrics (Equalized Odds and Predictive Equality), please run the following commands.

`sh exp_Fair_addmetric.sh`


## Contact
If you have any questions, please feel free to contact me with shuhanliu@mail.sdu.edu.cn.
