# De-DSI
This code is the implementation linked to the decentralized experiment (experiment 3) of the paper De-DSI: 
Decentralized Differentiable Search Index, DOI: https://doi.org/10.1145/3642970.3655837.

In order to run the decentralized experiment the following steps should be followed:

* Download the ORCAS dataset and place it in the root folder
* Install the required packages in 'requirements.txt'. When running this on Linux or MacOS, 
some problems with libsodium may be encountered. If so, see https://doc.libsodium.org/installation.
If that doesn't solve it
* Copy the 'data' folder as many times as the number of shards which you'd like to train
* Run main.py, it will populate the data folder with information about the training process 
(losses, accuracies, checkpointed models, datasets for each peer)
* Rename the folder to 'groupX' where X denotes the number of the shard and place it in the folder called 'aggregated_results'
* Once you have enough shards trained, you can run the ensemble in the 'Computing inter/intra-group performances.ipynb' notebooks
* The notebook which aggregates the results of the ensembles is the 'Data Analysis of Group Performances.ipynb'
* Depending on the number of shards required, small changes may need to be applied to the jupyter notebooks




