# De-DSI
This code is the implementation linked to the decentralized experiment (experiment 3) of the paper De-DSI: 
Decentralized Differentiable Search Index, 

DOI: https://doi.org/10.1145/3642970.3655837.

Arxiv link: https://arxiv.org/abs/2404.12237


In order to run the decentralized experiment the following steps should be followed:

* Download the [ORCAS](https://microsoft.github.io/msmarco/ORCAS.html) 'click data' dataset and place it in the root folder
* Install the required packages in 'requirements.txt'. When running this on Linux or MacOS, 
some problems with libsodium may be encountered. If so, see https://doc.libsodium.org/installation.
* The hyper-parameters can be found in 'vars.py'
* The 'main.py' script creates a folder structure: aggregated_results/groupX/{accuracies,datasets,losses,models}, where
X is the 'shard' variable from vars.py of the shard trained
* By varying the shard variable from vars.py the reader can train multiple shards, denoted as 'group' in the folder 'aggregated_results'
* After training a number of shards, the reader can run the 'Computing inter-group performance.ipynb' or 
'Computing intra-group performance.ipynb' jupyter notebooks which are designed to run the ensemble model 
on one shard or on multiple shards:
  * 'Computing intra-group performance.ipynb' picks a number of models from each shard and tests the ensemble 
  on the same shard from where the models were picked from
  * 'Computing inter-group performance.ipynb' picks a number of models from each shard and tests the ensemble on the dataset comprised
  of all shards together. Also performs computation of accuracy of each individual model on their their dataset witin the shard
  (both local - dataset owned by the peer, and global, dataset owned by all peers in the shard together)

  The jupyter notebooks are made to run for 3 shards currently, if the reader decides to include more/fewer shards 
small modifications to accommodate the code may need to be performed.

* The 'Data Analysis of Group Performances.ipynb' is meant to take the output of the previous notebooks and aggregate 
them in the 'df_total' table


NOTE: The Jupyter Notebooks are meant to guide the reader through the process of estimating the performance of the ensembles.
The reader may also perform this analysis themselves based on the output placed in the 'groupX' (located in 'aggregated_results'), 
after running 'main.py', if they so choose.