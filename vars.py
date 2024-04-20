accuracy_threshold = 0.01 #variable concerning early stopping for rolling_mean accuracy, currently unused
batches_save_threshold = 3000 #variable concerning the distance to each model checkpoint

shard = 2 #index of the shard. Vary it and run main.py to train different shards
batch_size = 32
peer_nbr = 10 #number of peers in the network
total_doc_count = peer_nbr * 500 #number of documents available in the global pool
number_of_docs_per_user =  [100, 200] #number of documents that could be chosen by any one peer
model_name = "t5-small" #T5 model used
# model_name = "t5-base"
