from asyncio import run
import os
import torch

from ipv8.taskmanager import TaskManager
from dataclasses import dataclass
import threading
from ipv8.community import Community, CommunitySettings
from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import overwrite_dataclass
from ipv8.types import Peer
from time import time, localtime, strftime

from sklearn.model_selection import train_test_split
from ipv8.util import run_forever
from ipv8_service import IPv8
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import pandas as pd
import random

from vars import *

# Check if CUDA is available and set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Enhance normal dataclasses for IPv8 (see the IPV8 serialization documentation)
dataclass = overwrite_dataclass(dataclass)

@dataclass(msg_id=1)  # The value 1 identifies this message and must be unique per community
class Query_res:
    query: str
    result: str

class LTRCommunity(Community):
    community_id = b'\x9d\x10\xaa\x8c\xfa\x0b\x19\xee\x96\x8d\xf4\x91\xea\xdc\xcb\x94\xa7\x1d\x8b\x00'

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        print ('-----------------------------------------------------------------')

        number_of_docs_for_this_user = np.random.randint(number_of_docs_per_user[0], number_of_docs_per_user[1])
        
        with open('./output.log', 'a') as file:
                file.write(' doc nbr:' + str(number_of_docs_for_this_user) + ' ')
        self.df = df[df['doc_id'].isin(list(docs_to_be_used.sample(number_of_docs_for_this_user)['doc_id']))]
       	with open('./output.log', 'a') as file:
                file.write(' df size ' + str(self.df.shape[0]) + ' ')

        self.batches_so_far = 0
        self.current_queries = []
        self.current_docs = []
        self.timestamps = []
        self.accuracies_avg = 0
        self.accuracies_sum = 0
        self.rolling_window = 100

        task_manager = TaskManager()
        self.losses = []
        self.accuracies = []
        self.accuracy = []
        self.add_message_handler(Query_res, self.on_message)
        self.ready_for_input = threading.Event()
        self.lamport_clock = 0
        self.past_data = {'queries': [], 'results': []}

    def on_peer_added(self, peer: Peer) -> None:
        print("I am:", self.my_peer, "I found:", peer)

    def train_model(self, queries, responses):
        self.got_here = False


        self.batches_so_far+=1
        # Tokenize the lists of queries and responses
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True).input_ids.to(device)
        labels = self.tokenizer(responses, padding=True, return_tensors="pt", truncation=True).input_ids.to(device)

        # Forward pass
        outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # Extract logits and convert to token IDs
        logits = outputs.logits
        predicted_token_ids = torch.argmax(logits, dim=-1)

        self.accuracy = []

        # Decode token IDs to text for each item in the batch
        for i in range(predicted_token_ids.size(0)):
            predicted_text = self.tokenizer.decode(predicted_token_ids[i], skip_special_tokens=True)
#            with open('./output.log', 'a') as file:
#                        file.write('\n' + predicted_text + ' ' + responses[i])
            if predicted_text == responses[i]:
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)

        acc = np.sum(self.accuracy) / len(self.accuracy)

        self.losses.append(round(float(loss.detach()),3))
        self.accuracies.append(acc)

        self.timestamps.append(int(time()))
        self.accuracies_sum += acc
        if len(self.accuracies) > self.rolling_window:
            self.accuracies_sum -= self.accuracies[-self.rolling_window]
            self.accuracies_avg = self.accuracies_sum / self.rolling_window
        else:
            self.accuracies_avg = self.accuracies_sum / len(self.accuracies)

        if self.batches_so_far % batches_save_threshold == 0:
            pd.DataFrame(list(zip(self.timestamps, self.accuracies)),
                         columns = ['timestamps','accuracies']).to_csv(f'aggregated_results/{root_folder_for_shard}/accuracies/{self.my_peer.address[1]}_accuracies.csv')
            pd.DataFrame(list(zip(self.timestamps, self.losses)),
                         columns = ['timestamps','losses']).to_csv(f'aggregated_results/{root_folder_for_shard}/losses/{self.my_peer.address[1]}_losses.csv')
            self.model.save_pretrained(f'aggregated_results/{root_folder_for_shard}/models/{self.my_peer.address[1]}_{self.batches_so_far}')
            self.df.to_csv(f'aggregated_results/{root_folder_for_shard}/datasets/{self.my_peer.address[1]}_df.csv')
            # self.change_df(test_df)
            # if self.accuracies_avg>0.95:
            if self.batches_so_far > 10:
                raise SystemExit
                end_time = time()
                print (end_time - start_time)


        print(f'peer port:{self.my_peer.address[1]}, loss: {round(float(loss.detach()),3)}, ACCURACY ": {round(acc,2)}, '
              f'ACCURACY_AVG: {round(self.accuracies_avg,2)}')
        with open('./output.log', 'a') as file:
                file.write(f'peer port:{self.my_peer.address[1]}, loss: {round(float(loss.detach()),3)}, ACCURACY ": {round(acc,2)}, '
              f'ACCURACY_AVG: {round(self.accuracies_avg,2)}')
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()





    def get_querynres(self):
        self.row = np.random.randint(0, self.df.shape[0])
        # print ('ROW NUMBER: ', self.row)
        query = self.df.iloc[self.row]['query']
        # selected_res = docs[docs == self.df.iloc[self.row]['doc_id']].index[0]
        selected_res = self.df.iloc[self.row]['doc_id']
        self.row += 1
        return query, selected_res

    def started(self) -> None:
        print('Indexing (please wait)...')

        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)

        # Training loop
        self.model.train()

        self.network.add_peer_observer(self)
        print(self.get_peers())


        async def start_communication() -> None:
            print(f'running comms routine with {len(self.get_peers())} peers')
            if len(self.get_peers()) > 0:
                # if not self.lamport_clock:
                query, selected_res = self.get_querynres()
                print('starting comms & ending comms')
                self.cancel_pending_task("start_communication")



                p = random.choice(self.get_peers())
                self.ez_send(p, Query_res(query=query, result=selected_res))
            else:
                print('gonna try again')
                pass

        async def send_query() -> None:
            if len(self.get_peers()) == 0:
                return None
            new_query, new_res = self.get_querynres()
            if len(self.current_queries) == 0:
                for i in range ( round(batch_size/len(self.get_peers())) ):
                    new_query, new_res = self.get_querynres()
                    self.current_queries.append(new_query)
                    self.current_docs.append(new_res)

            p = random.choice(self.get_peers())

            self.check_batch_size_and_train()
            self.ez_send(p, Query_res(query=new_query, result=new_res))

        self.register_task("start_communication", start_communication, interval=5.0, delay=0)
        self.register_task("send_random_q_d_pair", send_query, interval=0.01, delay=0)
    def check_batch_size_and_train(self):
        if len(self.current_queries)>=batch_size:
            self.train_model(self.current_queries, self.current_docs)
            self.current_queries = []
            self.current_docs = []

    def change_df(self, df_changer):
            self.df = df_changer.copy()
            self.got_here = True

    @lazy_wrapper(Query_res)
    def on_message(self, peer: Peer, payload: Query_res) -> None:

        self.current_queries.append(payload.query)
        self.current_docs.append(payload.result)
        self.check_batch_size_and_train()




async def start_communities() -> None:
    for i in range(peer_nbr):
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "medium", f"certs/ec{i}.pem")
        builder.add_overlay("LTRCommunity", "my peer",
                            [WalkerDefinition(Strategy.RandomWalk, 10, {'timeout': 3.0})],
                            default_bootstrap_defs, {}, [('started',)])
        await IPv8(builder.finalize(),
                   extra_communities={'LTRCommunity': LTRCommunity}).start()
    await run_forever()




start_time = time()
print (start_time)
all_docs_till_now = []

df = pd.read_csv('./orcas.tsv', sep='\t', header=None,
                 names=['query_id', 'query', 'doc_id', 'doc'])


def find_shard_folders(directory):
    """
    List all folders within a given directory that have 'X' in their name.

    :param directory: The root directory to search within.
    :return: A list of paths to folders that contain 'X' in their name.
    """
    folders = []
    for root, dirs, files in os.walk(directory):
        # Filter directories in the current root that contain 'X'
        matched_dirs = [os.path.join(root, d) for d in dirs if 'group' in d]
        folders.extend(matched_dirs)

    return folders
previous_shards = find_shard_folders('aggregated_results')
print (previous_shards)
for s in previous_shards:
    prev_docs = pd.read_csv(f'{s}/datasets/train_df.csv')['doc_id'].unique()
    all_docs_till_now.extend(list(prev_docs))
    print ('docs till now', len(all_docs_till_now))
df = df[~df['doc_id'].isin(all_docs_till_now)]

root_folder_for_shard = f'group{shard}'
os.makedirs(f'aggregated_results/{root_folder_for_shard}/datasets', exist_ok=True)
os.makedirs(f'aggregated_results/{root_folder_for_shard}/models', exist_ok=True)
os.makedirs(f'aggregated_results/{root_folder_for_shard}/accuracies', exist_ok=True)
os.makedirs(f'aggregated_results/{root_folder_for_shard}/losses', exist_ok=True)
cnter = df.groupby('doc_id').count().sort_values('query',ascending = False) # get most referenced docs subset
cnter = cnter[cnter['query']>=2] # remove docs that are referenced only once to make stratify work
docs_to_be_used = cnter.sample(total_doc_count).reset_index()
df = df[df['doc_id'].isin(list(docs_to_be_used['doc_id']))]


with open('./output.log', 'a') as file:
                file.write(' doc nbr:' + str(docs_to_be_used.shape[0]) + ' shape:' + str(df.shape[0]))

train_df, test_df = train_test_split(df, test_size=0.5, random_state=42, stratify=df['doc_id']) # split into train and test
train_df.to_csv(f'aggregated_results/{root_folder_for_shard}/datasets/train_df.csv')
test_df.to_csv(f'aggregated_results/{root_folder_for_shard}/datasets/test_df.csv')

df = train_df.copy()
run(start_communities())
