import struct
import csv
import numpy as np
import torch
from txtai.embeddings import Embeddings
from itertools import combinations
from operator import itemgetter
from model import LTRModel
import warnings
from vars import num_epochs
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class LTR(LTRModel):
    """
    LTR class for learning to rank.
    
    Attributes:
        metadata: mapping of article uid to title
        embeddings_map: mapping of article uid to feature vector
        embeddings: txtai embeddings model
        results: local cache of query => results
    """
    embeddings_map = {}
    embeddings = None
    results = {}

    def __init__(self, quantize: bool, df):
        super().__init__(quantize, df)

        with open('data/embeddings.bin', 'rb') as embeddings_bin:
            format_str = '8s768f'
            while True:
                bin_data = embeddings_bin.read(struct.calcsize(format_str))
                if not bin_data:
                    break
                data = struct.unpack(format_str, bin_data)
                uid = data[0].decode('ascii').strip()
                features = list(data[1:])
                self.embeddings_map[uid] = features

        self.embeddings = Embeddings({ 'path': 'allenai/specter' })
        self.embeddings.load('data/embeddings_index.tar.gz')

    def embed(self, x: str) -> list[float]:
        """
        Get vector representation of a (query) string.
        """
        return self.embeddings.batchtransform([(None, x, None)])[0]

    def gen_train_data(self, query: str, selected_res: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate training data based on the selected relevant document.

        Args:
            query: query string
            selected_res: index of selected result

        Returns:
            Tuple of input vectors and corresponding labels
        """
        # Check if the query results are already cached, else retrieve using embeddings
        query_vector = self.embed(query)

        # Now we generate one-hot labels based on selected results. The selected document is "1", and all others are "0".
        labels = [1 if i == selected_res else 0 for i in range(self.number_of_documents)]


        # Pair each document embedding with the query embedding
        train_data = query_vector

        return np.array(train_data), np.array(labels)

    def query(self, query: str) -> list[str]:
        """
        Returns ranked list of results (titles) for a query.
        If results to this query are unknown, semantic search is performed, and the model is trained.
        """
        if query not in self.results:
            # bootstrap model with semantic search results
            self.results[query] = [x for x, _ in self.embeddings.search(query, 5)]

        # Here, we're going to score each document individually and rank based on those scores
        query_vector = self.embed(query)

        # Preparing the input for the model. Each row will be the concatenation of the query vector with a document vector.
        input_vectors = np.array([query_vector + self.embeddings_map[doc_id] for doc_id in self.results[query]])

        # Convert to a PyTorch tensor
        input_vectors = torch.from_numpy(input_vectors).float()

        # Get scores from the model
        scores = self.model(input_vectors)  # This line assumes your model returns multiple scores in one forward pass.

        # Convert scores to a Python list and pair each score with the document's index
        scored_results = list(enumerate(scores.tolist()))

        # Sort the results based on the scores
        ranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

        # Retrieve the document IDs based on their new ranking
        ranked_doc_ids = [self.results[query][index] for index, _ in ranked_results]

        return [doc_id for doc_id in ranked_doc_ids]

    def on_result_selected(self, query: str, selected_res: int):
        """
        Retrains the model with the selected result as the most relevant.
        """
        train_data, labels = self.gen_train_data(query, selected_res)
        self.train(torch.from_numpy(train_data).float(), torch.tensor(labels), num_epochs)

        # Update the cached results with a new ranking based on the retrained model
        self.results[query] = self.query(query)

