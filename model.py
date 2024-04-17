import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.ao.quantization import get_default_qat_qconfig, default_observer, default_weight_observer, prepare_qat, fuse_modules, QuantStub, DeQuantStub
from utils import *

class LTRModel:
    
    def __init__(self, quantize: bool, df) -> None:
        self._quantize = quantize
        self.number_of_documents = df['doc_id'].nunique()
        self.accuracy = []
        print ('number of documents:', self.number_of_documents)


        layers = [
            ('lin1', nn.Linear(768, 256)),
            ('relu1', nn.ReLU()),
            ('lin4', nn.Linear(256, self.number_of_documents)),
        ]

        if self._quantize:
            self.model = nn.Sequential(OrderedDict([
                ('quant', QuantStub())] + layers + [('dequant', DeQuantStub())
            ]))
            self.model.eval()
            self.model.qconfig = get_default_qat_qconfig('onednn')
            self.model = fuse_modules(self.model,
                [['lin1', 'relu1'], ['lin2', 'relu2'], ['lin3', 'relu3']])
            self.model = prepare_qat(self.model.train())
        else:
            self.model = nn.Sequential(OrderedDict(layers))

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def serialize_model(self) -> io.BytesIO:
        buffer = io.BytesIO()
        if self._quantize:
            self.model.eval()
            model = torch.quantization.convert(self.model, inplace=False)
        else:
            model = self.model

        torch.save(model, buffer)

        return buffer

    def make_input(self, query_vector):
        """
        Make query input for model.
        """
        return np.array([query_vector], dtype=np.float32).flatten()

    def _train_step(self, train_data: np.ndarray, label: int) -> float:
        if label.shape[0] > 1:  # Checking if one-hot encoded
            label = torch.argmax(label, dim=0)
        output = self.model(torch.FloatTensor(train_data))  # Ensure input is a proper torch tensor
        loss = self._criterion(output, label)  # CrossEntropy expects 2D input
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self.accuracy.append((torch.argmax(output, dim=0) == label))
        return loss.item()

    def train(self, train_data, labels, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            losses = self._train_step(train_data, labels)
        self.model.eval()

    def apply_updates(self, update_model):
        if self._quantize:
            updates = {
                name: module for name, module in update_model.named_modules() if isinstance(module, nn.quantized.Linear)
            }
            for name, param in self.model.named_parameters():
                layer, attr = name.split('.')
                if attr == 'weight':
                    data = updates[layer].weight()
                elif attr == 'bias':
                    data = updates[layer].bias()
                param.data = (param.data + torch.dequantize(data).data) / 2.0
        else:
            update_model_state = update_model.state_dict()
            for name, param in self.model.named_parameters():
                if name in update_model_state:
                    param.data = (param.data + update_model_state[name]) / 2.0
        
        print(fmt('Model updated', 'gray'))