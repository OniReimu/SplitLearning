import logging

import torch.distributed.rpc
import torch
import torch.nn as nn
from models import *
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import logging
import os
from collections import Counter
from copy import deepcopy

from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() and \
                      torch.backends.mps.is_built() else "cpu")

'''
The way to integrate SISA into the traditional split unlearning is as follows:
---
1. k clients conduct local training upon their local dataset for some N epochs, followed by the model weights of their layers getting frozen (locked). At this point, the backward propagation will only happen on each client. 

2. The k clients send the output of their intermediate layers to the server, along with the labels shared to the server. Then the server conduct training for some M epochs. At this point, the backward propagation will only happen on the server side, as the weights of all clients have been locked. 

3. After the M epochs training finished, the whole training process ends while clients can evaluate the model by requesting the server.
---
From my understanding, in all previous split learning approaches, the weights on clients' sides can also be updated and adjusted to align with the label. But in our case, after all local learning has been done, the model parameters on clients' side will be locked until the server finishes its training. 
This implies that the server's model should have enough capacity (a sufficiently large model comprising sufficient "idle weights") to be adjusted and aligned with a training task.
'''

class alice(object):
    def __init__(self,server,bob_model_rrefs,rank,args):
        self.client_id = rank
        self.local_epochs = args.epochs
        self.server_epochs = args.server_epochs
        self.start_logger()

        self.bob = server

        self.model = model1_sisa()

        self.criterion = nn.CrossEntropyLoss()

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum = 0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)

        self.lr = args.lr

        self.load_data(args)

    def train(self):
        self.logger.info("Local Training")
        
        for epoch in tqdm(range(self.local_epochs), desc="Epochs", ascii=" >="):
            for i, data in enumerate(tqdm(self.train_dataloader, desc="Batches", ascii=" >=")):
                inputs,labels = data
                # print(f"Batch size for batch-{i}: {inputs.size(0)}")  # Prints the batch size, batch size is 16 by default
                self.optimizer.zero_grad()

                activation_alice = self.model(inputs)
                loss = self.criterion(activation_alice,labels)

                # local backward pass
                loss.backward()

                self.optimizer.step()

    def eval(self):
        correct = 0
        total = 0
        
        # since we're not training, we don't need to calculate the gradients for our outputs
        self.bob.rpc_sync().switch_mode_to_eval()
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                activation_alice = self.model(images)
                outputs = self.bob.rpc_sync().inference(activation_alice)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.info(f"Alice-{self.client_id} Evaluating Data: {round(correct / total, 3)}")
        return correct, total
    
    
    def eval_breakdown(self, omit_label):
        correct = 0
        total = 0

        # These are for tracking the unlearned label
        correct_unlearned = 0
        total_unlearned = 0

        # These are for tracking the remaining labels
        correct_remaining = 0
        total_remaining = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        self.bob.rpc_sync().switch_mode_to_eval()
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data

                # calculate outputs by running images through the network
                activation_alice = self.model(images)

                # TODO: concat
                outputs = self.bob.rpc_sync().inference(activation_alice)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Now let's break this down for the unlearned and remaining labels
                is_unlearned = labels == omit_label
                is_remaining = labels != omit_label

                total_unlearned += is_unlearned.sum().item()
                total_remaining += is_remaining.sum().item()

                correct_unlearned += (predicted[is_unlearned] == labels[is_unlearned]).sum().item()
                correct_remaining += (predicted[is_remaining] == labels[is_remaining]).sum().item()

        self.logger.info(f"Alice-{self.client_id} Evaluating Data: {round(correct / total, 3)}")
        
        # "0" indicates there is no such a label in the local testing dataset
        self.logger.info(f"Alice-{self.client_id} Evaluating Unlearned label-{omit_label}: {round(correct_unlearned / total_unlearned if total_unlearned else 0, 3)}")
        self.logger.info(f"Alice-{self.client_id} Evaluating Remaining labels: {round(correct_remaining / total_remaining if total_remaining else 0, 3)}")

        return correct, total, correct_unlearned, total_unlearned, correct_remaining, total_remaining     
    
    def unlearn(self, omit_label):
        self.logger.info(f"Unlearning label: {omit_label}, and reset the model")

        # Reset model parameters
        self.reset_model()
        # Now we reset the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Filter out instances of the omitted label(s) from the training data
        # Prepare the unlearned dataset, which omits the given label
        unlearn_data = [(inputs, labels) for batch in self.train_dataloader for inputs, labels in zip(*batch) if labels != omit_label]

        # Convert unlearn_data to a DataLoader
        self.unlearn_dataloader = torch.utils.data.DataLoader(unlearn_data, batch_size=16)    
        self.logger.info("Retraining dataset: {}".format(dict(Counter(label.item() for data in self.unlearn_dataloader for label in data[1]))))
        self.logger.info("Test dataset (retraining): {}".format(dict(Counter(self.test_dataloader.dataset[:][1].numpy().tolist()))))
        
        for epoch in tqdm(range(self.local_epochs), desc="Epochs", ascii=" >="):
            for i, data in enumerate(tqdm(self.unlearn_dataloader, desc="Batches", ascii=" >=")):
                inputs,labels = data
                self.optimizer.zero_grad()

                activation_alice = self.model(inputs)
                loss = self.criterion(activation_alice,labels)

                # local backward pass
                loss.backward()

                self.optimizer.step()

    def load_data(self,args):
        self.train_dataloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.client_id}_train.pt"))
        self.test_dataloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.client_id}_test.pt"))

        self.n_train = len(self.train_dataloader.dataset)
        self.logger.info("Local Data Statistics:")
        self.logger.info("Dataset Size: {:.2f}".format(self.n_train))
        self.logger.info("Training dataset: {}".format(dict(Counter(self.train_dataloader.dataset[:][1].numpy().tolist()))))
        self.logger.info("Test dataset: {}".format(dict(Counter(self.test_dataloader.dataset[:][1].numpy().tolist()))))

    def start_logger(self):
        self.logger = logging.getLogger(f"alice{self.client_id}")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")

        fh = logging.FileHandler(filename=f"logs/alice{self.client_id}.log",mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        self.logger.addHandler(fh)

        self.logger.info("Alice is going insane!")

    def give_activation_and_labels(self, unlearned=False):
        activations = []
        labels_list = []
        dataloader = self.train_dataloader if not unlearned else self.unlearn_dataloader

        for i, data in enumerate(dataloader):
            inputs, labels = data
            # Detach the output tensor from the computational graph. By detaching, we're saying
            # we no longer need to compute gradients with respect to these tensors during backward 
            # pass because they're not learnable parameters (freezing weights) of our current model
            # (Bob's model). This is useful when you want to take the output of a network, do some operations on it,
            # and then compute some loss. If we didn't detach, the backward pass would attempt to 
            # go all the way back through Alice's model as well, which we want to avoid in this case.
            activations.append(self.model(inputs).detach())
            labels_list.append(labels)
        return activations, labels_list

    def give_weights(self):
        return deepcopy(self.model.state_dict())

    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def reset_model(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class bob(object):
    def __init__(self,args):
        self.server = RRef(self)
        self.model = model2_sisa_concat(num_clients=args.client_num_in_total)
        self.server_epochs = args.server_epochs

        self.client_num_in_total  = args.client_num_in_total
        self.start_logger()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)

        self.alices = {rank+1: rpc.remote(f"alice{rank+1}", alice, \
                        (self.server,list(map(lambda x: RRef(x),self.model.parameters())),rank+1,args)) for rank in range(args.client_num_in_total)}
        self.activation_and_labels_cache = {} # Cache for Alice's activation and labels

    def get_activation_and_labels(self, client_id, unlearned=False, unlearn_id=None):
        # Check if activation and labels are already cached
        cache_key = (client_id, unlearned, unlearn_id)
        if cache_key in self.activation_and_labels_cache:
            return self.activation_and_labels_cache[cache_key]
        
        # Otherwise, fetch and cache the activation and labels
        activation_and_labels = self.alices[client_id].rpc_sync().give_activation_and_labels(unlearned)
        self.activation_and_labels_cache[cache_key] = activation_and_labels
        return activation_and_labels

    def train_and_backward(self, unlearn_request_from_alices, unlearn_id):
        self.logger.info("Global Training")

        self.switch_mode_to_train()
        for epoch in tqdm(range(self.server_epochs), desc="Epochs", ascii=" >="):

            intermediate_inputs_list = []
            labels_list = []
            for client_id in range(1, self.client_num_in_total + 1):
                intermediate_inputs, labels = self.get_activation_and_labels(client_id, unlearned=True, unlearn_id=unlearn_id) \
                                                if client_id in unlearn_request_from_alices else \
                                                self.get_activation_and_labels(client_id, unlearned=False)
                # If intermediate_inputs and labels are already tensors or compatible arrays, append them directly
                intermediate_inputs_list.extend(intermediate_inputs)
                labels_list.extend(labels)

            concatenated_inputs = torch.cat(intermediate_inputs_list, dim=0)
            concatenated_labels = torch.cat(labels_list, dim=0)

            for intermediate_input, label in tqdm(zip(concatenated_inputs, concatenated_labels), total=len(concatenated_inputs), desc="Batches", ascii=True, mininterval=0.1):
                    self.optimizer.zero_grad()
                    # Detach activation from Alice's computation graph
                    output = self.model(intermediate_input.detach())
                    loss = self.criterion(output, label)

                    # global backward pass
                    loss.backward()

                    self.optimizer.step()



            # for client_id in range(1, self.client_num_in_total + 1): 
            #     intermediate_inputs, labels = self.get_activation_and_labels(client_id, unlearned=True, unlearn_id=unlearn_id) \
            #                                     if client_id in unlearn_request_from_alices else \
            #                                     self.get_activation_and_labels(client_id, unlearned=False)

            #     for intermediate_input, label in tqdm(zip(intermediate_inputs, labels), total=len(intermediate_inputs), desc="Batches", ascii=True, mininterval=0.1):
            #         self.optimizer.zero_grad()
            #         # Detach activation from Alice's computation graph
            #         output = self.model(intermediate_input.detach())
            #         loss = self.criterion(output, label)

            #         # global backward pass
            #         loss.backward()

            #         self.optimizer.step()

        self.logger.info("Global training completed.")

    def inference(self,x):
        return self.model(x)
    
    def train_request(self,client_id):
        # call the train request from alice
        self.logger.info(f"Train Request for Alice-{client_id}")
        self.alices[client_id].rpc_sync(timeout=0).train()

    def train_request_parallel(self):
        self.logger.info(f"Train all Alices in parallel")

        train_parallel = [self.alices[client_id].rpc_async(timeout=0).train() for client_id in
                      range(1, self.client_num_in_total + 1)]
        
        # Wait for all futures to complete
        for res in train_parallel:
            res.wait() # Wait for the remote call to complete

    def eval_request(self):
        self.logger.info("Initializing Evaluation of all Alices")
        total = []
        num_corr = []
        check_eval = [self.alices[client_id].rpc_async(timeout=0).eval() for client_id in
                      range(1, self.client_num_in_total + 1)]
        for check in check_eval:
            corr, tot = check.wait()
            total.append(tot)
            num_corr.append(corr)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr) / sum(total)))

    def unlearn_request(self, client_id, omit_label):
        # call the unlearn request from alice
        self.logger.info(f"Unlearn Request for Alice-{client_id} upon the label-{omit_label}")

        self.alices[client_id].rpc_sync(timeout=0).unlearn(omit_label)

    def eval_request_breakdown(self, omit_label):
        self.logger.info("Initializing Evaluation of all Alices. Breaking down to the unlearned and the remaining data")
        total = []
        total_unlearned = []
        total_remaining = []
        num_corr = []
        num_corr_unlearned = []
        num_corr_remaining = []

        check_eval = [self.alices[client_id].rpc_async(timeout=0).eval_breakdown(omit_label) for client_id in
                    range(1, self.client_num_in_total + 1)]

        for check in check_eval:
            corr, tot, corr_unlearned, tot_unlearned, corr_remaining, tot_remaining = check.wait()
            total.append(tot) # The server now has all the labels collected from all clients
            total_unlearned.append(tot_unlearned)
            total_remaining.append(tot_remaining)
            num_corr.append(corr)
            num_corr_unlearned.append(corr_unlearned)
            num_corr_remaining.append(corr_remaining)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr) / sum(total)))
        self.logger.info("Accuracy over unlearned data: {:.3f}".format(sum(num_corr_unlearned) / sum(total_unlearned)))
        self.logger.info("Accuracy over remaining data: {:.3f}".format(sum(num_corr_remaining) / sum(total_remaining)))

    def freeze_alice_weights(self, client_ids):
        for client_id in client_ids:
            self.logger.info("Server training starts. Freezing weights for Alices-{}.".format(client_id))
            self.alices[client_id].rpc_sync().freeze_weights()

    def unfreeze_alice_weights(self, client_ids):
        for client_id in client_ids:
            self.logger.info("Unfreezing weights for Alices-{}.".format(client_id))
            self.alices[client_id].rpc_sync().unfreeze_weights()   

    def switch_mode_to_train(self):
        self.model.train()  

    def switch_mode_to_eval(self):
        self.model.eval()          

    def start_logger(self):
        self.logger = logging.getLogger("bob")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")

        fh = logging.FileHandler(filename="logs/bob.log", mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        self.logger.addHandler(fh)
        self.logger.info("Bob Started Getting Tipsy")