# Split-Learning on Heterogenous Distributed MNIST

## Bob (coordinator)
Bob consists of two main functions 
1. *Train Request* 
   1. Request for Alice_x to update model weights to last trained Alice_x' weights.
   2. Perform flow of forward and backward pass in figure below for N*Batches iterations.
   3. Round robin fashion request for next Alice_x'' to begin training.
2. *Evaluation Request*
   1. Request for each Alice_x to perform evaluation on the test set.
   2. Aggregate results of each Alice_x and log the overall performance.
   
Details from https://dspace.mit.edu/bitstream/handle/1721.1/121966/1810.06060.pdf?sequence=2&isAllowed=y .

![Alt text](imgs/split_nn.PNG?raw=true  "Decentralized Split Learning Architecure")

# SISA-based Split-Learning on Heterogenous Distributed MNIST

## Learning

The SISA-based Split Learning process consists of three main stages:
1. *Local Training by Alice_x*
   1. Each of the Alice_x conducts local training on their respective local datasets for N epochs.
   2. Following this, the model weights of their layers are frozen (locked). Hence, any subsequent backward propagation will occur exclusively on the side of each Alice_x.
2. *Server Training by Bob*
   1. All Alice_x then send the output of their intermediate layers, along with the corresponding labels, to Bob.
   2. Bob then carries out its training phase for M epochs. During this period, backward propagation takes place solely on Bob's side since the weights of all Alice_x have been locked.
3. Evaluation
   1. Once Bob has completed its M epochs of training, the overall training process is considered complete.
   2. All Alice_x can then evaluate the model's performance by making requests to Bob.

Unlike traditional split learning approaches where the weights on the Alice_x side are updated and aligned with the labels, SISA-based split learning only allows for local learning. Once this is complete, the model parameters on Alice_x's side remain locked until Bob's training concludes. This requirement necessitates that Bob's model possesses ample capacity (i.e., a sufficiently large model with "idle weights") to align and adjust in response to a training task.

## Unlearning

The above process can be adjusted to accommodate an unlearning procedure, initiated by a specific Alice_x, as follows:
1. *Local Retraining by Requesting Alice_x*
   1. When an unlearning request is received from Alice_x, she reinitiates local training on her revised local dataset that excludes the specific labels she wishes to eliminate.
   2. This retraining occurs over N epochs. Subsequently, the model weights on Alice_x's side are frozen (locked).
2. *Server Retraining by Bob*
   1. The requesting Alice_x then sends the output of her intermediate layers, along with the updated labels, to Bob. During Bob's retraining phase, Bob continues to receive outputs from the intermediate layers of all Alice_x instances, as normal.
   2. However, during Bob's subsequent M epochs of retraining, backward propagation is prohibited from reaching the side of any Alice_x. It is constrained to occur solely on Bob's side.
3. *Minimal Impact on Other Alice_x Instances*
   1. During this unlearning procedure, the other Alice_x instances do not need to perform anything other than their regular forward propagation.
   2. Only the Alice_x that initiated the unlearning request is required to retrain her local model for a few additional epochs.
4. *Evaluation*
   1. Once Bob completes its M epochs of retraining, the overall training process, including the unlearning procedure, is considered complete.
   2. All Alice_x instances can then evaluate the model's performance by making requests to Bob.

Following the unlearning procedure, Bob's model may exhibit a reduced level of accuracy with respect to the omitted labels, yet it should still maintain its performance on the other labels. This approach ensures the unlearning procedure's impact is localized and minimal, making it an efficient solution for distributed learning environments where privacy and flexibility are crucial.

![Alt text](imgs/sisa_split_nn.jpg?raw=true  "SISA-based Decentralized Split Learning Architecure")

## Example run:
```python split_nn.py --epochs=2 --iterations=2 --world_size=5```

```python split_nn.py --epochs=2 --server_epochs 5 --iterations=2 --world_size=5 --sisa```

```python split_nn.py --epochs=2 --server_epochs 5 --iterations=2 --world_size=5 --sisa --control```

```
Split Learning Initialization

optional arguments:
  -h, --help            show this help message and exit
  --world_size WORLD_SIZE
                        The world size which is equal to 1 server + (world size - 1) clients
  --epochs EPOCHS       The number of epochs to run on the client training each iteration
  --server_epochs SERVER_EPOCHS               
                        (ONLY for SISA) The number of epochs to run on the server training each iteration
  --iterations ITERATIONS
                        The number of iterations to communication between clients and server
  --batch_size BATCH_SIZE
                        The batch size during the epoch training
  --partition_alpha PARTITION_ALPHA
                        Number to describe the uniformity during sampling (heterogenous data generation for LDA)
  --datapath DATAPATH   The folder path to all the local datasets
  --lr LR               Learning rate of local client (SGD)
  --sisa SISA           The trigger to switch over between sisa-based and non-sisa-based
  --control CONTROL     The trigger to run for the control group or not

```