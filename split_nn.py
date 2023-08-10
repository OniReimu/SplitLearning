import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import os
import argparse
from data.mnist_flat.mnist_flat_generator import load_mnist_image

# Set the limit of descriptors allocated for opening files by this python process
import resource
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

# Conditionally import classes
def get_alice_and_bob(args):
    if args.vanilla:
        from data_entities_vanilla import alice, bob
    elif args.concat:
        from data_entities_sisa_concat import alice, bob
    elif args.sisa:
        from data_entities_vanilla_sisa import alice, bob
    else: # Default case
        from data_entities import alice, bob

    return alice, bob

def init_env(args):
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5689"

    alice, bob = get_alice_and_bob(args)

    return alice, bob

def example(rank,world_size,args):
    _, bob = init_env(args)
    if rank == 0:
        rpc.init_rpc("bob", rank=rank, world_size=world_size)

        BOB = bob(args)
    
        unlearn_request_from_alices = [1]
        omit_label = 9

        if not args.control:
            if not args.sisa:
                for iter in range(args.iterations):
                    for client_id in range(1,world_size):
                        print(f"Training client {client_id}")
                        BOB.train_request(client_id)
                    # BOB.eval_request()
                    BOB.eval_request_breakdown(omit_label)


                # Test on the control group
                for iter in range(args.iterations):
                    BOB.unlearn_request(client_id=unlearn_request_from_alices[0], omit_label=omit_label)
                    BOB.eval_request_breakdown(omit_label)


            else:
                if args.concat:
                    print(f"Training all clients in parallel")
                    BOB.train_request_parallel()

                    BOB.freeze_alice_weights(range(1, args.client_num_in_total + 1))

                    print("Training server")
                    BOB.train_and_backward(unlearn_request_from_alices=[], unlearn_id=None)         
                    BOB.eval_request_breakdown(omit_label)

                else:
                    print(f"Training all clients in parallel")
                    BOB.train_request_parallel()                    

                    # for iter in range(args.iterations):
                    #     for client_id in range(1,world_size):
                    #         print(f"Training client {client_id}")
                    #         BOB.train_request(client_id)       

                    BOB.freeze_alice_weights(range(1, args.client_num_in_total + 1))

                    print("Training server")
                    BOB.train_and_backward(unlearn_request_from_alices=[], unlearn_id=None)         
                    BOB.eval_request_breakdown(omit_label)

                    # for iter in range(args.iterations):
                    #     print("Training server")
                    #     BOB.train_and_backward(unlearn_request_from_alices=[], unlearn_id=None)
                    #     # BOB.eval_request()
                    #     BOB.eval_request_breakdown(omit_label)

                    #-------------------Unlearn----------------------#

                    BOB.unfreeze_alice_weights(unlearn_request_from_alices)

                    print(f"Retraining client {unlearn_request_from_alices}")
                    BOB.unlearn_request(client_id=unlearn_request_from_alices[0], omit_label=omit_label)

                    # for iter in range(args.iterations):
                        # print(f"Retraining client {unlearn_request_from_alices}")
                        # BOB.unlearn_request(client_id=unlearn_request_from_alices[0], omit_label=omit_label)

                    BOB.freeze_alice_weights(unlearn_request_from_alices)

                    print("Retraining server upon the omitted labels")
                    BOB.train_and_backward(unlearn_request_from_alices, unlearn_id=omit_label)
                    BOB.eval_request_breakdown(omit_label)

                    # for iter in range(args.iterations):
                    #     print("Retraining server upon the omitted labels")
                    #     BOB.train_and_backward(unlearn_request_from_alices, unlearn_id=omit_label)
                    #     BOB.eval_request_breakdown(omit_label)

                    #-------------------------------------------------#
                
        else:
            #-----------------Control group-------------------#

            for iter in range(args.iterations):
                for client_id in range(1,world_size):
                    print(f"(Control group) Training client {client_id}")
                    if client_id in unlearn_request_from_alices:
                        BOB.train_request_control(client_id, omit_label)
                    else:
                        BOB.train_request(client_id)

            BOB.freeze_alice_weights(range(1, args.client_num_in_total + 1))

            for iter in range(args.iterations):
                print("(Control group) Training server")
                BOB.train_and_backward(unlearn_request_from_alices, unlearn_id=omit_label)
                BOB.eval_request()            
            
            #-------------------------------------------------#

        rpc.shutdown()
    else:
        rpc.init_rpc(f"alice{rank}", rank=rank, world_size=world_size)
        rpc.shutdown()

if __name__ == "__main__":
    log_dir = os.path.join(os.path.expanduser('~'), 'Documents/GitHub/SplitLearning/logs/')
    os.makedirs(log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--epochs',type=int,default=1,help='The number of epochs to run on the client training each iteration')
    parser.add_argument('--iterations',type=int,default=5,help='The number of iterations to communication between clients and server')
    parser.add_argument('--batch_size',type=int,default=16,help='The batch size during the epoch training')
    parser.add_argument('--partition_alpha',type=float,default=0.5,help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')
    parser.add_argument('--datapath',type=str,default="data/mnist_flat",help='folder path to all the local datasets')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate of local client (SGD)')
    
    parser.add_argument('--server_epochs',type=int,default=3,help='The number of epochs to run on the server training each iteration')
    
    parser.add_argument('--vanilla', action='store_true', help='The trigger to switch over vanilla or U-shape split learning')
    parser.add_argument('--sisa', action='store_true', help='The trigger to switch over sisa and non-sisa')
    parser.add_argument('--concat', action='store_true', help='The trigger to use a concat layer')
    parser.add_argument('--control', action='store_true', help='The trigger to run for the control group or not')
    
    args = parser.parse_args()
    if args.concat and not args.sisa:
        raise ValueError("The --concat option can only be used with the --sisa option.")
    if args.vanilla and (args.sisa or args.concat):
        raise ValueError("The --vanilla option cannot be used with --sisa or --concat.")
    if args.control and (args.vanilla or args.sisa or args.concat):
        raise ValueError("The --control option cannot be used with any other options.")

    args.client_num_in_total = args.world_size - 1


    load_mnist_image(args)

    world_size = args.world_size

    mp.spawn(example,
             args=(world_size,args),
             nprocs=world_size,
             join=True)