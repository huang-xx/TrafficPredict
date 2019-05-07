import argparse
import logging
import os
import pickle
import time

import torch
from torch.autograd import Variable

from criterion import Gaussian2DLikelihood
from model import SRNN
from st_graph import ST_GRAPH
from utils import DataLoader, set_logger

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    parser = argparse.ArgumentParser()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # RNN size
    parser.add_argument(
        "--node_rnn_size",
        type=int,
        default=64,
        help="Size of Human Node RNN hidden state",
    )
    parser.add_argument(
        "--edge_rnn_size",
        type=int,
        default=128,
        help="Size of Human Human Edge RNN hidden state",
    )

    # Input and output size
    parser.add_argument(
        "--node_input_size", type=int, default=3, help="Dimension of the node features"
    )
    parser.add_argument(
        "--edge_input_size",
        type=int,
        default=3,
        help="Dimension of the edge features, the 3th parameter is set to 10",
    )
    parser.add_argument(
        "--node_output_size", type=int, default=5, help="Dimension of the node output"
    )

    # Embedding size
    parser.add_argument(
        "--node_embedding_size",
        type=int,
        default=64,
        help="Embedding size of node features",
    )
    parser.add_argument(
        "--edge_embedding_size",
        type=int,
        default=64,
        help="Embedding size of edge features",
    )

    # Attention vector dimension
    parser.add_argument("--attention_size", type=int, default=64, help="Attention size")

    # Sequence length
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--pred_length", type=int, default=6, help="Predicted sequence length"
    )

    # Batch size
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    # Number of epochs
    parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs")

    # Gradient value at which it should be clipped
    parser.add_argument(
        "--grad_clip", type=float, default=10.0, help="clip gradients at this value"
    )
    # Lambda regularization parameter (L2)
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=0.00005,
        help="L2 regularization parameter",
    )

    # Learning rate parameter
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate"
    )
    # Decay rate for the learning rate parameter
    parser.add_argument(
        "--decay_rate", type=float, default=0.99, help="decay rate for the optimizer"
    )

    # Dropout rate
    parser.add_argument("--dropout", type=float, default=0, help="Dropout probability")

    # Use GPU or CPU
    parser.add_argument(
        "--use_cuda", action="store_true", default=True, help="Use GPU or CPU"
    )

    args = parser.parse_args()

    train(args)


def train(args):
    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length + 1, forcePreProcess=False)
    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, args.seq_length + 1)

    # Log directory
    log_directory = "../log/"

    # Logging file
    log_file_curve = open(os.path.join(log_directory, "log_curve.txt"), "w")
    log_file = open(os.path.join(log_directory, "val.txt"), "w")

    # Save directory
    save_directory = "../save/"

    # Open the configuration file
    with open(os.path.join(save_directory, "config.pkl"), "wb") as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, "srnn_model_" + str(x) + ".tar")

    # Initialize net
    net = SRNN(args)
    if args.use_cuda:
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)

    # learning_rate = args.learning_rate
    logging.info("Training begin")
    best_val_loss = 100
    best_epoch = 0
    # Training
    for epoch in range(args.num_epochs):
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        # dataloader.num_batches = 10. 1 epoch have 10 batches
        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get batch data
            x, _, _, d = dataloader.next_batch(randomUpdate=True)

            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                # Construct the graph for the current sequence
                stgraph.readGraph([x[sequence]])
                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()
                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float())
                # nodes[0] represent all the person's corrdinate show up in  frame 0.
                if args.use_cuda:
                    nodes = nodes.cuda()
                edges = Variable(torch.from_numpy(edges).float())
                if args.use_cuda:
                    edges = edges.cuda()

                # Define hidden states
                numNodes = nodes.size()[1]

                hidden_states_node_RNNs = Variable(
                    torch.zeros(numNodes, args.node_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_node_RNNs = hidden_states_node_RNNs.cuda()

                hidden_states_edge_RNNs = Variable(
                    torch.zeros(numNodes * numNodes, args.edge_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_edge_RNNs = hidden_states_edge_RNNs.cuda()

                cell_states_node_RNNs = Variable(
                    torch.zeros(numNodes, args.node_rnn_size)
                )
                if args.use_cuda:
                    cell_states_node_RNNs = cell_states_node_RNNs.cuda()

                cell_states_edge_RNNs = Variable(
                    torch.zeros(numNodes * numNodes, args.edge_rnn_size)
                )
                if args.use_cuda:
                    cell_states_edge_RNNs = cell_states_edge_RNNs.cuda()

                hidden_states_super_node_RNNs = Variable(
                    torch.zeros(3, args.node_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_super_node_RNNs = hidden_states_super_node_RNNs.cuda()

                cell_states_super_node_RNNs = Variable(
                    torch.zeros(3, args.node_rnn_size)
                )
                if args.use_cuda:
                    cell_states_super_node_RNNs = cell_states_super_node_RNNs.cuda()

                hidden_states_super_node_Edge_RNNs = Variable(
                    torch.zeros(3, args.edge_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_super_node_Edge_RNNs = (
                        hidden_states_super_node_Edge_RNNs.cuda()
                    )

                cell_states_super_node_Edge_RNNs = Variable(
                    torch.zeros(3, args.edge_rnn_size)
                )
                if args.use_cuda:
                    cell_states_super_node_Edge_RNNs = (
                        cell_states_super_node_Edge_RNNs.cuda()
                    )

                # Zero out the gradients
                net.zero_grad()
                optimizer.zero_grad()
                # Forward prop
                outputs, _, _, _, _, _, _, _, _, _ = net(
                    nodes[: args.seq_length],
                    edges[: args.seq_length],
                    nodesPresent[:-1],
                    edgesPresent[:-1],
                    hidden_states_node_RNNs,
                    hidden_states_edge_RNNs,
                    cell_states_node_RNNs,
                    cell_states_edge_RNNs,
                    hidden_states_super_node_RNNs,
                    hidden_states_super_node_Edge_RNNs,
                    cell_states_super_node_RNNs,
                    cell_states_super_node_Edge_RNNs,
                )

                # Compute loss
                loss = Gaussian2DLikelihood(
                    outputs, nodes[1:], nodesPresent[1:], args.pred_length
                )
                loss_batch += loss.item()
                # embed()
                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

                # Reset the stgraph
                stgraph.reset()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            logging.info(
                "{}/{} (epoch {}), train_loss = {:.12f}, time/batch = {:.12f}".format(
                    epoch * dataloader.num_batches + batch,
                    args.num_epochs * dataloader.num_batches,
                    epoch,
                    loss_batch,
                    end - start,
                )
            )
        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches
        # Log it
        log_file_curve.write(str(epoch) + "," + str(loss_epoch) + ",")

        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        for batch in range(dataloader.valid_num_batches):
            # Get batch data

            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float())
                if args.use_cuda:
                    nodes = nodes.cuda()
                edges = Variable(torch.from_numpy(edges).float())
                if args.use_cuda:
                    edges = edges.cuda()

                # Define hidden states
                numNodes = nodes.size()[1]

                hidden_states_node_RNNs = Variable(
                    torch.zeros(numNodes, args.node_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_node_RNNs = hidden_states_node_RNNs.cuda()

                hidden_states_edge_RNNs = Variable(
                    torch.zeros(numNodes * numNodes, args.edge_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_edge_RNNs = hidden_states_edge_RNNs.cuda()
                cell_states_node_RNNs = Variable(
                    torch.zeros(numNodes, args.node_rnn_size)
                )
                if args.use_cuda:
                    cell_states_node_RNNs = cell_states_node_RNNs.cuda()
                cell_states_edge_RNNs = Variable(
                    torch.zeros(numNodes * numNodes, args.edge_rnn_size)
                )
                if args.use_cuda:
                    cell_states_edge_RNNs = cell_states_edge_RNNs.cuda()

                hidden_states_super_node_RNNs = Variable(
                    torch.zeros(3, args.node_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_super_node_RNNs = hidden_states_super_node_RNNs.cuda()

                cell_states_super_node_RNNs = Variable(
                    torch.zeros(3, args.node_rnn_size)
                )
                if args.use_cuda:
                    cell_states_super_node_RNNs = cell_states_super_node_RNNs.cuda()

                hidden_states_super_node_Edge_RNNs = Variable(
                    torch.zeros(3, args.edge_rnn_size)
                )
                if args.use_cuda:
                    hidden_states_super_node_Edge_RNNs = (
                        hidden_states_super_node_Edge_RNNs.cuda()
                    )

                cell_states_super_node_Edge_RNNs = Variable(
                    torch.zeros(3, args.edge_rnn_size)
                )
                if args.use_cuda:
                    cell_states_super_node_Edge_RNNs = (
                        cell_states_super_node_Edge_RNNs.cuda()
                    )

                outputs, _, _, _, _, _, _, _, _, _ = net(
                    nodes[: args.seq_length],
                    edges[: args.seq_length],
                    nodesPresent[:-1],
                    edgesPresent[:-1],
                    hidden_states_node_RNNs,
                    hidden_states_edge_RNNs,
                    cell_states_node_RNNs,
                    cell_states_edge_RNNs,
                    hidden_states_super_node_RNNs,
                    hidden_states_super_node_Edge_RNNs,
                    cell_states_super_node_RNNs,
                    cell_states_super_node_Edge_RNNs,
                )

                # Compute loss
                loss = Gaussian2DLikelihood(
                    outputs, nodes[1:], nodesPresent[1:], args.pred_length
                )

                loss_batch += loss.item()

                # Reset the stgraph
                stgraph.reset()

            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch = loss_epoch / dataloader.valid_num_batches

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch

        # Record best epoch and best validation loss
        logging.info("(epoch {}), valid_loss = {:.3f}".format(epoch, loss_epoch))
        logging.info(
            "Best epoch {}, Best validation loss {}".format(best_epoch, best_val_loss)
        )
        # Log it
        log_file_curve.write(str(loss_epoch) + "\n")

        # Save the model after each epoch
        logging.info("Saving model")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path(epoch),
        )

    # Record the best epoch and best validation loss overall
    logging.info(
        "Best epoch {}, Best validation loss {}".format(best_epoch, best_val_loss)
    )
    # Log it
    log_file.write(str(best_epoch) + "," + str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()


if __name__ == "__main__":
    set_logger(os.path.join("./", "train.log"))
    main()
