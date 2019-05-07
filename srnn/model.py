import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NodeRNN(nn.Module):
    """
    Class representing human Node RNNs in the st-graph
    """

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(NodeRNN, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        # Store required sizes
        self.rnn_size = args.node_rnn_size  # 128
        self.output_size = args.node_output_size  # 5
        self.embedding_size = args.node_embedding_size  # 64
        self.input_size = args.node_input_size  # 3
        self.edge_rnn_size = args.edge_rnn_size  # 128

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to embed edgeRNN hidden states
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(
            self.edge_rnn_size * 2, self.embedding_size
        )

        # The LSTM cell
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, node_type, h_temporal, h_spatial_other, h, c):
        """
        Forward pass for the model
        params:
        pos : input position  #[5,2]
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node #  [5,256]
        h_spatial_other : output of the attention module    # [5, 256]
        h : hidden state of the current nodeRNN    #[5,128]
        c : cell state of the current nodeRNN #[5,128]
        """
        # Encode the input position
        new_pos = torch.cat(
            (pos, (torch.ones(pos.shape[0], 1) * node_type).cuda()), dim=1
        )
        encoded_input = self.encoder_linear(new_pos)
        encoded_input = self.relu(encoded_input)
        # encoded_input = self.dropout(encoded_input)  # [5,128]

        # Concat both the embeddings
        h_edges = torch.cat((h_temporal, h_spatial_other), 1)  # [3, 256]
        h_edges = self.edge_attention_embed(h_edges)  # [3,64]
        h_edges_embedded = self.relu(h_edges)
        # h_edges_embedded = self.dropout(h_edges_embedded)  # [5,128]

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), 1)  # [3, 128]

        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        # out = self.output_linear(h_new)
        return h_new, c_new
        # return out, h_new, c_new


class SuperNodeRNN(nn.Module):
    """
    Class representing human Node RNNs in the st-graph
    """

    def __init__(self, args):
        super(SuperNodeRNN, self).__init__()

        self.args = args
        self.use_cuda = args.use_cuda

        # Store required sizes
        self.edge_rnn_size = args.edge_rnn_size
        self.rnn_size = args.node_rnn_size
        self.embedding_size = args.node_embedding_size  # 64
        self.input_size = 64

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.lstm_input_size = self.embedding_size + self.edge_rnn_size
        self.cell = nn.LSTMCell(self.lstm_input_size, self.rnn_size)

    def forward(self, weighted_supernode_f_u_ped, h_uu, h, c):
        # Encode the input position
        weighted_supernode_f_u_ped = weighted_supernode_f_u_ped.unsqueeze(0)
        encoded_input = self.encoder_linear(weighted_supernode_f_u_ped)
        encoded_input = self.relu(encoded_input)
        encoded_input = torch.cat((encoded_input, h_uu), dim=1)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new


class EdgeRNN(nn.Module):
    """
    Class representing the Human-Human Edge RNN in the s-t graph
    """

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(EdgeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.edge_rnn_size  # 128
        self.embedding_size = args.edge_embedding_size  # 64
        self.input_size = args.edge_input_size  # 12

        # Linear layer to embed input
        self.encoder_c_ij = nn.Linear(2, 1)
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, c_ij_ori, h, c):
        """
        Forward pass for the model
        params:
        inp : input edge features     
        h : hidden state of the current edgeRNN  
        c : cell state of the current edgeRNN
        """
        # Encode the input position
        encoded_cij = F.sigmoid(self.encoder_c_ij(c_ij_ori))
        new_inp = torch.cat((inp, encoded_cij), dim=1)
        encoded_input = self.encoder_linear(new_inp)
        encoded_input = self.relu(encoded_input)
        # encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new


class SuperNodeEdgeRNN(nn.Module):
    """
    Class representing the Human-Human Edge RNN in the s-t graph
    """

    def __init__(self, args):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(SuperNodeEdgeRNN, self).__init__()

        self.args = args
        # Store required sizes
        self.rnn_size = args.edge_rnn_size
        self.embedding_size = args.edge_embedding_size
        self.input_size = 64

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):
        """
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        """
        new_inp = inp.unsqueeze(0)
        # Encode the input position
        encoded_input = self.encoder_linear(new_inp)
        encoded_input = self.relu(encoded_input)
        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))
        return h_new, c_new


class EdgeAttention(nn.Module):
    """
    Class representing the attention module
    """

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(EdgeAttention, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.edge_rnn_size = args.edge_rnn_size  # 128
        self.node_rnn_size = args.node_rnn_size  # 64
        self.attention_size = args.attention_size  # 64

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.Linear(self.edge_rnn_size, self.attention_size)

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer = nn.Linear(self.edge_rnn_size, self.attention_size)

    def forward(self, h_temporal, h_spatials):
        """
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN     #[1, 128]
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.  #[9, 128]
        """
        # Number of spatial edges
        num_edges = h_spatials.size()[0]

        # Embed the temporal edgeRNN hidden state
        temporal_embed = self.temporal_edge_layer(h_temporal)
        temporal_embed = temporal_embed.squeeze(0)

        # Embed the spatial edgeRNN hidden states
        spatial_embed = self.spatial_edge_layer(h_spatials)

        # Dot based attention
        attn = torch.mv(spatial_embed, temporal_embed)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)
        # Softmax
        attn = torch.nn.functional.softmax(attn)

        # Compute weighted value
        weighted_value = torch.mv(torch.t(h_spatials), attn)

        return weighted_value, attn


class SRNN(nn.Module):
    """
    Class representing the SRNN model
    """

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(SRNN, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if self.infer:
            # Test time
            self.seq_length = 1
            self.obs_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length
            self.obs_length = args.seq_length - args.pred_length

        # Store required sizes
        self.node_rnn_size = args.node_rnn_size
        self.edge_rnn_size = args.edge_rnn_size
        self.output_size = args.node_output_size  # 5

        # Initialize the Node and Edge RNNs
        self.pedNodeRNN = NodeRNN(args, infer)
        self.bicNodeRNN = NodeRNN(args, infer)
        self.carNodeRNN = NodeRNN(args, infer)

        self.EdgeRNN_spatial = EdgeRNN(args, infer)
        self.pedEdgeRNN_temporal = EdgeRNN(args, infer)
        self.bycEdgeRNN_temporal = EdgeRNN(args, infer)
        self.carEdgeRNN_temporal = EdgeRNN(args, infer)

        self.pedSuperNodeEdgeRNN = SuperNodeEdgeRNN(args)
        self.bycSuperNodeEdgeRNN = SuperNodeEdgeRNN(args)
        self.carSuperNodeEdgeRNN = SuperNodeEdgeRNN(args)

        self.pedSuperNodeRNN = SuperNodeRNN(args)
        self.bycSuperNodeRNN = SuperNodeRNN(args)
        self.carSuperNodeRNN = SuperNodeRNN(args)

        self.linear_embeding = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 5)
        self.relu = nn.ReLU()

        # Initialize attention module
        self.attn = EdgeAttention(args, infer)  # Instance Layer
        # self.node_attn = NodeAttention(args)   # Category Layer

    def final_instance_node_output(self, ped_h_nodes, h_u):
        h_u = h_u.repeat((ped_h_nodes.shape[0], 1))
        h_concate = torch.cat((ped_h_nodes, h_u), dim=1)
        output = self.linear_embeding(h_concate)
        output = self.relu(output)
        h2_mt = output
        out = self.output_layer(output)
        return out, h2_mt

    def forward(
        self,
        nodes,
        edges,
        nodesPresent,
        edgesPresent,
        hidden_states_node_RNNs,
        hidden_states_edge_RNNs,
        cell_states_node_RNNs,
        cell_states_edge_RNNs,
        hidden_states_super_node_RNNs,
        hidden_states_super_node_Edge_RNNs,
        cell_states_super_node_RNNs,
        cell_states_super_node_Edge_RNNs,
    ):
        """
        Forward pass for the model
        params:
        nodes : input node features
        edges : input edge features
        nodesPresent : A list of lists, of size seq_length
        Each list contains the nodeIDs that are present in the frame
        edgesPresent : A list of lists, of size seq_length
        Each list contains tuples of nodeIDs that have edges in the frame
        hidden_states_node_RNNs : A tensor of size          numNodes x node_rnn_size
        Contains hidden states of the node RNNs
        hidden_states_edge_RNNs : A tensor of size          numNodes x numNodes x edge_rnn_size
        Contains hidden states of the edge RNNs

        returns:
        outputs : A tensor of shape seq_length x numNodes x 5
        Contains the predictions for next time-step
        hidden_states_node_RNNs
        hidden_states_edge_RNNs
        """
        # Get number of nodes
        numNodes = nodes.size()[1]
        # Initialize output array
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:
            outputs = outputs.cuda()

        # Data structure to store attention weights
        attn_weights = [{} for _ in range(self.seq_length)]

        weighted_supernode_f_u_ped = torch.zeros(self.node_rnn_size).cuda()
        weighted_supernode_f_u_byc = torch.zeros(self.node_rnn_size).cuda()
        weighted_supernode_f_u_car = torch.zeros(self.node_rnn_size).cuda()

        # For each frame   #  self.seq_length = 10
        for framenum in range(self.seq_length):
            edgeIDs = edgesPresent[framenum]
            c_ij_ori_spatial = (
                torch.tensor([[t[0], t[1]] for t in edgeIDs if t[0] != t[1]])
                .float()
                .cuda()
            )
            c_ij_ori_temporal_ped = (
                torch.tensor([[t[0], t[1]] for t in edgeIDs if t[2] == "pedestrian/T"])
                .float()
                .cuda()
            )
            c_ij_ori_temporal_byc = (
                torch.tensor([[t[0], t[1]] for t in edgeIDs if t[2] == "bicycle/T"])
                .float()
                .cuda()
            )
            c_ij_ori_temporal_car = (
                torch.tensor([[t[0], t[1]] for t in edgeIDs if t[2] == "car/T"])
                .float()
                .cuda()
            )
            # Separate temporal and spatial edges
            temporal_edges = [x for x in edgeIDs if x[0] == x[1]]
            spatial_edges = [x for x in edgeIDs if x[0] != x[1]]

            # Find the nodes present in the current frame
            nodeIDs = nodesPresent[framenum]

            nodes_current = nodes[framenum]  # [10,26,2]
            edges_current = edges[framenum]  # [676,2]

            # Initialize temporary tensors
            hidden_states_nodes_from_edges_temporal = Variable(
                torch.zeros(numNodes, self.edge_rnn_size)
            )
            hidden_states_nodes_from_edges_spatial = Variable(
                torch.zeros(numNodes, self.edge_rnn_size)
            )
            if self.use_cuda:
                hidden_states_nodes_from_edges_spatial = (
                    hidden_states_nodes_from_edges_spatial.cuda()
                )
                hidden_states_nodes_from_edges_temporal = (
                    hidden_states_nodes_from_edges_temporal.cuda()
                )

            # If there are any edges
            if len(edgeIDs) != 0:
                # Temporal Edges
                if len(temporal_edges) != 0:
                    temporal_edges_id_and_type = [
                        item for item in edgeIDs if item[0] == item[1]
                    ]

                    list_of_temporal_edges_ped = Variable(
                        torch.LongTensor(
                            [
                                x[0] * numNodes + x[0]
                                for x in temporal_edges_id_and_type
                                if x[2] == "pedestrian/T"
                            ]
                        )
                    ).cuda()
                    list_of_temporal_edges_byc = Variable(
                        torch.LongTensor(
                            [
                                x[0] * numNodes + x[0]
                                for x in temporal_edges_id_and_type
                                if x[2] == "bicycle/T"
                            ]
                        )
                    ).cuda()
                    list_of_temporal_edges_car = Variable(
                        torch.LongTensor(
                            [
                                x[0] * numNodes + x[0]
                                for x in temporal_edges_id_and_type
                                if x[2] == "car/T"
                            ]
                        )
                    ).cuda()

                    list_of_temporal_nodes_ped = torch.LongTensor(
                        [x[0] for x in edgeIDs if x[2] == "pedestrian/T"]
                    ).cuda()
                    list_of_temporal_nodes_byc = torch.LongTensor(
                        [x[0] for x in edgeIDs if x[2] == "bicycle/T"]
                    ).cuda()
                    list_of_temporal_nodes_car = torch.LongTensor(
                        [x[0] for x in edgeIDs if x[2] == "car/T"]
                    ).cuda()

                    ped_edges_temporal_start_end = torch.index_select(
                        edges_current, 0, list_of_temporal_edges_ped
                    )
                    byc_edges_temporal_start_end = torch.index_select(
                        edges_current, 0, list_of_temporal_edges_byc
                    )
                    car_edges_temporal_start_end = torch.index_select(
                        edges_current, 0, list_of_temporal_edges_car
                    )
                    ped_hidden_temporal_start_end = torch.index_select(
                        hidden_states_edge_RNNs, 0, list_of_temporal_edges_ped
                    )
                    byc_hidden_temporal_start_end = torch.index_select(
                        hidden_states_edge_RNNs, 0, list_of_temporal_edges_byc
                    )
                    car_hidden_temporal_start_end = torch.index_select(
                        hidden_states_edge_RNNs, 0, list_of_temporal_edges_car
                    )

                    ped_cell_temporal_start_end = torch.index_select(
                        cell_states_edge_RNNs, 0, list_of_temporal_edges_ped
                    )
                    byc_cell_temporal_start_end = torch.index_select(
                        cell_states_edge_RNNs, 0, list_of_temporal_edges_byc
                    )
                    car_cell_temporal_start_end = torch.index_select(
                        cell_states_edge_RNNs, 0, list_of_temporal_edges_car
                    )
                    # Do forward pass through temporaledgeRNN

                    if ped_edges_temporal_start_end.shape[0] > 0:
                        ped_h_temporal, ped_c_temporal = self.pedEdgeRNN_temporal(
                            ped_edges_temporal_start_end,
                            c_ij_ori_temporal_ped,
                            ped_hidden_temporal_start_end,
                            ped_cell_temporal_start_end,
                        )
                        hidden_states_edge_RNNs[
                            list_of_temporal_edges_ped
                        ] = ped_h_temporal
                        cell_states_edge_RNNs[
                            list_of_temporal_edges_ped
                        ] = ped_c_temporal
                        hidden_states_nodes_from_edges_temporal[
                            list_of_temporal_nodes_ped
                        ] = ped_h_temporal

                    if byc_edges_temporal_start_end.shape[0] > 0:
                        byc_h_temporal, byc_c_temporal = self.bycEdgeRNN_temporal(
                            byc_edges_temporal_start_end,
                            c_ij_ori_temporal_byc,
                            byc_hidden_temporal_start_end,
                            byc_cell_temporal_start_end,
                        )
                        hidden_states_edge_RNNs[
                            list_of_temporal_edges_byc
                        ] = byc_h_temporal
                        cell_states_edge_RNNs[
                            list_of_temporal_edges_byc
                        ] = byc_c_temporal
                        hidden_states_nodes_from_edges_temporal[
                            list_of_temporal_nodes_byc
                        ] = byc_h_temporal

                    if car_edges_temporal_start_end.shape[0] > 0:
                        car_h_temporal, car_c_temporal = self.carEdgeRNN_temporal(
                            car_edges_temporal_start_end,
                            c_ij_ori_temporal_car,
                            car_hidden_temporal_start_end,
                            car_cell_temporal_start_end,
                        )
                        hidden_states_edge_RNNs[
                            list_of_temporal_edges_car
                        ] = car_h_temporal
                        cell_states_edge_RNNs[
                            list_of_temporal_edges_car
                        ] = car_c_temporal
                        hidden_states_nodes_from_edges_temporal[
                            list_of_temporal_nodes_car
                        ] = car_h_temporal

                # Spatial Edges
                if len(spatial_edges) != 0:
                    # Get the spatial edges
                    list_of_spatial_edges = Variable(
                        torch.LongTensor(
                            [x[0] * numNodes + x[1] for x in edgeIDs if x[0] != x[1]]
                        )
                    )  # len [90]
                    if self.use_cuda:
                        list_of_spatial_edges = list_of_spatial_edges.cuda()
                    # Get nodes associated with the spatial edges
                    list_of_spatial_nodes = np.array(
                        [x[0] for x in edgeIDs if x[0] != x[1]]
                    )  # len 90

                    # Get the corresponding edge features
                    edges_spatial_start_end = torch.index_select(
                        edges_current, 0, list_of_spatial_edges
                    )  # len edges_current 100 # spatial_edges in current frame
                    # Get the corresponding hidden states
                    hidden_spatial_start_end = torch.index_select(
                        hidden_states_edge_RNNs, 0, list_of_spatial_edges
                    )
                    # Get the corresponding cell states
                    cell_spatial_start_end = torch.index_select(
                        cell_states_edge_RNNs, 0, list_of_spatial_edges
                    )  # [20, 256]

                    # Do forward pass through spatialedgeRNN
                    h_spatial, c_spatial = self.EdgeRNN_spatial(
                        edges_spatial_start_end,
                        c_ij_ori_spatial,
                        hidden_spatial_start_end,
                        cell_spatial_start_end,
                    )

                    # Update the hidden state and cell state
                    hidden_states_edge_RNNs[list_of_spatial_edges] = h_spatial
                    cell_states_edge_RNNs[list_of_spatial_edges] = c_spatial

                    # pass it to attention module
                    # For each node
                    for node in range(numNodes):
                        # Get the indices of spatial edges associated with this node
                        l = np.where(list_of_spatial_nodes == node)[0]
                        if len(l) == 0:
                            # If the node has no spatial edges, nothing to do
                            continue
                        l = torch.LongTensor(l)
                        if self.use_cuda:
                            l = l.cuda()
                        # What are the other nodes with these edges?
                        node_others = [
                            x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]
                        ]
                        h_node = hidden_states_nodes_from_edges_temporal[
                            node
                        ]  # (h_vv)^t in paper graph

                        # Do forward pass through attention module
                        hidden_attn_weighted, attn_w = self.attn(
                            h_node.view(1, -1), h_spatial[l]
                        )
                        # Store the attention weights
                        attn_weights[framenum][node] = (
                            attn_w.data.cpu().numpy(),
                            node_others,
                        )

                        # Store the output of attention module in temporary tensor
                        hidden_states_nodes_from_edges_spatial[
                            node
                        ] = hidden_attn_weighted  # Attention module output

            # If there are nodes in this frame
            if len(nodeIDs) != 0:
                """
                # Get list of nodes
                list_of_nodes = Variable(torch.LongTensor(nodeIDs))
                if self.use_cuda:
                    list_of_nodes = list_of_nodes.cuda()
                list_of_nodes = list_of_nodes[:,0]
                """

                list_of_nodes_ped = Variable(
                    torch.LongTensor([x[0] for x in nodeIDs if int(x[1]) == 1])
                ).cuda()
                list_of_nodes_byc = Variable(
                    torch.LongTensor([x[0] for x in nodeIDs if int(x[1]) == 2])
                ).cuda()
                list_of_nodes_car = Variable(
                    torch.LongTensor([x[0] for x in nodeIDs if int(x[1]) == 3])
                ).cuda()

                # Get their node features
                # nodes_current_selected = torch.index_select(nodes_current, 0, list_of_nodes)  #[5,2]
                ped_nodes_current_selected = torch.index_select(
                    nodes_current, 0, list_of_nodes_ped
                )
                byc_nodes_current_selected = torch.index_select(
                    nodes_current, 0, list_of_nodes_byc
                )
                car_nodes_current_selected = torch.index_select(
                    nodes_current, 0, list_of_nodes_car
                )

                ped_hidden_nodes_current = torch.index_select(
                    hidden_states_node_RNNs, 0, list_of_nodes_ped
                )
                byc_hidden_nodes_current = torch.index_select(
                    hidden_states_node_RNNs, 0, list_of_nodes_byc
                )
                car_hidden_nodes_current = torch.index_select(
                    hidden_states_node_RNNs, 0, list_of_nodes_car
                )

                ped_cell_nodes_current = torch.index_select(
                    cell_states_node_RNNs, 0, list_of_nodes_ped
                )
                byc_cell_nodes_current = torch.index_select(
                    cell_states_node_RNNs, 0, list_of_nodes_byc
                )
                car_cell_nodes_current = torch.index_select(
                    cell_states_node_RNNs, 0, list_of_nodes_car
                )

                # Get the temporal edgeRNN hidden states corresponding to these nodes
                ped_h_temporal_other = hidden_states_nodes_from_edges_temporal[
                    list_of_nodes_ped
                ]
                byc_h_temporal_other = hidden_states_nodes_from_edges_temporal[
                    list_of_nodes_byc
                ]
                car_h_temporal_other = hidden_states_nodes_from_edges_temporal[
                    list_of_nodes_car
                ]
                ped_h_spatial_other = hidden_states_nodes_from_edges_spatial[
                    list_of_nodes_ped
                ]
                byc_h_spatial_other = hidden_states_nodes_from_edges_spatial[
                    list_of_nodes_byc
                ]
                car_h_spatial_other = hidden_states_nodes_from_edges_spatial[
                    list_of_nodes_car
                ]

                if ped_nodes_current_selected.shape[0] > 0:
                    ped_h_nodes, ped_c_nodes = self.pedNodeRNN(
                        ped_nodes_current_selected,
                        1,
                        ped_h_temporal_other,
                        ped_h_spatial_other,
                        ped_hidden_nodes_current,
                        ped_cell_nodes_current,
                    )
                    hidden_states_node_RNNs[list_of_nodes_ped] = ped_h_nodes
                    cell_states_node_RNNs[list_of_nodes_ped] = ped_c_nodes
                    instance_cnt_ped = ped_h_nodes.shape[0]
                    """
                    for k in range(instance_cnt_ped):
                        weighted_supernode_f_u_ped_next_time = (
                            weighted_supernode_f_u_ped_next_time
                            + self.node_attn(
                                ped_h_nodes[k, :].unsqueeze(0), ped_c_nodes
                            )
                        )
                    """
                    weighted_supernode_f_u_ped_next_time = ped_h_nodes * F.softmax(
                        ped_c_nodes
                    )
                    weighted_supernode_f_u_ped_next_time = (
                        torch.sum(weighted_supernode_f_u_ped_next_time, dim=0)
                        / instance_cnt_ped
                    )

                    delta_weighted_supernode_f_u_ped = (
                        weighted_supernode_f_u_ped_next_time
                        - weighted_supernode_f_u_ped
                    )

                    ped_hidden_states_super_node_Edge_RNNs = torch.index_select(
                        hidden_states_super_node_Edge_RNNs, 0, torch.tensor(0).cuda()
                    )
                    ped_cell_states_super_node_Edge_RNNs = torch.index_select(
                        cell_states_super_node_Edge_RNNs, 0, torch.tensor(0).cuda()
                    )

                    h_uu_ped, c_uu_ped = self.pedSuperNodeEdgeRNN(
                        delta_weighted_supernode_f_u_ped,
                        ped_hidden_states_super_node_Edge_RNNs,
                        ped_cell_states_super_node_Edge_RNNs,
                    )

                    hidden_states_super_node_Edge_RNNs[0] = h_uu_ped
                    cell_states_super_node_Edge_RNNs[0] = c_uu_ped

                    weighted_supernode_f_u_ped = weighted_supernode_f_u_ped_next_time

                    ped_hidden_states_super_node_RNNs = torch.index_select(
                        hidden_states_super_node_RNNs, 0, torch.tensor(0).cuda()
                    )
                    ped_cell_states_super_node_RNNs = torch.index_select(
                        cell_states_super_node_RNNs, 0, torch.tensor(0).cuda()
                    )
                    h_u_ped, c_u_ped = self.pedSuperNodeRNN(
                        weighted_supernode_f_u_ped,
                        h_uu_ped,
                        ped_hidden_states_super_node_RNNs,
                        ped_cell_states_super_node_RNNs,
                    )
                    hidden_states_super_node_RNNs[0] = h_u_ped
                    cell_states_super_node_RNNs[0] = c_u_ped
                    output, h2_mt = self.final_instance_node_output(
                        ped_h_nodes, h_u_ped
                    )
                    outputs[framenum * numNodes + list_of_nodes_ped] = output
                    hidden_states_node_RNNs[list_of_nodes_ped] = h2_mt

                if byc_nodes_current_selected.shape[0] > 0:
                    byc_h_nodes, byc_c_nodes = self.bicNodeRNN(
                        byc_nodes_current_selected,
                        2,
                        byc_h_temporal_other,
                        byc_h_spatial_other,
                        byc_hidden_nodes_current,
                        byc_cell_nodes_current,
                    )
                    hidden_states_node_RNNs[list_of_nodes_byc] = byc_h_nodes
                    cell_states_node_RNNs[list_of_nodes_byc] = byc_c_nodes
                    instance_cnt_byc = byc_h_nodes.shape[0]
                    weighted_supernode_f_u_byc_next_time = byc_h_nodes * F.softmax(
                        byc_c_nodes
                    )
                    weighted_supernode_f_u_byc_next_time = (
                        torch.sum(weighted_supernode_f_u_byc_next_time, dim=0)
                        / instance_cnt_byc
                    )
                    delta_weighted_supernode_f_u_byc = (
                        weighted_supernode_f_u_byc_next_time
                        - weighted_supernode_f_u_byc
                    )

                    byc_hidden_states_super_node_Edge_RNNs = torch.index_select(
                        hidden_states_super_node_Edge_RNNs, 0, torch.tensor(1).cuda()
                    )
                    byc_cell_states_super_node_Edge_RNNs = torch.index_select(
                        cell_states_super_node_Edge_RNNs, 0, torch.tensor(1).cuda()
                    )

                    h_uu_byc, c_uu_byc = self.bycSuperNodeEdgeRNN(
                        delta_weighted_supernode_f_u_byc,
                        byc_hidden_states_super_node_Edge_RNNs,
                        byc_cell_states_super_node_Edge_RNNs,
                    )

                    hidden_states_super_node_Edge_RNNs[1] = h_uu_byc
                    cell_states_super_node_Edge_RNNs[1] = c_uu_byc

                    weighted_supernode_f_u_byc = weighted_supernode_f_u_byc_next_time

                    byc_hidden_states_super_node_RNNs = torch.index_select(
                        hidden_states_super_node_RNNs, 0, torch.tensor(1).cuda()
                    )
                    byc_cell_states_super_node_RNNs = torch.index_select(
                        cell_states_super_node_RNNs, 0, torch.tensor(1).cuda()
                    )
                    h_u_byc, c_u_byc = self.bycSuperNodeRNN(
                        weighted_supernode_f_u_byc,
                        h_uu_byc,
                        byc_hidden_states_super_node_RNNs,
                        byc_cell_states_super_node_RNNs,
                    )
                    hidden_states_super_node_RNNs[1] = h_u_byc
                    cell_states_super_node_RNNs[1] = c_u_byc

                    output, h2_mt = self.final_instance_node_output(
                        byc_h_nodes, h_u_byc
                    )
                    outputs[framenum * numNodes + list_of_nodes_byc] = output
                    hidden_states_node_RNNs[list_of_nodes_byc] = h2_mt

                if car_nodes_current_selected.shape[0] > 0:
                    car_h_nodes, car_c_nodes = self.carNodeRNN(
                        car_nodes_current_selected,
                        3,
                        car_h_temporal_other,
                        car_h_spatial_other,
                        car_hidden_nodes_current,
                        car_cell_nodes_current,
                    )
                    hidden_states_node_RNNs[list_of_nodes_car] = car_h_nodes
                    cell_states_node_RNNs[list_of_nodes_car] = car_c_nodes
                    instance_cnt_car = car_h_nodes.shape[0]
                    weighted_supernode_f_u_car_next_time = car_h_nodes * F.softmax(
                        car_c_nodes
                    )
                    weighted_supernode_f_u_car_next_time = (
                        torch.sum(weighted_supernode_f_u_car_next_time, dim=0)
                        / instance_cnt_car
                    )
                    delta_weighted_supernode_f_u_car = (
                        weighted_supernode_f_u_car_next_time
                        - weighted_supernode_f_u_car
                    )

                    car_hidden_states_super_node_Edge_RNNs = torch.index_select(
                        hidden_states_super_node_Edge_RNNs, 0, torch.tensor(2).cuda()
                    )
                    car_cell_states_super_node_Edge_RNNs = torch.index_select(
                        cell_states_super_node_Edge_RNNs, 0, torch.tensor(2).cuda()
                    )
                    h_uu_car, c_uu_car = self.carSuperNodeEdgeRNN(
                        delta_weighted_supernode_f_u_car,
                        car_hidden_states_super_node_Edge_RNNs,
                        car_cell_states_super_node_Edge_RNNs,
                    )  # [1,128]
                    hidden_states_super_node_Edge_RNNs[2] = h_uu_car
                    cell_states_super_node_Edge_RNNs[2] = c_uu_car
                    weighted_supernode_f_u_car = weighted_supernode_f_u_car_next_time

                    car_hidden_states_super_node_RNNs = torch.index_select(
                        hidden_states_super_node_RNNs, 0, torch.tensor(2).cuda()
                    )
                    car_cell_states_super_node_RNNs = torch.index_select(
                        cell_states_super_node_RNNs, 0, torch.tensor(2).cuda()
                    )
                    h_u_car, c_u_car = self.carSuperNodeRNN(
                        weighted_supernode_f_u_car,
                        h_uu_car,
                        car_hidden_states_super_node_RNNs,
                        car_cell_states_super_node_RNNs,
                    )
                    hidden_states_super_node_RNNs[2] = h_u_car
                    cell_states_super_node_RNNs[2] = c_u_car

                    output, h2_mt = self.final_instance_node_output(
                        car_h_nodes, h_u_car
                    )
                    outputs[framenum * numNodes + list_of_nodes_car] = output
                    hidden_states_node_RNNs[list_of_nodes_car] = h2_mt

        # Reshape the outputs carefully
        outputs_return = Variable(
            torch.zeros(self.seq_length, numNodes, self.output_size)
        )
        if self.use_cuda:
            outputs_return = outputs_return.cuda()

        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[
                    framenum * numNodes + node, :
                ]

        # return outputs_return
        return (
            outputs_return,
            hidden_states_node_RNNs,
            hidden_states_edge_RNNs,
            cell_states_node_RNNs,
            cell_states_edge_RNNs,
            hidden_states_super_node_RNNs,
            hidden_states_super_node_Edge_RNNs,
            cell_states_super_node_RNNs,
            cell_states_super_node_Edge_RNNs,
            attn_weights,
        )
