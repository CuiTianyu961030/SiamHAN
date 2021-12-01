import json
import numpy as np
import ipaddress
import random


class DataLoader(object):
    def __init__(self, max_graph_nodes_nb=50, max_feature_dim=50):
        self.source_file = "data/cstnet.json"
        self.max_graph_nodes_nb = max_graph_nodes_nb
        self.max_feature_dim = max_feature_dim
        self.train_val_test_ratio = [0.8, 0.1, 0.1]
        self.scs_adj = np.zeros([])
        self.scs_adj = np.zeros([])
        self.fcf_adj = np.zeros([])
        self.fsf_adj = np.zeros([])
        self.feature = np.zeros([])
        self.mask = np.zeros([])
        self.user_id_label = np.zeros([])

    def build_data(self):
        dataset = []
        f = open(self.source_file, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()

        json_string = ""
        for line in lines:
            if line[:-1] == '{':
                json_string = ""
            json_string += line[:-1]
            if line[:-1] == '}':
                dataset.append(json.loads(json_string))

        # Determine the maximum number of nodes in each user_graph as adj.shape
        graph_nodes_nb = []
        for user_data in dataset:
            client_list = []
            for i in range(int(user_data['nodes']['node_count'])):
                graph_nodes_count = 1
                graph_nodes_count += int(user_data['nodes']['node_'+str(i)]['connection_features']['node_count'])
                graph_nodes_count += len(user_data['nodes']['node_'+str(i)]['client_features'].keys()) - 1
                for j in range(int(user_data['nodes']['node_'+str(i)]['connection_features']['node_count'])):
                    graph_nodes_count += len(user_data['nodes']['node_'+str(i)]
                                             ['connection_features']['node_'+str(j)].keys()) - 1
                client_list.append(graph_nodes_count)
            graph_nodes_nb.append(client_list)

        # Remove outlier data to keep matrix not too sparse
        outlier_index = []
        for i, nbs in enumerate(graph_nodes_nb):
            for nb in nbs:
                if nb > self.max_graph_nodes_nb:
                    outlier_index.append(i)
        for i in sorted(outlier_index, reverse=True):
            dataset.pop(i)
            graph_nodes_nb.pop(i)

        # adj_shape = max([max(nb) for nb in graph_nodes_nb])
        adj_shape = 49
        clients_nb = sum([len(nb) for nb in graph_nodes_nb])
        user_nb = len(graph_nodes_nb)

        print("----- User data info -----")
        print("max graph nb.node limit: ", self.max_graph_nodes_nb)
        print("max feature dim limit: ", self.max_feature_dim)
        print("nb.user: ", user_nb)
        print("nb.client node: ", clients_nb)
        print("max graph nb.node: ", adj_shape)

        # Build adj matrix for each graph (SCS, FCF, FSF)
        scs_adj = np.zeros((clients_nb, adj_shape, adj_shape))
        fcf_adj = np.zeros((clients_nb, adj_shape, adj_shape))
        fsf_adj = np.zeros((clients_nb, adj_shape, adj_shape))

        client_index = 0
        for user_data in dataset:
            for i in range(int(user_data['nodes']['node_count'])):

                # SCS
                read_node_index = 0
                read_node_index += 1
                server_nb = int(user_data['nodes']['node_'+str(i)]['connection_features']['node_count'])
                for j in range(read_node_index, read_node_index + server_nb):
                    scs_adj[client_index+i][0][j] = 1.0
                    scs_adj[client_index+i][j][0] = 1.0
                read_node_index += server_nb

                # FCF
                client_field_nb = len(user_data['nodes']['node_' + str(i)]['client_features'].keys()) - 1
                for j in range(read_node_index, read_node_index + client_field_nb):
                    fcf_adj[client_index+i][0][j] = 1.0
                    fcf_adj[client_index+i][j][0] = 1.0
                read_node_index += client_field_nb

                # FSF
                for j in range(server_nb):
                    server_field_nb = len(user_data['nodes']['node_'+str(i)]
                                          ['connection_features']['node_'+str(j)].keys()) - 1
                    for k in range(read_node_index, read_node_index + server_field_nb):
                        fsf_adj[client_index+i][j+1][k] = 1.0
                        fsf_adj[client_index+i][k][j+1] = 1.0
                    read_node_index += server_field_nb
            client_index += int(user_data['nodes']['node_count'])

        # Build vocabulary
        vocabulary = []
        vocabulary.extend(['<PAD>', '<UNK>'])
        vocabulary.extend([str(i) for i in range(10)])
        vocabulary.extend([chr(i) for i in range(97, 123)])
        vocabulary.extend([' ', '.', ';', ':', '*', '-', '\'', '/', '(', ')', '&', ',', '_'])

        # token2char_dict = {token: char for token, char in enumerate(vocabulary)}
        char2token_dict = {char: token for token, char in enumerate(vocabulary)}

        # Build node feature
        feature = np.zeros((clients_nb, adj_shape, self.max_feature_dim))
        client_index = 0
        for user_data in dataset:
            for i in range(int(user_data['nodes']['node_count'])):

                # Client node
                read_node_index = 0
                ip = ipaddress.ip_address(user_data['nodes']['node_'+str(i)]['client_features']['ip']).exploded
                node_feature = self.char2token(char2token_dict, ''.join(ip.split(':')))
                feature[client_index+i][0][:len(node_feature)] = node_feature
                read_node_index += 1

                # Server nodes
                server_nb = int(user_data['nodes']['node_'+str(i)]['connection_features']['node_count'])
                for j in range(read_node_index, read_node_index + server_nb):
                    ip = ipaddress.ip_address(user_data['nodes']['node_'+str(i)]
                                              ['connection_features']['node_'+str(j-1)]['ip']).exploded
                    node_feature = self.char2token(char2token_dict, ''.join(ip.split(':')))
                    feature[client_index+i][j][:len(node_feature)] = node_feature
                read_node_index += server_nb

                # Client field nodes
                client_field_nb = len(user_data['nodes']['node_'+str(i)]['client_features'].keys()) - 1
                keys = sorted(list(user_data['nodes']['node_'+str(i)]['client_features'].keys() - ["ip"]))
                for j, key in zip(range(read_node_index, read_node_index + client_field_nb), keys):
                    # if key == 'ciphersuites':
                    if 'ciphersuites' in key:
                        ciphersuites = ''.join(user_data['nodes']['node_'+str(i)]['client_features'][key])
                        node_feature = self.char2token(char2token_dict, ciphersuites)
                    elif key == 'com_method':
                        com_method = user_data['nodes']['node_'+str(i)]['client_features'][key]
                        if str(type(com_method)) == '<class \'str\'>':
                            node_feature = self.char2token(char2token_dict, com_method)
                        else:
                            node_feature = self.char2token(char2token_dict, ''.join(com_method))
                    else:
                        node_feature = self.char2token(char2token_dict,
                                                       user_data['nodes']['node_'+str(i)]['client_features'][key])

                    if len(node_feature) <= self.max_feature_dim:
                        feature[client_index+i][j][:len(node_feature)] = node_feature
                    else:
                        feature[client_index+i][j] = node_feature[:self.max_feature_dim]

                # Server field nodes
                read_node_index += client_field_nb
                for j in range(server_nb):
                    server_field_nb = len(user_data['nodes']['node_'+str(i)]
                                          ['connection_features']['node_'+str(j)].keys()) - 1
                    keys = sorted(list(user_data['nodes']['node_'+str(i)]
                                       ['connection_features']['node_'+str(j)].keys() - ["ip"]))
                    for k, key in zip(range(read_node_index, read_node_index + server_field_nb), keys):
                        if key == 'stream_count':
                            stream_count = str(user_data['nodes']['node_'+str(i)]
                                               ['connection_features']['node_'+str(j)][key])
                            node_feature = self.char2token(char2token_dict, stream_count)
                        elif key == 'first_connection':
                            first_connection = ''.join(user_data['nodes']['node_'+str(i)]
                                                       ['connection_features']['node_'+str(j)][key].split('_'))
                            node_feature = self.char2token(char2token_dict, first_connection)
                        else:
                            node_feature = self.char2token(char2token_dict,
                                                           user_data['nodes']['node_'+str(i)]
                                                           ['connection_features']['node_'+str(j)][key])
                        if len(node_feature) <= self.max_feature_dim:
                            feature[client_index+i][k][:len(node_feature)] = node_feature
                        else:
                            feature[client_index+i][k] = node_feature[:self.max_feature_dim]
                    read_node_index += server_field_nb
            client_index += int(user_data['nodes']['node_count'])

        # Build mask matrix
        mask = np.zeros((clients_nb, adj_shape))
        index = 0
        for user_graphs in graph_nodes_nb:
            for nb in user_graphs:
                mask[index][:nb] = 1
                index += 1

        # Build user id labels
        user_id_label = np.zeros((clients_nb, 1))
        index = 0
        for i, user_graphs in enumerate(graph_nodes_nb):
            for _ in user_graphs:
                user_id_label[index] = i
                index += 1

        # Used for inference
        self.scs_adj, self.fcf_adj, self.fsf_adj, self.feature, self.user_id_label, self.mask\
            = scs_adj, fcf_adj, fsf_adj, feature, user_id_label, mask

        print("----- Build graph data -----")
        print("SCS adj.shape: ", scs_adj.shape)
        print("FCF adj.shape: ", fcf_adj.shape)
        print("FSF adj.shape: ", fsf_adj.shape)
        print("node feature.shape: ", feature.shape)
        print("mask.shape: ", mask.shape)
        print("user_label.shape: ", user_id_label.shape)

        # Used for training
        scs_adj, fcf_adj, fsf_adj, feature, mask, label = self.load_data(scs_adj, fcf_adj, fsf_adj,
                                                                         feature, mask, user_id_label)
        scs_biases, fcf_biases, fsf_biases, feature = self.preprocess(scs_adj, fcf_adj, fsf_adj, feature)

        return scs_biases, fcf_biases, fsf_biases, feature, mask, label

    def load_data(self, scs_adj, fcf_adj, fsf_adj, feature, mask, user_id_label):
        # Keep nb.neg sample = nb.pos sample
        data_pair = []
        label = []

        for i in range(len(user_id_label) - 1):
            for j in range(i + 1, len(user_id_label)):
                data_pair.append((i, j))
                if user_id_label[i] == user_id_label[j]:
                    label.append(1)
                # elif user_id_label[i] != user_id_label[j] and (feature[i][0][:12] == feature[j][0][:12]).all():
                #     label.append(2)
                else:
                    label.append(0)

        label = np.array(label)
        pos_index = np.argwhere(label == 1)
        neg_index = np.argwhere(label == 0)

        nb_samples = 10000  # Change with a suitable number for the dataset

        # test mode - mask this part of codes
        temp_index = pos_index
        for _ in range(int(nb_samples / len(pos_index))-1):
            temp_index = np.concatenate([temp_index, pos_index], axis=0)
        pos_index = temp_index

        neg_index_sample = random.sample(range(0, len(neg_index)), len(pos_index))
        neg_index = neg_index[neg_index_sample]

        # Build train val test data
        pos_pair = [data_pair[pos_index[i][0]] for i in range(len(pos_index))]
        neg_pair = [data_pair[neg_index[i][0]] for i in range(len(neg_index))]
        random.shuffle(pos_pair)
        random.shuffle(neg_pair)

        assert sum(self.train_val_test_ratio) <= 1 and 0 not in self.train_val_test_ratio

        train_pos_nb = int(len(pos_pair) * self.train_val_test_ratio[0])
        val_pos_nb = int(len(pos_pair) * self.train_val_test_ratio[1])
        test_pos_nb = len(pos_pair) - train_pos_nb - val_pos_nb
        train_neg_nb = int(len(neg_pair) * self.train_val_test_ratio[0])
        val_neg_nb = int(len(neg_pair) * self.train_val_test_ratio[1])
        test_neg_nb = len(neg_pair) - train_neg_nb - val_neg_nb

        train_scs_adj = np.zeros((2, train_pos_nb + train_neg_nb, scs_adj.shape[1], scs_adj.shape[2]))
        val_scs_adj = np.zeros((2, val_pos_nb + val_neg_nb, scs_adj.shape[1], scs_adj.shape[2]))
        test_scs_adj = np.zeros((2, test_pos_nb + test_neg_nb, scs_adj.shape[1], scs_adj.shape[2]))

        train_fcf_adj = np.zeros((2, train_pos_nb + train_neg_nb, fcf_adj.shape[1], fcf_adj.shape[2]))
        val_fcf_adj = np.zeros((2, val_pos_nb + val_neg_nb, fcf_adj.shape[1], fcf_adj.shape[2]))
        test_fcf_adj = np.zeros((2, test_pos_nb + test_neg_nb, fcf_adj.shape[1], fcf_adj.shape[2]))

        train_fsf_adj = np.zeros((2, train_pos_nb + train_neg_nb, fsf_adj.shape[1], fsf_adj.shape[2]))
        val_fsf_adj = np.zeros((2, val_pos_nb + val_neg_nb, fsf_adj.shape[1], fsf_adj.shape[2]))
        test_fsf_adj = np.zeros((2, test_pos_nb + test_neg_nb, fsf_adj.shape[1], fsf_adj.shape[2]))

        train_feature = np.zeros((2, train_pos_nb + train_neg_nb, feature.shape[1], feature.shape[2]))
        val_feature = np.zeros((2, val_pos_nb + val_neg_nb, feature.shape[1], feature.shape[2]))
        test_feature = np.zeros((2, test_pos_nb + test_neg_nb, feature.shape[1], feature.shape[2]))

        train_mask = np.zeros((train_pos_nb + train_neg_nb, mask.shape[1]))
        val_mask = np.zeros((val_pos_nb + val_neg_nb, mask.shape[1]))
        test_mask = np.zeros((test_pos_nb + test_neg_nb, mask.shape[1]))

        index = 0
        for graph_1_index, graph_2_index in pos_pair[:train_pos_nb]:
            train_feature[0][index] = feature[graph_1_index]
            train_feature[1][index] = feature[graph_2_index]
            train_scs_adj[0][index] = scs_adj[graph_1_index]
            train_scs_adj[1][index] = scs_adj[graph_2_index]
            train_fcf_adj[0][index] = fcf_adj[graph_1_index]
            train_fcf_adj[1][index] = fcf_adj[graph_2_index]
            train_fsf_adj[0][index] = fsf_adj[graph_1_index]
            train_fsf_adj[1][index] = fsf_adj[graph_2_index]
            train_mask[index] = mask[graph_1_index] \
                if np.sum(mask[graph_1_index]) <= np.sum(mask[graph_2_index]) else mask[graph_2_index]
            index += 1
        for graph_1_index, graph_2_index in neg_pair[:train_neg_nb]:
            train_feature[0][index] = feature[graph_1_index]
            train_feature[1][index] = feature[graph_2_index]
            train_scs_adj[0][index] = scs_adj[graph_1_index]
            train_scs_adj[1][index] = scs_adj[graph_2_index]
            train_fcf_adj[0][index] = fcf_adj[graph_1_index]
            train_fcf_adj[1][index] = fcf_adj[graph_2_index]
            train_fsf_adj[0][index] = fsf_adj[graph_1_index]
            train_fsf_adj[1][index] = fsf_adj[graph_2_index]
            train_mask[index] = mask[graph_1_index] \
                if np.sum(mask[graph_1_index]) <= np.sum(mask[graph_2_index]) else mask[graph_2_index]
            index += 1
        index = 0
        for graph_1_index, graph_2_index in pos_pair[train_pos_nb:train_pos_nb + val_pos_nb]:
            val_feature[0][index] = feature[graph_1_index]
            val_feature[1][index] = feature[graph_2_index]
            val_scs_adj[0][index] = scs_adj[graph_1_index]
            val_scs_adj[1][index] = scs_adj[graph_2_index]
            val_fcf_adj[0][index] = fcf_adj[graph_1_index]
            val_fcf_adj[1][index] = fcf_adj[graph_2_index]
            val_fsf_adj[0][index] = fsf_adj[graph_1_index]
            val_fsf_adj[1][index] = fsf_adj[graph_2_index]
            val_mask[index] = mask[graph_1_index] \
                if np.sum(mask[graph_1_index]) <= np.sum(mask[graph_2_index]) else mask[graph_2_index]
            index += 1
        for graph_1_index, graph_2_index in neg_pair[train_neg_nb:train_neg_nb + val_neg_nb]:
            val_feature[0][index] = feature[graph_1_index]
            val_feature[1][index] = feature[graph_2_index]
            val_scs_adj[0][index] = scs_adj[graph_1_index]
            val_scs_adj[1][index] = scs_adj[graph_2_index]
            val_fcf_adj[0][index] = fcf_adj[graph_1_index]
            val_fcf_adj[1][index] = fcf_adj[graph_2_index]
            val_fsf_adj[0][index] = fsf_adj[graph_1_index]
            val_fsf_adj[1][index] = fsf_adj[graph_2_index]
            val_mask[index] = mask[graph_1_index] \
                if np.sum(mask[graph_1_index]) <= np.sum(mask[graph_2_index]) else mask[graph_2_index]
            index += 1
        index = 0
        for graph_1_index, graph_2_index in pos_pair[-test_pos_nb:]:
            test_feature[0][index] = feature[graph_1_index]
            test_feature[1][index] = feature[graph_2_index]
            test_scs_adj[0][index] = scs_adj[graph_1_index]
            test_scs_adj[1][index] = scs_adj[graph_2_index]
            test_fcf_adj[0][index] = fcf_adj[graph_1_index]
            test_fcf_adj[1][index] = fcf_adj[graph_2_index]
            test_fsf_adj[0][index] = fsf_adj[graph_1_index]
            test_fsf_adj[1][index] = fsf_adj[graph_2_index]
            test_mask[index] = mask[graph_1_index] \
                if np.sum(mask[graph_1_index]) <= np.sum(mask[graph_2_index]) else mask[graph_2_index]
            index += 1
        for graph_1_index, graph_2_index in neg_pair[-test_neg_nb:]:
            test_feature[0][index] = feature[graph_1_index]
            test_feature[1][index] = feature[graph_2_index]
            test_scs_adj[0][index] = scs_adj[graph_1_index]
            test_scs_adj[1][index] = scs_adj[graph_2_index]
            test_fcf_adj[0][index] = fcf_adj[graph_1_index]
            test_fcf_adj[1][index] = fcf_adj[graph_2_index]
            test_fsf_adj[0][index] = fsf_adj[graph_1_index]
            test_fsf_adj[1][index] = fsf_adj[graph_2_index]
            test_mask[index] = mask[graph_1_index] \
                if np.sum(mask[graph_1_index]) <= np.sum(mask[graph_2_index]) else mask[graph_2_index]
            index += 1

        train_label = np.concatenate((np.ones(train_pos_nb), np.zeros(train_neg_nb)))
        val_label = np.concatenate((np.ones(val_pos_nb), np.zeros(val_neg_nb)))
        test_label = np.concatenate((np.ones(test_pos_nb), np.zeros(test_neg_nb)))

        print("----- Build pair samples -----")
        print('nb.pos pair: ', len(pos_pair))
        print('nb.neg pair: ', len(neg_pair))
        print("train val test ratio: ", self.train_val_test_ratio)
        print("nb.train pair: ", len(train_feature[0]))
        print("nb.val pair: ", len(val_feature[0]))
        print("nb.test pair: ", len(test_feature[0]))

        # Shuffle data
        train_index = np.arange(len(train_feature[0]))
        val_index = np.arange(len(val_feature[0]))
        test_index = np.arange(len(test_feature[0]))
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index)

        for i in [0, 1]:
            train_scs_adj[i] = train_scs_adj[i][train_index]
            train_fcf_adj[i] = train_fcf_adj[i][train_index]
            train_fsf_adj[i] = train_fsf_adj[i][train_index]
            train_feature[i] = train_feature[i][train_index]

            val_scs_adj[i] = val_scs_adj[i][val_index]
            val_fcf_adj[i] = val_fcf_adj[i][val_index]
            val_fsf_adj[i] = val_fsf_adj[i][val_index]
            val_feature[i] = val_feature[i][val_index]

            test_scs_adj[i] = test_scs_adj[i][test_index]
            test_fcf_adj[i] = test_fcf_adj[i][test_index]
            test_fsf_adj[i] = test_fsf_adj[i][test_index]
            test_feature[i] = test_feature[i][test_index]

        train_mask = train_mask[train_index]
        val_mask = val_mask[val_index]
        test_mask = test_mask[test_index]
        train_label = train_label[train_index]
        val_label = val_label[val_index]
        test_label = test_label[test_index]

        return [train_scs_adj, val_scs_adj, test_scs_adj], [train_fcf_adj, val_fcf_adj, test_fcf_adj], \
            [train_fsf_adj, val_fsf_adj, test_fsf_adj], [train_feature, val_feature, test_feature], \
            [train_mask, val_mask, test_mask], [train_label, val_label, test_label]

    def preprocess(self, scs_adj, fcf_adj, fsf_adj, feature):
        scs_biases = [np.array([self.adj_to_bias(pair, [pair.shape[1]]*pair.shape[0], nhood=1)
                                for pair in split_data]) for split_data in scs_adj]
        fcf_biases = [np.array([self.adj_to_bias(pair, [pair.shape[1]]*pair.shape[0], nhood=1)
                                for pair in split_data]) for split_data in fcf_adj]
        fsf_biases = [np.array([self.adj_to_bias(pair, [pair.shape[1]]*pair.shape[0], nhood=1)
                                for pair in split_data]) for split_data in fsf_adj]
        feature = [np.array([[self.preprocess_features(feature_graph) for feature_graph in pair]
                             for pair in split_data]) for split_data in feature]
        return scs_biases, fcf_biases, fsf_biases, feature

    def char2token(self, vocabulary_dict, field_value):
        feature_value = []
        for c in field_value:
            if c not in vocabulary_dict.keys():
                feature_value.append(vocabulary_dict['<UNK>'])
            else:
                feature_value.append(vocabulary_dict[c])
        return feature_value

    def adj_to_bias(self, adj, sizes, nhood=1):
        """
         Prepare adjacency matrix by expanding up to a given neighbourhood.
         This will insert loops on every node.
         Finally, the matrix is converted to bias vectors.
         Expected shape: [graph, nodes, nodes]
        """
        nb_graphs = adj.shape[0]
        mt = np.empty(adj.shape)
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])
            for _ in range(nhood):
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
        return -1e9 * (1.0 - mt)

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = r_mat_inv.dot(features)
        return features
