from __future__ import division
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from dataloader import DataLoader
from models.siamhan import SiamHAN
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

checkpt_file = 'pre_trained/siamhan-pretrain.ckpt'
checkpt_meta_file = 'pre_trained/siamhan-pretrain.ckpt.meta'

attn_hid_units = [256, 256]  # numbers of hidden units per each attention head in each layer
dense_hid_units = [768, 96]
n_heads = [4, 4, 6]  # additional entry for the output layer
residual = True
nonlinearity = tf.nn.elu
model = SiamHAN()

if __name__ == '__main__':

    data_loader = DataLoader()
    data_loader.build_data()
    scs_adj, fcf_adj, fsf_adj, feature, user_id_label, mask = \
        data_loader.scs_adj, data_loader.fcf_adj, data_loader.fsf_adj, data_loader.feature, data_loader.user_id_label, data_loader.mask

    scs_biases = data_loader.adj_to_bias(scs_adj, [scs_adj.shape[1]] * scs_adj.shape[0], nhood=1)
    fcf_biases = data_loader.adj_to_bias(fcf_adj, [fcf_adj.shape[1]] * fcf_adj.shape[0], nhood=1)
    fsf_biases = data_loader.adj_to_bias(fsf_adj, [fsf_adj.shape[1]] * fsf_adj.shape[0], nhood=1)
    address = feature[:, 0, :32]
    feature = np.array([data_loader.preprocess_features(feature_graph) for feature_graph in feature])

    # Rebuild labels
    history_label = 0
    addr_list = []
    addro_list = []
    del_label = []
    del_index = []
    index_list = []
    count = 0
    new_label = []
    for i, (addr, label) in enumerate(zip(address, user_id_label)):
        if label > history_label:
            if len(set(addr_list)) > 1:
                del_label.append(history_label)
                del_index.extend(index_list)
            else:
                label_list = []
                for _ in range(len(addr_list)):
                    label_list.append(count)
                new_label.extend(label_list)
                count += 1
            addr_list = []
            addro_list = []
            index_list = []
        addr_list.append(addr[:12].tostring())
        addro_list.append(addr)
        index_list.append(i)
        history_label = label
    new_label.append(count)

    del_index = np.array(del_index)
    scs_biases = np.delete(scs_biases, del_index, axis=0)
    fcf_biases = np.delete(fcf_biases, del_index, axis=0)
    fsf_biases = np.delete(fsf_biases, del_index, axis=0)
    mask = np.delete(mask, del_index, axis=0)
    address = np.delete(address, del_index, axis=0)
    feature = np.delete(feature, del_index, axis=0)
    user_id_label = np.array(new_label)

    saver = tf.train.import_meta_graph(checkpt_meta_file)

    with tf.Session() as sess:
        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))

        graph = tf.get_default_graph()
        ftr_in_list = [[graph.get_operation_by_name('input/ftr_in' + i + j).outputs[0]
                        for j in ['_0', '_1', '_2']] for i in ['_1', '_2']]
        bias_in_list = [[graph.get_operation_by_name('input/bias_in' + i + j).outputs[0]
                         for j in ['_0', '_1', '_2']] for i in ['_1', '_2']]
        attn_drop = graph.get_operation_by_name('input/attn_drop').outputs[0]
        ffd_drop = graph.get_operation_by_name('input/ffd_drop').outputs[0]
        is_train = graph.get_operation_by_name('input/is_train').outputs[0]
        lbl_in = graph.get_operation_by_name('input/lbl_in').outputs[0]
        msk_in = graph.get_operation_by_name('input/msk_in').outputs[0]

        nb_graph = feature.shape[0]
        nb_nodes = feature.shape[1]

        feed_dict = {
            lbl_in: np.zeros([1]),
            msk_in: np.zeros([1, nb_nodes]),
            is_train: False,
            attn_drop: 0.0,
            ffd_drop: 0.0
        }

        acc = []
        hit_nb = []
        auc_list = []

        for tracking_nb in [100, 200, 300]:
            tracking_label = random.sample(range(0, user_id_label[-1]), tracking_nb)

            index = np.array([np.argwhere(user_id_label == label)[0][0] for label in tracking_label])
            label_index = []
            for label in tracking_label:
                label_index.append(np.argwhere(user_id_label == label))

            tracking_scs_biases = scs_biases[index]
            tracking_fcf_biases = fcf_biases[index]
            tracking_fsf_biases = fsf_biases[index]
            tracking_feature = feature[index]
            tracking_address = address[index]
            tracking_mask = mask[index]

            margin = 5
            nb_total_true = 0

            score = []
            score_label = []

            count = 0
            for i in range(len(tracking_label)):
                nb_true = 0
                for j in range(nb_graph):
                    if j != index[i]:
                        count += 1
                        feature_list = [[feature[j], feature[j], feature[j]],
                                        [tracking_feature[i], tracking_feature[i], tracking_feature[i]]]
                        biases_list = [[scs_biases[j], fcf_biases[j], fsf_biases[j]],
                                       [tracking_scs_biases[i], tracking_fcf_biases[i], tracking_fsf_biases[i]]]
                        for k in [0, 1]:
                            feed_dict.update(
                                {name: np.expand_dims(data, axis=0) for name, data in
                                 zip(ftr_in_list[k], feature_list[k])}
                            )
                            feed_dict.update(
                                {name: np.expand_dims(data, axis=0) for name, data in
                                 zip(bias_in_list[k], biases_list[k])}
                            )
                            feed_dict.update(
                                {msk_in: np.expand_dims(mask[j], axis=0) if np.sum(mask[j]) <= np.sum(tracking_mask[i])
                                else np.expand_dims(tracking_mask[i], axis=0)}
                            )
                        if (tracking_address[i][:12] == address[j][:12]).all():
                            distance = sess.run('output/distance:0', feed_dict=feed_dict)
                            score.append(distance)
                            if [j] not in label_index[i]:
                                score_label.append(0)
                            else:
                                score_label.append(1)
                        else:
                            distance = np.inf  # Different prefix (inference speed optimization)

                        if ([j] not in label_index[i] and distance >= margin) or (
                                [j] in label_index[i] and distance < margin):
                            nb_true += 1
                nb_total_true += nb_true

            score = 1 - np.array(score) / max(score)
            auc = roc_auc_score(score_label, score)
            acc = nb_total_true / ((nb_graph - 1) * len(tracking_label))

            print('Tracking %s users: auc = %.5f, acc = %.5f' % (tracking_nb, auc, acc))
