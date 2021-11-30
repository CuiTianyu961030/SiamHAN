from __future__ import division
from dataloader import DataLoader
from models.siamhan import SiamHAN
import numpy as np
from collections import Counter
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
    scs_adj, fcf_adj, fsf_adj, feature, user_id_label = \
        data_loader.scs_adj, data_loader.fcf_adj, data_loader.fsf_adj, data_loader.feature, data_loader.user_id_label

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

        user_inference = {0: [0]}
        history_max_id = 0
        margin = 3.5
        nb_false_create = 0  # pred increase but label doesn't increase
        nb_false_no_create = 0  # label increase but pred doesn't increase
        nb_false = 0
        nb_true = 1
        history_user_id = 0
        history_pred_id = 0
        history_accuracy = 0
        false_created_distance = []
        false_classified_distance = []

        former_pred_list = [0]
        nb_former_label = 1
        nb_true_new = 0
        nb_total_new = 0
        history_max_pred = 0

        acc = []
        hit_nb = []

        for i in range(1, nb_graph):
            distance = []
            print('----- Round %s -----' % i)
            for j in range(0, i):
                # print(i, j)
                feature_list = [[feature[j], feature[j], feature[j]],
                                [feature[i], feature[i], feature[i]]]
                biases_list = [[scs_biases[j], fcf_biases[j], fsf_biases[j]],
                               [scs_biases[i], fcf_biases[i], fsf_biases[i]]]
                for k in [0, 1]:
                    feed_dict.update(
                        {name: np.expand_dims(data, axis=0) for name, data in zip(ftr_in_list[k], feature_list[k])}
                    )
                    feed_dict.update(
                        {name: np.expand_dims(data, axis=0) for name, data in zip(bias_in_list[k], biases_list[k])}
                    )
                if (address[i][:12] == address[j][:12]).all():
                    distance.append(sess.run('output/distance:0', feed_dict=feed_dict))
                else:
                    distance.append(np.inf)  # Different prefix (inference speed optimization)

            distance = np.array(distance)
            print('raw_distance', distance)
            for user_id, node_id in user_inference.items():
                distance[node_id] = np.max(distance[node_id])
            print('update_distance', distance)
            closet_node = np.argmin(distance)
            print('closet_node_id', closet_node)
            if distance[closet_node] >= margin:
                history_max_id += 1
                user_inference[history_max_id] = [i]
                pred = history_max_id
            else:
                for user_id, node_id in user_inference.items():
                    if closet_node in node_id:
                        user_inference[user_id].append(i)
                        pred = user_id
                        break

            # print('label', user_id_label[i])
            print('fixed_label', user_id_label[i] + nb_false_create - nb_false_no_create)
            print('predict', pred)
            print('finished_list', user_inference.items())

            if pred > user_id_label[i] + nb_false_create - nb_false_no_create:
                nb_false += 1
                false_created_distance.append(float(distance[-1]))
                print('Classification result = false created')
            elif pred < user_id_label[i] + nb_false_create - nb_false_no_create:
                nb_false += 1
                false_classified_distance.append(float(distance[closet_node]))
                print('Classification result = false classified')
            else:
                nb_true += 1
                print('Classification result = true')

            # print('false created distance', false_created_distance)
            # print('false classified distance', false_classified_distance)

            if user_id_label[i] == history_user_id + 1 and pred != history_pred_id + 1:
                nb_false_no_create += 1
            if user_id_label[i] != history_user_id + 1 and pred == history_pred_id + 1:
                nb_false_create += 1

            if user_id_label[i] > history_user_id:
                maxNum_sample = np.array(Counter(former_pred_list).most_common(len(set(former_pred_list))))
                # print(maxNum_sample)

                maxNum_sample = maxNum_sample[maxNum_sample[:, 0] >= history_max_pred]
                if len(maxNum_sample) == 0:
                    nb_true_new += 0
                else:
                    maxNum_sample = maxNum_sample[maxNum_sample[:, 1] == maxNum_sample[0][1]][-1]
                    nb_true_new += maxNum_sample[1]
                nb_total_new += nb_former_label
                accuracy_new = nb_true_new / nb_total_new
                print('Current acc = %.5f' % accuracy_new)
                former_pred_list = []
                nb_former_label = 0
                history_max_pred = history_max_id
            former_pred_list.append(pred)
            nb_former_label += 1

            nb_total = i + 1
            accuracy = nb_true / nb_total
            false_rate = nb_false / nb_total

            if history_accuracy > accuracy:
                print('Accuracy decline')
            else:
                print('Accuracy increase')

            history_user_id = user_id_label[i]
            history_pred_id = history_max_id
            history_accuracy = accuracy

        nb_total = i + 1
        accuracy = nb_true / nb_total
        false_rate = nb_false / nb_total
        print('Final acc = %.5f' % accuracy)

        sess.close()
