from models.siamhan import SiamHAN
from dataloader import DataLoader
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

checkpt_file = 'pre_trained/siamhan.ckpt'

# training params
batch_size = 256
nb_epochs = 500
patience = 50  # 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
attn_hid_units = [256, 256]  # numbers of hidden units per each attention head in each layer
dense_hid_units = [768, 96]
n_heads = [4, 4, 6]  # additional entry for the output layer
residual = True
nonlinearity = tf.nn.elu
model = SiamHAN()


if __name__ == '__main__':
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. attn layers: ' + str(len(attn_hid_units)))
    print('nb. attn units per layer: ' + str(attn_hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('nb. fc layers: ' + str(len(dense_hid_units)))
    print('nb. fc units per layer: ' + str(dense_hid_units))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))


    data_loader = DataLoader()
    scs_biases, fcf_biases, fsf_biases, feature, mask, label = data_loader.build_data()

    train_biases_list = [[scs_biases[0][0], fcf_biases[0][0], fsf_biases[0][0]], [scs_biases[0][1], fcf_biases[0][1], fsf_biases[0][1]]]
    val_biases_list = [[scs_biases[1][0], fcf_biases[1][0], fsf_biases[1][0]], [scs_biases[1][1], fcf_biases[1][1], fsf_biases[1][1]]]
    test_biases_list = [[scs_biases[2][0], fcf_biases[2][0], fsf_biases[2][0]], [scs_biases[2][1], fcf_biases[2][1], fsf_biases[2][1]]]
    train_feature_list = [[feature[0][0], feature[0][0], feature[0][0]], [feature[0][1], feature[0][1], feature[0][1]]]
    val_feature_list = [[feature[1][0], feature[1][0], feature[1][0]], [feature[1][1], feature[1][1], feature[1][1]]]
    test_feature_list = [[feature[2][0], feature[2][0], feature[2][0]], [feature[2][1], feature[2][1], feature[2][1]]]

    train_mask = mask[0]
    val_mask = mask[1]
    test_mask = mask[2]
    train_label = label[0]
    val_label = label[1]
    test_label = label[2]

    nb_nodes = train_feature_list[0][0].shape[1]
    ft_size = train_feature_list[0][0].shape[2]

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in_list = [[tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, ft_size),
                                           name='ftr_in_1_{}'.format(i)) for i in range(len(train_feature_list[0]))],
                           [tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, ft_size),
                                           name='ftr_in_2_{}'.format(i)) for i in range(len(train_feature_list[0]))]]
            bias_in_list = [[tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, nb_nodes),
                                            name='bias_in_1_{}'.format(i)) for i in range(len(train_biases_list[0]))],
                            [tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, nb_nodes),
                                            name='bias_in_2_{}'.format(i)) for i in range(len(train_biases_list[0]))]]
            lbl_in = tf.placeholder(dtype=tf.float32, shape=(None,), name='lbl_in')
            msk_in = tf.placeholder(dtype=tf.int32, shape=(None, nb_nodes), name='msk_in')
            attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
            is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')

        # forward
        logits, att_val_1, att_val_2 = model.inference(inputs_list=ftr_in_list, bias_mat_list=bias_in_list,
                                                                  attn_drop=attn_drop, ffd_drop=ffd_drop,
                                                                  attn_hid_units=attn_hid_units, n_heads=n_heads,
                                                                  dense_hid_units=dense_hid_units, training=is_train,
                                                                  residual=residual, activation=nonlinearity, mask=msk_in)

        msk_resh = tf.reshape(msk_in, [-1])

        loss = model.contrastive_loss(logits, lbl_in)
        accuracy, predictions = model.predictions_accuracy(logits, lbl_in)

        # optimize
        train_op = model.training(loss, lr, l2_coef)
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        with tf.Session() as sess:
            sess.run(init_op)

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = train_feature_list[0][0].shape[0]
                logits_list = []

                # training
                while (tr_step + 1) * batch_size < tr_size:

                    feed_dict = {
                        lbl_in: train_label[tr_step * batch_size:(tr_step + 1) * batch_size],
                        msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                        is_train: True,
                        attn_drop: 0.0,
                        ffd_drop: 0.0
                    }
                    for i in [0, 1]:
                        feed_dict.update({
                                name: data[tr_step * batch_size:(tr_step + 1) * batch_size]
                                for name, data in zip(ftr_in_list[i], train_feature_list[i])})
                        feed_dict.update({
                                name: data[tr_step * batch_size:(tr_step + 1) * batch_size]
                                for name, data in zip(bias_in_list[i], train_biases_list[i])})

                    _, loss_value_tr, acc_tr, att_val_train_1, att_val_train_2, logits_train = sess.run([train_op, loss, accuracy, att_val_1, att_val_2, logits],
                                                                       feed_dict=feed_dict)
                    logits_list.append(logits_train)

                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1


                vl_step = 0
                vl_size = val_feature_list[0][0].shape[0]

                # validation
                while (vl_step + 1) * batch_size < vl_size:

                    feed_dict = {
                        lbl_in: val_label[vl_step * batch_size:(vl_step + 1) * batch_size],
                        msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                        is_train: False,
                        attn_drop: 0.0,
                        ffd_drop: 0.0
                    }
                    for i in [0, 1]:
                        feed_dict.update({
                            name: data[vl_step * batch_size:(vl_step + 1) * batch_size]
                            for name, data in zip(ftr_in_list[i], val_feature_list[i])})
                        feed_dict.update({
                            name: data[vl_step * batch_size:(vl_step + 1) * batch_size]
                            for name, data in zip(bias_in_list[i], val_biases_list[i])})

                    loss_value_vl, acc_vl, logits_val, predictions_vl = sess.run([loss, accuracy, logits, predictions], feed_dict=feed_dict)

                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                # print('Epoch: {}, att_val_1: {}, att_val_2: {}'.format(epoch, att_val_train_1, att_val_train_2))
                print('Epoch %s Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                      (epoch, train_loss_avg / tr_step, train_acc_avg / tr_step,
                       val_loss_avg / vl_step, val_acc_avg / vl_step))

                if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                    if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg / vl_step
                        vlss_early_model = val_loss_avg / vl_step
                        saver.save(sess, checkpt_file)
                    vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

            saver.restore(sess, checkpt_file)
            print('load model from : {}'.format(checkpt_file))

            ts_size = test_feature_list[0][0].shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            # test
            while (ts_step + 1) * batch_size < ts_size:

                feed_dict = {
                    lbl_in: test_label[ts_step * batch_size:(ts_step + 1) * batch_size],
                    msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                    is_train: False,
                    attn_drop: 0.0,
                    ffd_drop: 0.0
                }
                for i in [0, 1]:
                    feed_dict.update({
                        name: data[ts_step * batch_size:(ts_step + 1) * batch_size]
                        for name, data in zip(ftr_in_list[i], test_feature_list[i])})
                    feed_dict.update({
                        name: data[ts_step * batch_size:(ts_step + 1) * batch_size]
                        for name, data in zip(bias_in_list[i], test_biases_list[i])})

                loss_value_ts, acc_ts = sess.run([loss, accuracy], feed_dict=feed_dict)

                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

            print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

            sess.close()
