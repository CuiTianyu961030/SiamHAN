import tensorflow as tf
from models import layers
from models.base_gattn import BaseGAttN


class HAN(BaseGAttN):
    def inference(self, model_id, inputs_list, bias_mat_list, attn_drop, ffd_drop, training, mask,
                  attn_hid_units, n_heads, dense_hid_units, activation=tf.nn.elu, residual=False, mp_att_size=128):

        with tf.variable_scope('han', reuse=tf.AUTO_REUSE):
            embed_list = []
            for inputs, bias_mat in zip(inputs_list, bias_mat_list):
                attns = []
                for _ in range(n_heads[0]):
                    attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=attn_hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
                h_1 = tf.concat(attns, axis=-1)

                for i in range(1, len(attn_hid_units)):
                    h_old = h_1
                    attns = []
                    for _ in range(n_heads[i]):
                        attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                                      out_sz=attn_hid_units[i],
                                                      activation=activation,
                                                      in_drop=ffd_drop,
                                                      coef_drop=attn_drop, residual=residual))
                    h_1 = tf.concat(attns, axis=-1)
                embed_list.append(tf.expand_dims(h_1, axis=2))

            multi_embed = tf.concat(embed_list, axis=2)
            nodes_embed, att_val = layers.simple_attn(multi_embed, mp_att_size,
                                                      time_major=False,
                                                      return_alphas=True)

            embed_len = tf.reduce_min(tf.reduce_sum(mask, axis=-1), axis=0)
            graph_embed, graph_attn = layers.simple_attn_2(nodes_embed[:, :embed_len], mp_att_size,
                                                      time_major=False,
                                                      return_alphas=True)
        return graph_embed, att_val, graph_attn


class SiamHAN(BaseGAttN):
    def inference(self, inputs_list, bias_mat_list, attn_drop, ffd_drop, training, mask,
                  attn_hid_units, n_heads, dense_hid_units, activation=tf.nn.elu, nb_classes=2, residual=False):
        base_network = HAN()
        graph_embed_1, att_val_1, graph_attn_1 = base_network.inference(model_id=0,
                                                          inputs_list=inputs_list[0], attn_drop=attn_drop,
                                                          ffd_drop=ffd_drop, bias_mat_list=bias_mat_list[0],
                                                          attn_hid_units=attn_hid_units, n_heads=n_heads,
                                                          dense_hid_units=dense_hid_units, training=training,
                                                          residual=residual, activation=activation, mask=mask)

        graph_embed_2, att_val_2, graph_attn_2 = base_network.inference(model_id=1,
                                                          inputs_list=inputs_list[1], attn_drop=attn_drop,
                                                          ffd_drop=ffd_drop, bias_mat_list=bias_mat_list[1],
                                                          attn_hid_units=attn_hid_units, n_heads=n_heads,
                                                          dense_hid_units=dense_hid_units, training=training,
                                                          residual=residual, activation=activation, mask=mask)
        semantic_attn = [att_val_1, att_val_2]
        graph_attn = [graph_attn_1, graph_attn_2]

        with tf.name_scope("output"):

            logits = tf.sqrt(tf.reduce_sum(tf.square(graph_embed_1 - graph_embed_2), 1) + 1e-6, name='distance')
            attval = tf.concat(semantic_attn, axis=0, name='semantic_attention')
            graph_attn = tf.concat(graph_attn, axis=0, name='graph_attention')

        return logits, att_val_1, att_val_2
