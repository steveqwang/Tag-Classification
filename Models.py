import tensorflow as tf


class FastText(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(FastText, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.dense1 = tf.keras.layers.Dense(self.config.HIDDEN_DIM, input_shape=(None, self.config.EMBEDDING_DIM))

        self.dense2 = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.HIDDEN_DIM))

    def call(self, inputs, training=None, mask=None):
        embed = self.embedding(inputs)

        mask = tf.cast(tf.tile(tf.expand_dims(tf.sign(inputs), axis=-1), [1, 1, self.config.EMBEDDING_DIM]), tf.float32)

        out = tf.reduce_mean(embed * mask, 1)

        z = self.dense2(self.dense1(out))

        return tf.keras.activations.softmax(z)


class TextCNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextCNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.conv1 = tf.keras.layers.Conv1D(self.config.NUM_CHANNELS, self.config.kernal_size[0], activation='relu',
                                            input_shape=(self.config.MAX_LENGTH, self.config.EMBEDDING_DIM))
        self.maxPool1 = tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[0] + 1)

        self.conv2 = tf.keras.layers.Conv1D(self.config.NUM_CHANNELS, self.config.kernal_size[1], activation='relu',
                                            input_shape=(self.config.MAX_LENGTH, self.config.EMBEDDING_DIM))

        self.maxPool2 = tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[1] + 1)

        self.conv3 = tf.keras.layers.Conv1D(self.config.NUM_CHANNELS, self.config.kernal_size[2], activation='relu',
                                            input_shape=(self.config.MAX_LENGTH, self.config.EMBEDDING_DIM))

        self.maxPool3 = tf.keras.layers.MaxPool1D(pool_size=self.config.MAX_LENGTH - self.config.kernal_size[2] + 1)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.NUM_CHANNELS * 3))

    def call(self, inputs, training=None, mask=None):

        mask = tf.cast(tf.tile(tf.expand_dims(tf.sign(inputs), axis=-1), [1, 1, self.config.EMBEDDING_DIM]), tf.float32)

        embed = self.embedding(inputs) * mask

        conv1 = tf.squeeze(self.maxPool1(self.conv1(embed)), 1)
        conv2 = tf.squeeze(self.maxPool2(self.conv2(embed)), 1)
        conv3 = tf.squeeze(self.maxPool3(self.conv3(embed)), 1)

        conv_seq = tf.concat([conv1, conv2, conv3], axis=-1)

        if training:
            conv_seq = self.dropout(conv_seq)

        out = self.dense(conv_seq)

        return tf.keras.activations.softmax(out)


class TextRNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.forward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM)
        self.backward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM, go_backwards=True)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.RNN_DIM*2))

    def call(self, inputs, training=None, mask=None):

        embed = self.embedding(inputs)

        mask = self.embedding.compute_mask(inputs)

        forward_out = self.forward_LSTM(embed, mask=mask)
        backward_out = self.backward_LSTM(embed, mask=mask)

        out = tf.concat([forward_out, backward_out], axis=-1)

        if training:
            out = self.dropout(out)

        out = self.dense(out)

        return tf.keras.activations.softmax(out)


class TextRCNN(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(TextRCNN, self).__init__()
        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.forward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM, return_sequences=True)
        self.backward_LSTM = tf.keras.layers.LSTM(self.config.RNN_DIM, return_sequences=True, go_backwards=True)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense1 = tf.keras.layers.Dense(self.config.HIDDEN_DIM, input_shape=(None, self.config.RNN_DIM * 2 + self.config.EMBEDDING_DIM),
                                            activation='tanh')

        self.dense2 = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, self.config.HIDDEN_DIM))

    def call(self, inputs, training=None, mask=None):

        mask1 = tf.cast(tf.tile(tf.expand_dims(tf.sign(inputs), axis=-1), [1, 1, self.config.EMBEDDING_DIM]), tf.float32)

        embed = self.embedding(inputs)

        mask = self.embedding.compute_mask(inputs)
        forward_out = self.forward_LSTM(embed, mask=mask)
        backward_out = self.backward_LSTM(embed, mask=mask)

        out = tf.concat([forward_out, embed * mask1, backward_out], axis=-1)

        out = self.dense1(out)

        out = tf.keras.layers.MaxPool1D(self.config.MAX_LENGTH)(out)

        if training:
            out = self.dropout(out)

        out = self.dense2(tf.squeeze(out, 1))

        return tf.keras.activations.softmax(out)


class SelfAttention(tf.keras.Model):
    def __init__(self, config, VOCAB_SIZE):
        super(SelfAttention, self).__init__()

        self.config = config

        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE + 1, self.config.EMBEDDING_DIM, mask_zero=True)

        self.weight = []
        for i in range(self.config.num_layer):
            sub_layer = []
            for j in range(self.config.num_heads):
                dense_Q = tf.keras.layers.Dense(self.config.dim, input_shape=(None, self.config.EMBEDDING_DIM),
                                                name='dense_Q_'+str(i)+"_"+str(j))
                dense_K = tf.keras.layers.Dense(self.config.dim, input_shape=(None, self.config.EMBEDDING_DIM),
                                                name='dense_K_'+str(i)+"_"+str(j))
                dense_V = tf.keras.layers.Dense(self.config.dim, input_shape=(None, self.config.EMBEDDING_DIM),
                                                name='dense_V_'+str(i)+"_"+str(j))
                sub_layer.append([dense_Q, dense_K, dense_V])
            multihead_dense = tf.keras.layers.Dense(self.config.EMBEDDING_DIM,
                                                    input_shape=(None, self.config.dim * self.config.num_heads),
                                                    name='dense_Q_'+str(i))
            FF_1 = tf.keras.layers.Dense(self.config.EMBEDDING_DIM * 2, input_shape=(None, self.config.EMBEDDING_DIM),
                                         name='dense_FF1_'+str(i))
            FF_2 = tf.keras.layers.Dense(self.config.EMBEDDING_DIM, input_shape=(None, self.config.EMBEDDING_DIM),
                                         name='dense_FF2_' + str(i))
            self.weight.append([sub_layer, multihead_dense, FF_1, FF_2])

        self.dense = tf.keras.layers.Dense(self.config.OUTPUT_DIM, input_shape=(None, 10))

        self.conv = tf.keras.layers.Conv1D(1, 5, strides=5)

    def call(self, inputs, training=None, mask=None):

        mask = tf.cast(tf.tile(tf.expand_dims(tf.sign(inputs), axis=-1), [1, 1, self.config.EMBEDDING_DIM]), tf.float32)

        out = self.embedding(inputs) * mask

        for i in range(self.config.num_layer):
            multihead_out = self.build_multiHead_attention(out, out, out, self.weight[i][0], self.weight[i][1])

            out = tf.math.l2_normalize(out+multihead_out, axis=-1)

            FF_out = self.build_FFNN(out, self.weight[i][2], self.weight[i][3])

            out = FF_out + out

        return tf.keras.activations.softmax(self.dense(tf.squeeze(self.conv(out), axis=-1)))

    def build_scaled_dotProduct_attention(self, Q, K, V):

        Q_K = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / (self.config.EMBEDDING_DIM ** 0.5)

        out = tf.matmul(tf.keras.activations.softmax(Q_K), V)

        return out

    def build_multiHead_attention(self, Q, K, V, weights, multi_dense):

        heads = []

        for i in range(self.config.num_heads):
            dense_Q = weights[i][0]
            dense_K = weights[i][1]
            dense_V = weights[i][2]
            heads.append(self.build_scaled_dotProduct_attention(dense_Q(Q), dense_K(K), dense_V(V)))

        multiHead = tf.concat(heads, axis=-1)

        return multi_dense(multiHead)

    def build_FFNN(self, inputs, FF1, FF2):

        return FF2(tf.keras.activations.relu(FF1(inputs)))



