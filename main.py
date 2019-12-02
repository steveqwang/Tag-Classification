from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils import *
from Models import *
from configuration import Configuration
import tensorflow as tf
import time

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':

    print("Building Inputs.....")

    vocab = Vocab()

    vocab.build(['data/aws_title.txt', 'data/azure_title.txt', 'data/gcp_title.txt'])

    X_aws = preprocess('data/aws_title.txt', vocab)
    Y_aws = [[1, 0, 0] for _ in range(len(X_aws))]
    aws_X_train, aws_X_test, aws_Y_train, aws_Y_test = train_test_split(X_aws, Y_aws, test_size=0.15, random_state=42)
    X_azure = preprocess('data/azure_title.txt', vocab)
    Y_azure = [[0, 1, 0] for _ in range(len(X_azure))]
    azure_X_train, azure_X_test, azure_Y_train, azure_Y_test = train_test_split(X_azure, Y_azure, test_size=0.15, random_state=42)
    X_gcp = preprocess('data/gcp_title.txt', vocab)
    Y_gcp = [[0, 0, 1] for _ in range(len(X_gcp))]
    gcp_X_train, gcp_X_test, gcp_Y_train, gcp_Y_test = train_test_split(X_gcp, Y_gcp, test_size=0.15, random_state=42)

    X_train = aws_X_train + azure_X_train + gcp_X_train
    Y_train = aws_Y_train + azure_Y_train + gcp_Y_train

    X_test = aws_X_test + azure_X_test + gcp_X_test
    Y_test = aws_Y_test + azure_Y_test + gcp_Y_test

    X_train, Y_train = shuffle(X_train, Y_train)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    train = DataLoader(X_train, Y_train, 20)
    test = DataLoader(X_test, Y_test, 20)
    # print(train.X)

    vocab_size = vocab.size

    mode = 'SelfAttention'

    print("Building Model.....")

    config = Configuration(mode)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/'+mode+'/ckp/cp.ckpt', verbose=1, save_weights_only=True, period=1)
    # tb_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir='model\\' + mode + '\\log\\', update_freq='batch', embeddings_freq=1)

    # model = FastText(config, vocab_size)
    # model = TextCNN(config, vocab_size)
    # model = TextRNN(config, vocab_size)
    # model = TextRCNN(config, vocab_size)
    model = SelfAttention(config, vocab_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.MeanAbsoluteError()],
                  run_eagerly=True)

    print("Start Training.....")

    start_time = time.time()
    model.fit(train.X, train.Y, epochs=10, batch_size=64,
              validation_data=(test.X, test.Y),
              callbacks=[cp_callback])#, tb_callback])
    end_time = time.time()

    print('time for 10 epochs: ' + str(end_time-start_time))
    model.evaluate(test.X, test.Y)
    model.summary()
