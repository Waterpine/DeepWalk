import numpy as np
import tensorflow as tf


def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("")
        saver.restore(sess, tf.train.latest_checkpoint(""))
        graph = tf.get_default_graph()
        embeddings = graph.get_tensor_by_name("embeddings").eval()
    return embeddings


def model(embedding_size):
    embeddings = tf.placeholder(tf.float32, [None, embedding_size], name='embeddings')
    labels = tf.placeholder(tf.int32, [None, 1], name='label')
    weight = tf.Variable(tf.random_uniform([embedding_size, 1], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.zeros(1), dtype=tf.float32)
    logits = tf.add(tf.matmul(embeddings, weight), bias)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    save = tf.train.Saver()


def train():
    embeddings = load_model()
    model(embedding_size=np.shape(embeddings)[1])
    with tf.Session() as sess:
        pass


if __name__ == '__main__':
    pass



