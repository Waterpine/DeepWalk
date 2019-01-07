import os
import time
import numpy as np
import tensorflow as tf
import networkx as nx
from sklearn import metrics


def load_graph():
    g = nx.read_gexf("data/BlogCatalog.gexf")
    return g


def loss_and_metric(logits, labels):
    labels = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)

    predictions = tf.argmax(logits, -1)
    actuals = tf.argmax(labels, -1)

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))

    hits = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

    return cross_entropy_loss, predictions, actuals


def label_to_list(label):
    c = []
    a = label[1:len(label) - 1]
    b = a.split(' ')
    for idx in b:
        c.append(int(idx.split(',')[0]))
    return c


def one_hot_embed(label):
    """
    :param node: node_id
    :param graph: G
    :return: embedding [0 0 0 ... 1 0 0 ... 0]
    """
    emb_line = np.zeros(39)
    emb_line[label[0] - 1] = 1
    return emb_line


def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("embedding/7000.ckpt-7000.meta")
        saver.restore(sess, tf.train.latest_checkpoint("embedding"))
        graph = tf.get_default_graph()
        embeddings = graph.get_tensor_by_name("embeddings:0").eval()
    return embeddings


def get_batch(graph, batch_size, type):
    embeddings_total = load_model()
    if type == 'train':
        embeddings = embeddings_total[:int(len(embeddings_total) * 0.6)]
        num_batch = len(embeddings) // batch_size
        for num in range(num_batch):
            x, y = [], []
            for idx in range(batch_size):
                label = label_to_list(graph.node[str(batch_size * num + idx + 1)]['label'])
                if len(label) == 1:
                    x.append(embeddings_total[batch_size * num + idx + 1])
                    y_emb = one_hot_embed(label)
                    y.append(y_emb)
            yield x, y
    elif type == 'test':
        embeddings = embeddings_total[int(len(embeddings_total) * 0.6):]
        x, y = [], []
        for idx in range(len(embeddings)):
            num = int(len(embeddings_total) * 0.6) + idx
            label = label_to_list(graph.node[str(num)]['label'])
            if len(label) == 1:
                x.append(embeddings_total[num])
                y_emb = one_hot_embed(label)
                y.append(y_emb)
        yield x, y
    else:
        raise ValueError("type must be test or train")


def classification_model(embedding_size):
    graph = load_graph()
    node_classes = 39
    classification_graph = tf.Graph()
    with classification_graph.as_default():
        embeddings_c = tf.placeholder(tf.float32, [None, embedding_size], name='embeddings_c')
        labels_c = tf.placeholder(tf.int32, [None, node_classes], name='label_c')
        weight_c = tf.Variable(tf.truncated_normal([embedding_size, node_classes], stddev=0.1), dtype=tf.float32)
        bias_c = tf.Variable(tf.zeros(node_classes), dtype=tf.float32)
        logits_c = tf.nn.softmax(tf.add(tf.matmul(embeddings_c, weight_c), bias_c))
        # loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_c, logits=logits_c))
        loss_c, predict_c, true_c = loss_and_metric(logits=logits_c, labels=labels_c)
        optimizer_c = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss_c)

    with tf.Session(graph=classification_graph) as sess:
        epochs = 100
        iteration = 1
        total_loss = 0
        sess.run(tf.global_variables_initializer())
        # train
        for e in range(1, epochs + 1):
            batches = get_batch(graph=graph, batch_size=128, type='train')
            start = time.time()
            for x, y in batches:
                feed = {embeddings_c: x,
                        labels_c: y}
                train_loss, _ = sess.run([loss_c, optimizer_c], feed_dict=feed)
                total_loss += train_loss
                if iteration % 10 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(total_loss / 10),
                          "{:.4f} sec/batch".format((end - start) / 10))
                    total_loss = 0
                    start = time.time()
                iteration += 1

        # test
        batches = get_batch(graph=graph, batch_size=128, type='test')
        for x, y in batches:
            print(np.shape(x))
            print(np.shape(y))
            feed = {embeddings_c: x,
                    labels_c: y}
            y_predict, y_true = sess.run([predict_c, true_c], feed_dict=feed)
            print("predict:", y_predict)
            print("true:", y_true)
            print('macro-f1-score:', metrics.f1_score(y_true, y_predict,
                                                      labels=list(range(1, node_classes + 1)), average='macro'))
            print('micro-f1-score:', metrics.f1_score(y_true, y_predict,
                                                      labels=list(range(1, node_classes + 1)), average='micro'))


if __name__ == '__main__':
    embeddings = load_model()
    print(np.shape(embeddings)[0])
    print(np.shape(embeddings)[1])
    embedding_size = np.shape(embeddings)[1]
    classification_model(embedding_size)

