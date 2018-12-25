import numpy as np
import networkx as nx
import tensorflow as tf
import time
import random


def load_graph():
    g = nx.read_gexf("data/data.gexf")
    return g


def one_hot_embed(node, graph):
    emb_line = np.zeros(len(graph.nodes()) + 1)
    emb_line[int(node)] = 1
    # a = graph.node[node]['label'][1:len(graph.node[node]['label']) - 1]
    # b = a.split(' ')
    # for idx in b:
    #     emb_line[int(idx.split(',')[0]) - 1] = 1
    return emb_line


def random_walk(graph, start, walk_length):
    path = []
    path.append(start)
    for _ in range(walk_length - 1):
        neighbor = graph.neighbors(start)
        start = neighbor[random.randint(0, len(neighbor) - 1)]
        path.append(start)
    return path


# N-gram
def get_targets(path, idx, window_size=5):
    """
    :param path:
    :param idx:
    :param window_size:
    :return:
    """
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(path[start_point: idx] + path[idx+1: end_point+1])
    return list(targets)


def get_batch(graph, window_size=5):
    nodes_list = graph.nodes()
    random.shuffle(nodes_list)
    for node in nodes_list:
        x, y = [], []
        path = random_walk(graph=graph, start=node, walk_length=50)
        for idx in range(len(path)):
            batch_x = path[idx]
            batch_y = get_targets(path, idx, window_size)
            x_emb = one_hot_embed(batch_x, graph)
            for _ in range(len(batch_y)):
                x.append(x_emb)
            y.extend(batch_y)
        yield x, y


def deep_walk(graph, embedding_size, num_sampled):
    node_num = len(graph.nodes()) + 1
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, node_num], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        embeddings = tf.Variable(tf.random_uniform([node_num, embedding_size], -1, 1))
        embed = tf.matmul(inputs, embeddings)
        # embed = tf.nn.embedding_lookup(embeddings, inputs)
        weights = tf.Variable(tf.truncated_normal([node_num, embedding_size], stddev=0.1), dtype=tf.float32)
        biases = tf.Variable(tf.zeros(node_num), dtype=tf.float32)
        loss = tf.nn.nce_loss(weights=weights,
                              biases=biases,
                              labels=labels,
                              inputs=embed,
                              num_sampled=num_sampled,
                              num_classes=node_num)
        # loss = tf.nn.sampled_softmax_loss(weights, biases, labels, embed, num_sampled, vocabulary_size)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        epochs = 10
        window_size = 5
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())
        for e in range(1, epochs + 1):
            batches = get_batch(graph, window_size)
            start = time.time()
            for x, y in batches:
                # print("x:", x)
                # print("y:", y)
                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                loss += train_loss
                if iteration % 10 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / 10),
                          "{:.4f} sec/batch".format((end - start) / 10))
                    loss = 0
                    start = time.time()

                iteration += 1


if __name__ == '__main__':
    graph = load_graph()
    deep_walk(graph, 20, 100)


