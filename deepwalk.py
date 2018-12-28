import os
import time
import random
import operator
import numpy as np
import networkx as nx
import tensorflow as tf


def load_graph():
    g = nx.read_gexf("data/data.gexf")
    return g


def del_files(path, best_iter):
    for root, _, files in os.walk(path):
        for name in files:
            print(name.split('.'))
            if str(best_iter) in name.split('.'):
                pass
            elif name == 'checkpoint':
                pass
            else:
                os.remove(os.path.join(root, name))


def one_hot_embed(node, graph):
    """
    :param node: node_id
    :param graph: G
    :return: embedding [0 0 0 ... 1 0 0 ... 0]
    """
    emb_line = np.zeros(len(graph.nodes()) + 1)
    emb_line[int(node)] = 1
    # a = graph.node[node]['label'][1:len(graph.node[node]['label']) - 1]
    # b = a.split(' ')
    # for idx in b:
    #     emb_line[int(idx.split(',')[0]) - 1] = 1
    return emb_line


# random walk sample
def random_walk(graph, times, start, walk_length):
    """
    :param graph: G
    :param start: start node
    :param walk_length: path_length
    :return: random walk path
    """
    path_total = []
    for _ in range(times):
        path = []
        path.append(start)
        for _ in range(walk_length - 1):
            neighbor = graph.neighbors(start)
            start = neighbor[random.randint(0, len(neighbor) - 1)]
            path.append(start)
        path_total.append(path)
    return path_total


# N-gram
def get_targets(path, idx, window_size):
    """
    :param path: random walk sample
    :param idx: target node
    :param window_size: window size
    :return: [... idx - 1 idx idx + 1 ...]
    """
    # target_window = np.random.randint(1, window_size + 1)
    target_window = window_size
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(path[start_point: idx] + path[idx+1: end_point+1])
    return list(targets)


# generate bacth  x: embedding of the node y: predict node_num
def get_batch(nodes_list, window_size):
    """
    :param graph: G
    :param window_size: window size
    :return: batch
    """
    for node in nodes_list:
        x, y = [], []
        path = random_walk(graph=graph, times=80, start=node, walk_length=50)
        for num in range(np.shape(path)[0]):
            for idx in range(len(path[num])):
                batch_x = path[num][idx]
                batch_y = get_targets(path[num], idx, window_size)
                x_emb = one_hot_embed(batch_x, graph)
                for _ in range(len(batch_y)):
                    x.append(x_emb)
                y.extend(batch_y)
        yield x, y


def deep_walk(graph, embedding_size, num_sampled, window_size):
    # checkpoint = os.path.join(os.getcwd(), 'embedding/model.ckpt')
    node_num = len(graph.nodes()) + 1
    nodes_list = graph.nodes()
    random.shuffle(nodes_list)
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, node_num], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        embeddings = tf.Variable(tf.random_uniform([node_num, embedding_size], -1, 1), name='embeddings')
        embed = tf.matmul(inputs, embeddings, name='embed')
        # embed = tf.nn.embedding_lookup(embeddings, inputs)
        weights = tf.Variable(tf.truncated_normal([node_num, embedding_size], stddev=0.1),
                              dtype=tf.float32, name='weights')
        biases = tf.Variable(tf.zeros(node_num), dtype=tf.float32, name='biases')
        loss = tf.nn.nce_loss(weights=weights,
                              biases=biases,
                              labels=labels,
                              inputs=embed,
                              num_sampled=num_sampled,
                              num_classes=node_num)
        # loss = tf.nn.sampled_softmax_loss(weights, biases, labels, embed, num_sampled, node_num)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
        saver = tf.train.Saver(max_to_keep=None)

    with tf.Session(graph=train_graph) as sess:
        epochs = 1
        iteration = 1
        loss = 0
        train_loss_dict = {}
        sess.run(tf.global_variables_initializer())
        for e in range(1, epochs + 1):
            batches = get_batch(nodes_list=nodes_list, window_size=window_size)
            start = time.time()
            for x, y in batches:
                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                loss += train_loss
                if iteration % 100 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / 100),
                          "{:.4f} sec/batch".format((end - start) / 100))
                    train_loss_dict[iteration] = train_loss
                    saver.save(sess, 'embedding/{}.ckpt'.format(iteration))
                    loss = 0
                    start = time.time()
                iteration += 1
            sorted_train_loss = sorted(train_loss_dict.items(),
                                       key=operator.itemgetter(1),
                                       reverse=True)
            best_iter, _ = sorted_train_loss[0]
            print("best_iter:", best_iter)
            del_files(path="embedding", best_iter=best_iter)


if __name__ == '__main__':
    graph = load_graph()
    deep_walk(graph=graph, embedding_size=200, num_sampled=5, window_size=10)

