import networkx as nx


def load_graph():
    g = nx.read_gexf("../data/data.gexf")
    return g


def label_to_list(label):
    c = []
    a = label[1:len(label) - 1]
    b = a.split(' ')
    for idx in b:
        c.append(int(idx.split(',')[0]))
    return c


if __name__ == '__main__':
    count = 0
    g = load_graph()
    total = len(g.nodes())
    for id in g.nodes():
        if len(label_to_list(g.node[id]['label'])) > 1:
            count += 1
    print("total: ", total)
    print("count: ", count)

