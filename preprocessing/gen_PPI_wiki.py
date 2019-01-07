import scipy.io as sio
import networkx as nx


def main():
    data = sio.loadmat('Homo_sapiens.mat')
    network = data['network'].toarray()
    group = data['group'].toarray()
    num_nodes = len(group)
    G = nx.Graph()
    for i in range(num_nodes):
        node_class = []
        for j in range(len(group[i])):
            if int(group[i][j]) == 1:
                node_class.append(j + 1)
        G.add_node(i + 1, label=node_class)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if int(network[i][j]) == 1:
                G.add_edge(i + 1, j + 1)
    nx.write_gexf(G, "../data/Wikipedia.gexf")


if __name__ == '__main__':
    main()


