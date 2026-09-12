from core.data_structures.Network import Network
import _pickle as cPickle


def delete_dec_nodes(network: Network, layer_indices=None) -> None:
    counter_inact = 0
    # counter_dec = 0
    if layer_indices is None:
        target_layer_indices = range(1, len(network.layers) - 1)
    else:
        target_layer_indices = sorted({
            int(index)
            for index in layer_indices
            if 0 < int(index) < len(network.layers) - 1
        })
    for i in target_layer_indices:
        temp_nodes = cPickle.loads(cPickle.dumps(network.layers[i].nodes, -1))
        # temp_nodes = copy.deepcopy(network.layers[i].nodes)
        for node in temp_nodes:
            if node.upper_bound == 0 and len(network.layers[i].nodes) > 1:
                # print(node)
                network.remove_node(node, i)
                counter_inact += 1

    print("消除了" + str(counter_inact) + "个非激活节点")
