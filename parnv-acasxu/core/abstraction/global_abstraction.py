from core.data_structures.Network import Network
from core.pre_process.pre_process import preprocess, preprocess_updated
from core.pre_process.after_preprocess import after_preprocess
from core.utils.debug_utils import debug_print
from core.visualization.visualize_network import visualize_network
from core.pre_process.pre_process_mine import do_process_before, do_process_after
# from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import (VERBOSE, FIRST_ABSTRACT_LAYER, INT_MIN, INT_MAX)
# from core.utils.verification_properties_utils import TEST_PROPERTY_ACAS
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.utils.cal_contribution import calculate_contribution
from core.utils.cal_layer_average_in_weight import cal_average_in_weight
from core.utils.combine_influ_of2nodes import combin_influnce
from core.data_structures.Abstract_action import Abstract_action
from core.utils.find_relation import find_relation
from core.utils.propagation import propagation_net
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.utils.ar_utils import calculate_weight_of_edge_between_two_part_groups
from core.abstraction.structured_merge import try_structured_merge_delete
from core.utils.layer_scope import get_last_hidden_layer_indices
import numpy as np
import copy
import time
import _pickle as cPickle


def global_abstraction_based_on_contribution(network: Network, test_property: dict, do_preprocess: bool = True,
                                             sequence_length: int = 50):
    network.generate_in_edge_weight()
    # print(network)
    # layer_influence = []
    # for i in range(1,len(network.layers)-1):
    #     layer_influence.append(cal_average_in_weight(network, i))
    # print(layer_influence)
    actions = []
    # last abstract layer for adversarial properties
    last_free_layer = len(network.layers) - 3
    sat = False
    t1 = time.time()
    do_process_before(network, test_property)
    t2 = time.time()
    print("bound analysis time {}".format(t2 - t1))
    ############### only for adversarial properties #######################
    random_inputs = []
    print("output upper bound")
    print(network.layers[-1].nodes[0].upper_bound)
    if network.layers[-1].nodes[0].upper_bound < 0.0001:
        print(network.layers[-1].nodes[0].upper_bound)
        print("XXXXXXXXXXXXXXXXXXXXXXXXX UNSAT,FOUND IN INTERVAL CALCULATION XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        return network, random_inputs, network, sat, actions
    ############   may introduce more interval computing methods later   ############
    abstract_layer_indices = get_last_hidden_layer_indices(network, count=2)
    if not abstract_layer_indices:
        print("No hidden layer is available for abstraction.")
        return network, random_inputs, network, sat, actions
    print("abstract layer indices: {}".format(abstract_layer_indices))
    nodename2contribution_map = {}
    # influnce_arg = 1
    # t1 = time.time()
    do_process_after(network, layer_indices=abstract_layer_indices)
    preprocess_updated(network, layer_indices=abstract_layer_indices)
    print(network)
    weights = network.generate_weights()
    print("weights", weights)

    # weights = np.mat(weights)
    # print("weights-shape", weights.shape)

    # print(network)
    # t2 = time.time()
    # print(t2-t1)
    # after_preprocess(network,False)
    # print(network)
    ############################ 加代码 #############################
    # do_process_after(network)
    # t3 = time.time()
    # print("del time")
    # print(t3-t2)
    preprocessed_network = cPickle.loads(cPickle.dumps(network, -1))
    preprocessed_network.generate_name2node_map()

    input_size = len(network.layers[0].nodes)
    # generate random inputs in the bound of the test property
    # property_dict = TEST_PROPERTY_ACAS[test_property]
    random_inputs = get_limited_random_inputs(
        input_size=input_size,
        test_property=test_property
    )
    # print(random_inputs)
    count = 0
    # print(random_inputs[0])
    for j in abstract_layer_indices:
        if j >= len(network.layers) - 1:
            continue
        #    influnce_arg = influnce_arg * layer_influence[i]
        for node in network.layers[j].nodes:
            # print(node.cal_contri())
            # print(node.name)
            # print(node.upper_bound)
            # print(node.lower_bound)
            node_contribution = node.cal_contri()
            nodename2contribution_map[node.name] = node_contribution
    # print(nodename2contribution_map)
    # print(nodename2contribution_map)
    # deleted_name2node= {}
    alert = 0
    while not has_violation(network, test_property, random_inputs):
        ####### loop #########
        # t0 = time.time()
        network.generate_in_edge_weight()
        network.generate_name2node_map()
        # nodes2edge_between_map = copy.deepcopy(network.get_nodes2edge_between_map())
        nodes2edge_between_map = network.get_nodes2edge_between_map()
        # print(nodes2edge_between_map)
        # print(nodename2contribution_map)
        if not nodename2contribution_map:
            break
        candicate_node = network.name2node_map[
            sorted(nodename2contribution_map.items(), key=lambda item: item[1])[0][0]]
        layer_index = int(candicate_node.name.split("_")[1])
        if layer_index == len(network.layers) - 2:
            alert += 1
            if alert == len(network.layers[layer_index].nodes) - 1:
                break
        # print(sorted(nodename2contribution_map.items(),key=lambda item:item[1]))
        # neg_inc = candicate_node.ar_type
        # print(nodename2contribution_map[candicate_node.name])
        index = int(candicate_node.name.split("_")[1])

        candicate_node_name = candicate_node.name
        structured_merge_result = try_structured_merge_delete(
            network=network,
            candidate_node=candicate_node,
            layer_index=index,
            nodes2edge_between_map=nodes2edge_between_map,
        )
        if structured_merge_result is None:
            print("No same-layer same-type structured merge target for {}".format(candicate_node_name))
            del nodename2contribution_map[candicate_node_name]
            continue

        target_node_name, merge_gap, precision_loss = structured_merge_result
        print("structured merge delete operation")
        print("candidate={}".format(candicate_node_name))
        print("target={}".format(target_node_name))
        print("preactivation_gap={}".format(merge_gap))
        print("precision_loss={}".format(precision_loss))

        if last_free_layer > layer_index:
            last_free_layer = layer_index

        del nodename2contribution_map[candicate_node_name]
        if target_node_name in nodename2contribution_map:
            del nodename2contribution_map[target_node_name]
        count += 1

    if count == 0:
        sat = True
        print("times of operation:" + str(count))
        return network, random_inputs, preprocessed_network, sat, actions
    else:
        print("times of operation:" + str(count))
        return network, random_inputs, preprocessed_network, sat, actions
