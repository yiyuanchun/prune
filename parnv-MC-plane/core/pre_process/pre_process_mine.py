from core.data_structures.Network import Network
from .read_input_bound import read_bounds_from_property
from .calc_bounds import calcu_bounds
from .crown_bounds import apply_crown_hidden_layer_bounds
from .generate_ori_map import genarate_symb_map, generate_ori_net_map
from .del_dec_nodes import delete_dec_nodes


def do_process_before(
        network: Network,
        test_property: str,
        NAIVE_BOUND_CALCULATION: bool = False,
        CROWN_BOUND_CALCULATION: bool = True,
) -> None:
    network.generate_name2node_map()
    generate_ori_net_map(network)
    # print(network)
    read_bounds_from_property(network, test_property)
    if NAIVE_BOUND_CALCULATION:
        calcu_bounds(network, network.name2node_map)
    else:
        genarate_symb_map(network, network.name2node_map)
    if CROWN_BOUND_CALCULATION:
        apply_crown_hidden_layer_bounds(network)
    # print(network)


def do_process_after(network: Network, layer_indices=None) -> None:
    delete_dec_nodes(network, layer_indices=layer_indices)
