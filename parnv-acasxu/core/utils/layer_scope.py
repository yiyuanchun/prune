from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structures.Network import Network


def get_last_hidden_layer_indices(network: "Network", count: int = 2) -> list:
    """
    Return the layer indices of the last `count` hidden layers.
    """
    if count <= 0:
        return []
    hidden_layer_indices = [
        index
        for index, layer in enumerate(network.layers)
        if layer.type_name == "hidden"
    ]
    return hidden_layer_indices[-count:]


def get_node_layer_index(node_name: str) -> Optional[int]:
    """
    Extract the layer index from a node/part name such as x_3_10_inc.
    """
    atom_name = node_name.split("+")[0]
    name_parts = atom_name.split("_")
    if len(name_parts) < 3:
        return None
    try:
        return int(name_parts[1])
    except ValueError:
        return None


def is_node_name_in_layer_indices(node_name: str, layer_indices: Iterable[int]) -> bool:
    layer_index = get_node_layer_index(node_name)
    return layer_index is not None and layer_index in set(layer_indices)
