import _pickle as cPickle

from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Network import Network


def _node_type_key(node: ARNode):
    parts = node.name.split("+")[0].split("_")
    monotone_type = node.ar_type
    pos_neg_type = None
    for part in parts:
        if part in ("pos", "neg"):
            pos_neg_type = part
            break
    return pos_neg_type, monotone_type


def _preactivation_lower(node: ARNode) -> float:
    return float(getattr(node, "preactivation_lower_bound", node.lower_bound))


def _preactivation_upper(node: ARNode) -> float:
    return float(getattr(node, "preactivation_upper_bound", node.upper_bound))


def _merge_gap(candidate_node: ARNode, target_node: ARNode) -> float:
    if candidate_node.ar_type == "inc":
        return _preactivation_upper(candidate_node) - _preactivation_lower(target_node)
    if candidate_node.ar_type == "dec":
        return _preactivation_lower(candidate_node) - _preactivation_upper(target_node)
    raise ValueError("Unsupported node ar_type for structured merge: {}".format(candidate_node.ar_type))


def _bias_adjustment(candidate_node: ARNode, gap: float) -> float:
    if candidate_node.ar_type == "inc":
        return float(gap) if gap > 0 else 0.0
    if candidate_node.ar_type == "dec":
        return float(gap) if gap < 0 else 0.0
    raise ValueError("Unsupported node ar_type for structured merge: {}".format(candidate_node.ar_type))


def _edge_transfer_loss(candidate_node: ARNode, target_node: ARNode, nodes2edge_between_map) -> float:
    loss = 0.0
    for out_edge in candidate_node.out_edges:
        target_edge = nodes2edge_between_map.get((target_node.name, out_edge.dest), None)
        target_weight = 0.0 if target_edge is None else float(target_edge.weight)
        loss += abs(float(out_edge.weight) + target_weight - target_weight)
    return loss


def _candidate_precision_loss(candidate_node: ARNode, target_node: ARNode, nodes2edge_between_map) -> float:
    gap = _merge_gap(candidate_node, target_node)
    return abs(_bias_adjustment(candidate_node, gap)) + _edge_transfer_loss(
        candidate_node,
        target_node,
        nodes2edge_between_map,
    )


def find_structured_merge_target(network: Network, candidate_node: ARNode, layer_index: int, nodes2edge_between_map):
    candidate_type = _node_type_key(candidate_node)
    candidate_targets = []
    for node in network.layers[layer_index].nodes:
        if node.name == candidate_node.name or node.deleted:
            continue
        if _node_type_key(node) != candidate_type:
            continue
        gap = _merge_gap(candidate_node, node)
        precision_loss = _candidate_precision_loss(candidate_node, node, nodes2edge_between_map)
        candidate_targets.append((node, precision_loss, abs(_bias_adjustment(candidate_node, gap)), abs(gap), gap))
    if not candidate_targets:
        return None
    candidate_targets.sort(key=lambda item: (item[1], item[2], item[3], item[0].name))
    return candidate_targets[0][0], candidate_targets[0][4], candidate_targets[0][1]


def _add_or_update_transferred_out_edge(
        network: Network,
        target_node: ARNode,
        dest_name: str,
        added_weight: float,
) -> None:
    for target_out_edge in target_node.out_edges:
        if target_out_edge.dest == dest_name:
            target_out_edge.weight += added_weight
            return

    new_edge = Edge(src=target_node.name, dest=dest_name, weight=added_weight)
    target_node.out_edges.append(new_edge)
    if dest_name in network.name2node_map:
        network.name2node_map[dest_name].in_edges.append(new_edge)


def structured_merge_delete_candidate(
        network: Network,
        candidate_node: ARNode,
        target_node: ARNode,
        layer_index: int,
        gap: float,
) -> None:
    deleted_snapshot = cPickle.loads(cPickle.dumps(candidate_node, -1))
    deleted_snapshot.deleted = True
    network.deleted_name2node[candidate_node.name] = deleted_snapshot

    for out_edge in list(candidate_node.out_edges):
        _add_or_update_transferred_out_edge(
            network=network,
            target_node=target_node,
            dest_name=out_edge.dest,
            added_weight=float(out_edge.weight),
        )

    bias_adjustment = _bias_adjustment(candidate_node, gap)
    if bias_adjustment != 0:
        target_node.bias += bias_adjustment

    network.remove_node(candidate_node, layer_index)
    network.generate_name2node_map()
    network.biases = network.generate_biases()
    network.weights = network.generate_weights()


def try_structured_merge_delete(
        network: Network,
        candidate_node: ARNode,
        layer_index: int,
        nodes2edge_between_map,
):
    target_result = find_structured_merge_target(
        network=network,
        candidate_node=candidate_node,
        layer_index=layer_index,
        nodes2edge_between_map=nodes2edge_between_map,
    )
    if target_result is None:
        return None
    target_node, gap, precision_loss = target_result
    structured_merge_delete_candidate(
        network=network,
        candidate_node=candidate_node,
        target_node=target_node,
        layer_index=layer_index,
        gap=gap,
    )
    return target_node.name, gap, precision_loss
