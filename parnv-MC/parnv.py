import sys
import os
import json
import copy
import time
import argparse
import io
import contextlib
import warnings
from typing import Any
# import timeout_decorator
'''
python parnv.py \
  --mnist-classifier /home/gpu/yyc_projects/Prune/data/models/mnist/mnist_fc_relu_32x2.onnx \
  --verification-epsilon 0.02 \
  --verification-sample-index 1 \
  --dataset-split test \
  --dataset-root "/home/gpu/yyc_project /data" \
  -m marabou_with_ar \
  -a global \
  -r global
'''
'''
python parnv.py \
  --mnist-classifier /home/gpu/yyc_projects/Prune/data/models/mnist/mnist_fc_relu_32x2.onnx \
  --verification-epsilon 0.02 \
  --verification-sample-index 1 \
  --dataset-split test \
  --dataset-root "/home/gpu/yyc_project /data" \
  -m marabou
'''


try:
    import pandas as pd
except ImportError:
    pd = None

warnings.filterwarnings("ignore", category=UserWarning, module=r"maraboupy(\..*)?")

from core.import_marabou import dynamically_import_marabou
from core.configuration import consts
from experiments.consts import BEST_CEGARABOU_METHODS
from core.utils.debug_utils import debug_print
from core.utils.verification_properties_utils import (
    is_satisfying_assignment,
)
from core.utils.mnist_property_utils import (
    load_mnist_sample,
    build_mnist_adversarial_property,
    build_mnist_property_id,
)
from core.utils.cifar10_property_utils import (
    load_cifar10_sample,
    build_cifar10_adversarial_property,
    build_cifar10_property_id,
)
import _pickle as cPickle


def _emit_visible_line(message: str) -> None:
    output_stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    print(message, file=output_stream, flush=True)


def extract_network_structure(network: Any) -> dict:
    layers = getattr(network, "layers", None)
    if not isinstance(layers, list) or len(layers) == 0:
        return {
            "layer_sizes": [],
            "layer_types": [],
            "input_dim": None,
            "hidden_dims": [],
            "output_dim": None,
            "num_layers": 0,
            "num_hidden_layers": 0,
            "architecture": "",
        }

    layer_sizes = [int(len(getattr(layer, "nodes", []))) for layer in layers]
    layer_types = [str(getattr(layer, "type_name", "unknown")) for layer in layers]
    input_dim = int(layer_sizes[0]) if layer_sizes else None
    output_dim = int(layer_sizes[-1]) if layer_sizes else None
    hidden_dims = [int(size) for size in layer_sizes[1:-1]]
    architecture = " -> ".join(str(size) for size in layer_sizes)

    return {
        "layer_sizes": layer_sizes,
        "layer_types": layer_types,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "num_layers": int(len(layer_sizes)),
        "num_hidden_layers": int(max(len(layer_sizes) - 2, 0)),
        "architecture": architecture,
    }


def print_network_structure(network: Any, model_path: str | None = None) -> dict:
    return print_labeled_network_structure(network=network, label="verification", model_path=model_path)


def print_labeled_network_structure(network: Any, label: str, model_path: str | None = None) -> dict:
    structure = extract_network_structure(network)
    _emit_visible_line("{}_network_structure".format(label))
    if model_path is not None:
        _emit_visible_line("{}_model_path={}".format(label, model_path))
    _emit_visible_line("{}_architecture={}".format(label, structure["architecture"]))
    _emit_visible_line("{}_input_dim={}".format(label, structure["input_dim"]))
    _emit_visible_line("{}_hidden_dims={}".format(label, structure["hidden_dims"]))
    _emit_visible_line("{}_output_dim={}".format(label, structure["output_dim"]))
    _emit_visible_line("{}_layer_types={}".format(label, structure["layer_types"]))
    return structure


def generate_results_filename(
        nnet_filename, property_id, mechanism, refinement_type,
        abstraction_type, refinement_sequence, abstraction_sequence
):
    net_name = os.path.basename(nnet_filename)
    # the rest of the name is the parameters
    return "__".join(["experiment",
                      "NN_{}".format(net_name),
                      "PID_{}".format(property_id),
                      "M_{}".format(mechanism),
                      "R_{}".format(refinement_type),
                      "A_{}".format(abstraction_type),
                      "RS_{}".format(refinement_sequence),
                      "AS_{}".format(abstraction_sequence),
                      "DATETIME_{}".format(consts.cur_time_str)
                      ])


def build_custom_property(args):
    dataset = str(getattr(args, "dataset", "mnist")).lower()
    common_loader_args = {
        "dataset_root": args.dataset_root,
        "dataset_split": args.dataset_split,
        "sample_index": args.verification_sample_index,
    }
    if dataset == "mnist":
        sample, label = load_mnist_sample(**common_loader_args)
        property_id = build_mnist_property_id(
            dataset_split=args.dataset_split,
            sample_index=args.verification_sample_index,
            epsilon=args.verification_epsilon,
        )
        test_property = build_mnist_adversarial_property(
            sample=sample,
            label=label,
            epsilon=args.verification_epsilon,
            output_threshold=args.output_threshold,
        )
    elif dataset == "cifar10":
        sample, label = load_cifar10_sample(**common_loader_args)
        property_id = build_cifar10_property_id(
            dataset_split=args.dataset_split,
            sample_index=args.verification_sample_index,
            epsilon=args.verification_epsilon,
        )
        test_property = build_cifar10_adversarial_property(
            sample=sample,
            label=label,
            epsilon=args.verification_epsilon,
            output_threshold=args.output_threshold,
        )
    else:
        raise ValueError("Unsupported dataset '{}'.".format(dataset))

    sample_metadata = {
        "dataset": dataset,
        "sample": sample,
        "label": label,
        "property_id": property_id,
        "sample_index": args.verification_sample_index,
        "dataset_split": args.dataset_split,
        "epsilon": args.verification_epsilon,
        "output_threshold": args.output_threshold,
    }
    return test_property, sample_metadata


def build_custom_mnist_property(args):
    """Backward-compatible alias for callers that used the old helper name."""
    if not hasattr(args, "dataset"):
        args.dataset = "mnist"
    if not hasattr(args, "output_threshold"):
        args.output_threshold = getattr(
            args,
            "mnist_output_threshold",
            2.220446049250313e-16,
        )
    return build_custom_property(args)


def _sample_metadata_result_pairs(sample_metadata):
    if sample_metadata is None:
        return []
    dataset = str(sample_metadata.get("dataset", "mnist")).lower()
    pairs = [
        ("dataset", dataset),
        ("sample_label", sample_metadata.get("label")),
        ("center_prediction", sample_metadata.get("center_prediction")),
    ]
    if dataset == "mnist":
        pairs.extend([
            ("mnist_label", sample_metadata.get("label")),
            ("mnist_center_prediction", sample_metadata.get("center_prediction")),
        ])
    elif dataset == "cifar10":
        pairs.extend([
            ("cifar10_label", sample_metadata.get("label")),
            ("cifar10_center_prediction", sample_metadata.get("center_prediction")),
        ])
    return pairs


def extract_query_result(one_exp_res):
    try:
        res_map = dict(one_exp_res)
    except Exception:
        return "UNKNOWN"
    return str(res_map.get("query_result", "UNKNOWN"))


def _write_experiment_result(
        results_directory,
        results_filename,
        res,
        verification_time_seconds,
):
    if pd is not None:
        df = pd.DataFrame.from_dict({x[0]: [x[1]] for x in res})
        df.to_json(os.path.join(results_directory, "df_" + results_filename))
    verification_result = extract_query_result(res)
    with open(
            os.path.join(results_directory, results_filename),
            "w",
            encoding="utf-8",
    ) as fw:
        fw.write("verification_result: {}\n".format(verification_result))
        fw.write(
            "verification_time_seconds: {:.6f}\n".format(
                float(verification_time_seconds)
            )
        )


# @timeout_decorator.timeout(72000)
def one_experiment(
        nnet_filename, refinement_type, abstraction_type, mechanism,
        refinement_sequence, abstraction_sequence, results_directory,
        property_id="", verbose=consts.VERBOSE,
        model_path=None, custom_test_property=None,
        mnist_metadata=None, sample_metadata=None,
        run_args=None, marabou_timeout_seconds=1200
):
    """

    Args:
        nnet_filename:
        refinement_type: "cegar" or "global"
        abstraction_type:
        mechanism: "marabou" otherwise marabou_with_ar
        refinement_sequence:
        abstraction_sequence:
        results_directory:
        property_id:
        verbose:

    Returns:
        res: experiment results

    """
    verification_started = time.perf_counter()
    if sample_metadata is None:
        sample_metadata = mnist_metadata
    if custom_test_property is None:
        raise ValueError("Image robustness verification requires a generated custom_test_property")
    if model_path is None:
        raise ValueError("Image robustness verification requires model_path")
    test_property = copy.deepcopy(custom_test_property)
    dynamically_import_marabou(query_type=test_property["type"])
    from core.nnet.read_nnet import network_from_nnet_file
    from core.nnet.read_nnet import network_from_onnx_file
    from core.utils.marabou_query_utils import reduce_property_to_basic_form, get_query
    from core.pre_process.crown_bounds import (
        compute_crown_hidden_layer_bounds,
        compute_crown_output_bounds,
    )
    from core.pre_process.dead_relu_pruning import (
        apply_input_bounds_from_property,
        classify_crown_output_bounds,
        prune_dead_relu_neurons,
    )
    fullname = model_path

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    results_filename = generate_results_filename(nnet_filename=nnet_filename,
                                                 property_id=property_id,
                                                 mechanism=mechanism,
                                                 refinement_type=refinement_type,
                                                 abstraction_type=abstraction_type,
                                                 refinement_sequence=refinement_sequence,
                                                 abstraction_sequence=abstraction_sequence)

    # for i in range(len(test_property["output"])):
    #     test_property["output"][i][1]["Lower"] = lower_bound
    # net  = network_from_nnet_file(fullname)

    model_suffix = os.path.splitext(fullname)[1].lower()
    if model_suffix == ".onnx":
        net = network_from_onnx_file(fullname)
    elif model_suffix == ".nnet":
        net = network_from_nnet_file(fullname)
    else:
        raise ValueError(
            "Unsupported model format '{}'. Please provide an .onnx or .nnet file.".format(model_suffix)
        )
    # test_accuraccy(net)
    # return
    network_structure = print_network_structure(net, model_path=fullname)

    if sample_metadata is not None:
        sample_size = len(sample_metadata["sample"])
        if network_structure["input_dim"] != sample_size:
            raise ValueError(
                "{} sample has {} inputs, but model '{}' expects {}.".format(
                    sample_metadata.get("dataset", "image"),
                    sample_size,
                    fullname,
                    network_structure["input_dim"],
                )
            )
        if network_structure["output_dim"] != 10:
            raise ValueError(
                "Image robustness verification expects an original 10-logit classifier; "
                "model '{}' has {} outputs.".format(
                    fullname,
                    network_structure["output_dim"],
                )
            )
        sample_dict = {index: float(value) for index, value in enumerate(sample_metadata["sample"])}
        center_output = net.speedy_evaluate(sample_dict)
        center_prediction = int(center_output.argmax())
        sample_metadata["center_prediction"] = center_prediction
        dataset_name = str(sample_metadata.get("dataset", "mnist"))
        print(
            "{} sample {} {} label={} center_prediction={}".format(
                dataset_name,
                sample_metadata.get("sample_index", "custom"),
                sample_metadata.get("dataset_split", ""),
                sample_metadata["label"],
                center_prediction,
            )
        )

    net, test_property = reduce_property_to_basic_form(network=net, test_property=test_property)
    apply_input_bounds_from_property(network=net, test_property=test_property)

    effective_marabou_timeout_seconds = (
        1200
        if marabou_timeout_seconds is None
        else int(marabou_timeout_seconds)
    )
    print("verification flow: CROWN dead-ReLU pruning -> Marabou")
    print("marabou_timeout_seconds={}".format(effective_marabou_timeout_seconds))

    crown_start_time = time.time()
    crown_hidden_bounds = compute_crown_hidden_layer_bounds(net)
    crown_output_bounds = compute_crown_output_bounds(net)
    crown_time = time.time() - crown_start_time
    crown_precheck_status, crown_precheck_reason = classify_crown_output_bounds(
        test_property=test_property,
        output_bounds=crown_output_bounds,
    )
    crown_hidden_preactivation_upper_bounds = (
        crown_hidden_bounds.preactivation_upper_bounds_by_hidden_layer
    )
    is_adversarial_property = test_property.get("type") == "adversarial"
    adversarial_query_mode = test_property.get(
        "_adversarial_query_mode",
        "disjunction" if is_adversarial_property else "",
    )
    adversarial_violation_operator = test_property.get(
        "_adversarial_violation_operator",
        "ge" if is_adversarial_property else "",
    )

    base_res = [
        ("net_name", nnet_filename),
        ("property_id", property_id),
        ("verification_flow", "crown_dead_relu_pruning_then_marabou"),
        ("crown_time", crown_time),
        ("crown_precheck_status", crown_precheck_status),
        ("crown_precheck_reason", crown_precheck_reason),
        ("crown_output_lower_bounds", json.dumps(crown_output_bounds.lower_bounds)),
        ("crown_output_upper_bounds", json.dumps(crown_output_bounds.upper_bounds)),
        ("crown_hidden_preactivation_upper_bounds", json.dumps(crown_hidden_preactivation_upper_bounds)),
        ("adversarial_output_order", test_property.get("_adversarial_output_order", "")),
        ("adversarial_query_mode", adversarial_query_mode),
        ("adversarial_margin", test_property.get("_adversarial_margin", "")),
        ("adversarial_violation_operator", adversarial_violation_operator),
        ("network_structure", json.dumps(network_structure)),
        ("marabou_timeout_seconds", effective_marabou_timeout_seconds),
        ("num_of_refine_steps", 0),
        ("abstraction_time", 0.0),
        ("total_ar_query_time", 0.0),
    ]
    base_res.extend(_sample_metadata_result_pairs(sample_metadata))

    if crown_precheck_status in ["SAT", "UNSAT"]:
        print("CROWN precheck concluded {}: {}".format(crown_precheck_status, crown_precheck_reason))
        res = base_res + [
            ("query_result", crown_precheck_status),
            ("orig_query_time", 0.0),
            ("pruned_query_time", 0.0),
            ("dead_relu_pruning_report", json.dumps({"total_pruned": 0, "pruned_by_layer": [], "skipped_layers": []})),
            ("pruned_network_structure", json.dumps(network_structure)),
            ("counter-example", []),
        ]
        _write_experiment_result(
            results_directory=results_directory,
            results_filename=results_filename,
            res=res,
            verification_time_seconds=time.perf_counter() - verification_started,
        )
        return res

    pruning_report = prune_dead_relu_neurons(
        network=net,
        crown_bounds=crown_hidden_bounds,
        preactivation_upper_threshold=0.0,
    )
    pruned_network_structure = print_labeled_network_structure(
        network=net,
        label="post_crown_pruning",
    )

    print("query using Marabou on CROWN-pruned network")
    print("dead ReLU neurons pruned: {}".format(pruning_report["total_pruned"]))
    t0 = time.time()
    vars1, stats1, query_result = get_query(
        network=net,
        test_property=test_property,
        verbose=consts.VERBOSE,
        marabou_timeout_seconds=effective_marabou_timeout_seconds,
    )
    t1 = time.time()
    pruned_query_time = t1 - t0
    counter_example = vars1 if query_result == "SAT" else []
    if verbose:
        print("pruned network query time = {}".format(pruned_query_time))

    res = base_res + [
        ("query_result", query_result),
        ("orig_query_time", 0.0),
        ("pruned_query_time", pruned_query_time),
        ("dead_relu_pruning_report", json.dumps(pruning_report)),
        ("pruned_network_structure", json.dumps(pruned_network_structure)),
        ("last_net_data", json.dumps(net.get_general_net_data())),
        ("counter-example", counter_example),
    ]
    _write_experiment_result(
        results_directory=results_directory,
        results_filename=results_filename,
        res=res,
        verification_time_seconds=time.perf_counter() - verification_started,
    )
    return res

    # The original abstraction/refinement implementation is intentionally not
    # reached by the new verification flow.

    # if mechanism is vanilla marabou
    if mechanism == "marabou":
        print("query using vanilla Marabou")
        print("net.get_general_net_data()")
        print(net.get_general_net_data())

        t0 = time.time()
        vars1, stats1, query_result = get_query(
            network=net,
            test_property=test_property,
            verbose=consts.VERBOSE,
            marabou_timeout_seconds=marabou_timeout_seconds,
        )
        t1 = time.time()
        # time to check property on net with marabou
        marabou_time = t1 - t0
        if verbose:
            print(f"query time = {marabou_time}")
        # if vars1:
        #     o_net = copy.deepcopy(net)
        #     net_output = o_net.speedy_evaluate(vars1)
        #     print(net_output[0])

        res = [
            ("net_name", nnet_filename),
            ("property_id", property_id),
            ("query_result", query_result),
            ("orig_query_time", marabou_time),
            ("net_data", json.dumps(net.get_general_net_data())),
            ("network_structure", json.dumps(network_structure)),
        ]
        res.extend(_sample_metadata_result_pairs(sample_metadata))
        _write_experiment_result(
            results_directory=results_directory,
            results_filename=results_filename,
            res=res,
            verification_time_seconds=time.perf_counter() - verification_started,
        )
        return res

    # otherwise mechanism is marabou_with_ar
    orig_net = copy.deepcopy(net)
    print("query using Marabou with AR")
    t2 = time.time()
    # do_process_before(net,property_id)
    random_input = [1]
    if abstraction_type == "global":
        net_before, random_input, processed_net, sat, actions = \
            global_abstraction_based_on_contribution(network=net, test_property=test_property)
        # print(net_before)
        net_before_p = cPickle.loads(cPickle.dumps(net_before, -1))
        # print('net_after-propagation')
        propagation_net(net_before)
        net = net_before
        # print(net_before)
    elif abstraction_type == "kmeans":
        net_before, random_input, processed_net, sat, actions = \
            kmeans_abstraction_based_on_contribution(network=net, test_property=test_property)
        # print(net_before)
        net_before_p = cPickle.loads(cPickle.dumps(net_before, -1))
        # print('net_after-propagation')
        propagation_net(net_before)
        net = net_before
    elif abstraction_type == "complete":
        net, processed_net = abstract_network(net)
    elif abstraction_type == "heuristic_alg2":
        net, sat = heuristic_abstract_alg2(
            network=net,
            test_property=test_property,
            sequence_length=abstraction_sequence
        )
    # elif abstraction_type == "heuristic_random":
    #     net = heuristic_abstract_random(
    #         network=net,
    #         test_property=test_property,
    #         sequence_length=abstraction_sequence
    #     )
    # elif abstraction_type == "heuristic_clustering":
    #     net = heuristic_abstract_clustering(
    #         network=net,
    #         test_property=test_property,
    #         sequence_length=abstraction_sequence
    #     )
    else:
        raise NotImplementedError("unknown abstraction")

    abstraction_network_structure = print_labeled_network_structure(
        network=net,
        label="post_abstraction",
    )

    # print(net)
    if (not sat) and random_input:
        abstraction_time = time.time() - t2
        num_of_refine_steps = 0
        ar_times = []
        ar_sizes = []
        refine_sequence_times = []
        spurious_examples = []
        while True:  # CEGAR / CETAR method
            t4 = time.time()

            # print("net.get_general_net_data()")
            print(net.get_general_net_data())

            vars1, stats1, query_result = get_query(
                network=net, test_property=test_property,
                verbose=consts.VERBOSE,
                marabou_timeout_seconds=marabou_timeout_seconds,
            )
            debug_print(f'query_result={query_result}')
            t5 = time.time()
            ar_times.append(t5 - t4)
            ar_sizes.append(net.get_general_net_data()["num_nodes"])
            # if verbose:
            print("query time after A and {} R steps is {}".format(num_of_refine_steps, t5 - t4))
            debug_print(net.get_general_net_data())
            if query_result == "UNSAT":
                # Validate UNSAT on the concrete network to avoid unsound early UNSAT
                # results caused by abstraction artifacts.
                orig_vars, _, orig_query_result = get_query(
                    network=orig_net,
                    test_property=test_property,
                    verbose=consts.VERBOSE,
                    marabou_timeout_seconds=marabou_timeout_seconds,
                )
                if orig_query_result != "UNSAT":
                    query_result = orig_query_result
                    if query_result == "SAT":
                        counter_example = orig_vars
                    if verbose:
                        print("abstract net UNSAT but original net is {} (use original result)".format(query_result))
                    break
                if verbose:
                    print("UNSAT (finish)")
                break
            if query_result == "SAT":
                if verbose:
                    print("SAT (have to check example on original net)")
                    print(vars1)
                # print(vars1)
                # debug_print(f'vars1={vars1}')
                # st = time.time()
                # orig_net_output = orig_net.evaluate(vars1)
                # print("evaluate: {}".format(time.time() - st))
                # st = time.time()
                orig_net_output = orig_net.speedy_evaluate(vars1)
                cur_net_output = net.speedy_evaluate(vars1)
                before_output = net_before_p.speedy_evaluate(vars1)
                print("current_output_value")
                print(cur_net_output)
                print(before_output)
                # print(f"orig_net_output={orig_net_output}")
                # print(f"orig_net.name2node_map={orig_net.name2node_map}")
                # print("speedy evaluate: {}".format(time.time() - st))
                nodes2variables, variables2nodes = orig_net.get_variables()
                # we got y'>3.99, check if also y'>3.99 for the same input
                if is_satisfying_assignment(network=orig_net,
                                            test_property=test_property,
                                            output=orig_net_output,
                                            variables2nodes=variables2nodes):

                    if verbose:
                        print("property holds also in orig - SAT (finish)")
                    counter_example = vars1
                    break  # also counter example for orig_net
                else:
                    spurious_examples.append(vars1)
                    t_cur_refine_start = time.time()
                    if verbose:
                        print("property doesn't holds in orig - spurious example")
                    num_of_refine_steps += 1
                    if verbose:
                        print("refine step #{}".format(num_of_refine_steps))
                    # refine until all spurious examples are satisfied
                    # since all spurious examples are satisfied in the original
                    # network, the loop stops until net will be fully refined
                    # print(vars1)
                    example = vars1
                    if abstraction_type != "heuristic_alg2":
                        ori_var2val = processed_net.evaluate(example)
                    refinement_sequences_counter = 0
                    refinement_exhausted = False
                    while True:
                        refinement_sequences_counter += 1
                        # print(f"refinement_sequences_counter={refinement_sequences_counter}")
                        if refinement_type == "cegar":
                            debug_print("cegar")
                            net = refine(network=net,
                                         sequence_length=refinement_sequence,
                                         example=vars1)
                        # else:
                        #     debug_print("weight_based")
                        #     net = refine(network=net,
                        #                  sequence_length=refinement_sequence)
                        elif refinement_type == "global":
                            debug_print("global")
                            net = global_refine(network=net_before_p, processed_net=processed_net,
                                                ori_var2val=ori_var2val, actions=actions, example=vars1)
                            if getattr(net, "global_refine_no_candidate", False):
                                if verbose:
                                    print("global refinement has no candidate; fallback to original network query")
                                orig_vars, _, query_result = get_query(
                                    network=orig_net,
                                    test_property=test_property,
                                    verbose=consts.VERBOSE,
                                    marabou_timeout_seconds=marabou_timeout_seconds,
                                )
                                if query_result == "SAT":
                                    counter_example = orig_vars
                                refinement_exhausted = True
                                break

                        # after refining, check if the current spurious example is
                        # already not a counter example (i.e. not satisfied in the
                        # refined network). stop if not satisfied, continue if yes
                        net_output = net.speedy_evaluate(vars1)
                        # print(f"net_output={net_output}")
                        # print(f"net.name2node_map={net.name2node_map}")
                        nodes2variables, variables2nodes = net.get_variables()
                        if not is_satisfying_assignment(
                                network=net,
                                test_property=test_property,
                                output=net_output,
                                variables2nodes=variables2nodes):
                            net_before_p = cPickle.loads(cPickle.dumps(net, -1))
                            propagation_net(net)
                            break
                    if refinement_exhausted:
                        break
                    # print(net)
                    t_cur_refine_end = time.time()
                    refine_sequence_times.append(t_cur_refine_end - t_cur_refine_start)
    elif sat:
        query_result = "SAT"
        abstraction_time = 0
        ar_times = [0.0]
        ar_sizes = [net.get_general_net_data()["num_nodes"]]
        refine_sequence_times = []
        num_of_refine_steps = 0
    else:
        # This branch can be reached when abstraction exits early (for example,
        # interval checks). Do not infer UNSAT without a concrete solver call.
        t_fallback_query_start = time.time()
        vars1, stats1, query_result = get_query(
            network=orig_net,
            test_property=test_property,
            verbose=consts.VERBOSE,
            marabou_timeout_seconds=marabou_timeout_seconds,
        )
        t_fallback_query_end = time.time()
        if query_result == "SAT":
            counter_example = vars1
        abstraction_time = 0
        ar_times = [t_fallback_query_end - t_fallback_query_start]
        ar_sizes = [net.get_general_net_data()["num_nodes"]]
        refine_sequence_times = []
        num_of_refine_steps = 0
    refinement_network_structure = print_labeled_network_structure(
        network=net,
        label="post_refinement",
    )
    t3 = time.time()

    # time to check property on net with marabou using CEGAR
    total_ar_time = t3 - t2
    if verbose:
        print("ar query time = {}".format(total_ar_time))

    # time to check property on the last queried network in CEGAR
    # (some branches skip the query loop, so fall back to 0.0)
    last_net_ar_time = float(ar_times[-1]) if ar_times else 0.0
    if verbose:
        print("last ar net query time = {}".format(last_net_ar_time))

    res = [
        ("net_name", nnet_filename),
        ("property_id", property_id),
        ("abstraction_time", abstraction_time),
        ("query_result", query_result),
        ("num_of_refine_steps", num_of_refine_steps),
        ("total_ar_query_time", total_ar_time),
        ("ar_times", json.dumps(ar_times)),
        ("ar_sizes", json.dumps(ar_sizes)),
        ("refine_sequence_times", json.dumps(refine_sequence_times)),
        ("last_net_data", json.dumps(net.get_general_net_data())),
        ("network_structure", json.dumps(network_structure)),
        ("abstraction_network_structure", json.dumps(abstraction_network_structure)),
        ("refinement_network_structure", json.dumps(refinement_network_structure)),
        ("counter-example", counter_example)
        # ("last_query_time", last_net_ar_time)
    ]
    res.extend(_sample_metadata_result_pairs(sample_metadata))
    _write_experiment_result(
        results_directory=results_directory,
        results_filename=results_filename,
        res=res,
        verification_time_seconds=time.perf_counter() - verification_started,
    )
    return res


def parse_args():
    # parse args
    parser = argparse.ArgumentParser(
        description="Verify one MNIST or CIFAR-10 robustness property."
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset family used to load the verification sample.",
    )
    parser.add_argument("-m", "--mechanism",
                        dest="mechanism",
                        default="marabou_with_ar",  # "marabou_with_ar",
                        choices=["marabou", "marabou_with_ar"],
                        type=str)
    parser.add_argument("-a", "--abstraction_type",
                        dest="abstraction_type",
                        default="naive",  # BEST_CEGARABOU_METHODS["A"],
                        choices=["naive", "alg2", "random", "clustering", "global", "kmeans"])
    parser.add_argument("-r", "--refinement_type",
                        dest="refinement_type",
                        default=BEST_CEGARABOU_METHODS["R"],
                        choices=["cegar", "weight_based", "global"])
    parser.add_argument("-as", "--abstraction_sequence",
                        dest="abstraction_sequence",
                        type=int,
                        default=BEST_CEGARABOU_METHODS["AS"],
                        choices=[100, 250])
    parser.add_argument("-rs", "--refinement_sequence",
                        dest="refinement_sequence",
                        type=int,
                        default=BEST_CEGARABOU_METHODS["RS"],
                        choices=[50, 100])
    parser.add_argument("-d", "--results_directory",
                        dest="results_directory",
                        default=consts.results_directory)
    parser.add_argument("--classifier",
                        "--mnist-classifier",
                        "--cifar10-classifier",
                        dest="classifier",
                        required=True,
                        help="Path to a 10-output MNIST/CIFAR-10 classifier in .onnx or .nnet format")
    parser.add_argument("--verification-epsilon",
                        dest="verification_epsilon",
                        type=float,
                        required=True,
                        help="L-infinity perturbation radius for image robustness verification")
    parser.add_argument("--verification-sample-index",
                        dest="verification_sample_index",
                        type=int,
                        required=True,
                        help="Sample index inside the selected dataset split")
    parser.add_argument("--dataset-split",
                        dest="dataset_split",
                        required=True,
                        choices=["train", "test"],
                        help="Whether the sample comes from the train or test split")
    parser.add_argument("--dataset-root",
                        dest="dataset_root",
                        required=True,
                        help="Root directory containing MNIST IDX files or CIFAR-10 Python batches")
    parser.add_argument("--marabou-timeout-seconds",
                        dest="marabou_timeout_seconds",
                        type=int,
                        default=1200,
                        help="Marabou solver timeout in seconds.")
    parser.add_argument("--output-threshold",
                        "--mnist-output-threshold",
                        "--cifar10-output-threshold",
                        dest="output_threshold",
                        type=float,
                        default=2.220446049250313e-16,
                        help="Threshold used for pairwise outputs: output[i] >= threshold.")
    args = parser.parse_args()

    # patch: old names are complete/heuristic, new names are naive/alg1 resp.
    if args.abstraction_type == "alg2":
        args.abstraction_type = "heuristic_alg2"
    elif args.abstraction_type == "random":
        args.abstraction_type = "heuristic_random"
    # if args.abstraction_type == "clustering":
    #     args.abstraction_type = "heuristic_clustering"
    elif args.abstraction_type == "naive":
        args.abstraction_type = "complete"

    # patch: old names are cegar/cetar, new names are cegar/weight_based resp.
    if args.refinement_type == "weight_based":
        args.refinement_type = "cetar"

    return args


if __name__ == '__main__':
    program_start_time = time.perf_counter()
    args = parse_args()
    custom_test_property, sample_metadata = build_custom_property(args)
    run_property_id = sample_metadata["property_id"]
    model_path = args.classifier
    nnet_filename = os.path.basename(args.classifier)

    stdout_buffer = io.StringIO()
    stdout_context = contextlib.nullcontext() if consts.VERBOSE else contextlib.redirect_stdout(stdout_buffer)
    with stdout_context:
        one_exp_res = one_experiment(
            nnet_filename=nnet_filename,
            property_id=run_property_id,
            mechanism=args.mechanism,
            refinement_type=args.refinement_type,
            abstraction_type=args.abstraction_type,
            refinement_sequence=args.refinement_sequence,
            abstraction_sequence=args.abstraction_sequence,
            results_directory=args.results_directory,
            model_path=model_path,
            custom_test_property=custom_test_property,
            sample_metadata=sample_metadata,
            run_args=vars(args),
            marabou_timeout_seconds=args.marabou_timeout_seconds)

    total_verification_time = time.perf_counter() - program_start_time
    verification_result = extract_query_result(one_exp_res)
    print("total_verification_time_seconds: {:.6f}".format(total_verification_time))
    print("verification_result: {}".format(verification_result))
