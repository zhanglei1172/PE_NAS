"""
There are three representations
'xnas': the NASBench201SearchSpace object
'op_indices': A list of six ints, which is the simplest representation
'arch_str': The string representation used in the original nasbench201 paper

This file currently has the following conversions:
xnas -> op_indices
op_indices -> xnas
xnas -> arch_str

Note: we could add more conversions, but this is all we need for now
"""

import torch

from xnas.core.primitives import AbstractPrimitive

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def convert_xnas_to_op_indices(xnas_object):

    cell = xnas_object._get_child_graphs(single_instances=True)[0]
    ops = []
    for i, j in EDGE_LIST:
        ops.append(cell.edges[i, j]["op"].get_op_name)

    return [OP_NAMES.index(name) for name in ops]


def convert_op_indices_to_xnas(op_indices, xnas_object):
    """
    Converts op indices to a xnas object
    input: op_indices (list of six ints)
    xnas_object is an empty NasBench201SearchSpace() object.
    Do not call this method with a xnas object that has already been
    discretized (i.e., all edges have a single op).

    output: none, but the xnas object now has all edges set
    as in genotype.

    warning: this method will modify the edges in xnas_object.
    """

    # create a dictionary of edges to ops
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        edge_op_dict[EDGE_LIST[i]] = OP_NAMES[index]

    def add_op_index(edge):
        # function that adds the op index from the dictionary to each edge
        if (edge.head, edge.tail) in edge_op_dict:
            for i, op in enumerate(edge.data.op):
                if op.get_op_name == edge_op_dict[(edge.head, edge.tail)]:
                    index = i
                    break
            edge.data.set("op_index", index, shared=True)

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the one in op_index
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        chosen_op = primitives[edge.data.op_index]
        primitives[edge.data.op_index] = update_batchnorms(chosen_op)

        edge.data.set("op", primitives[edge.data.op_index])
        edge.data.set("primitives", primitives)  # store for later use

    def update_batchnorms(op: AbstractPrimitive) -> AbstractPrimitive:
        """ Makes batchnorms in the op affine, if they exist """
        init_params = op.init_params
        has_batchnorm = False

        for module in op.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                has_batchnorm = True
                break

        if not has_batchnorm:
            return op

        if 'affine' in init_params:
            init_params['affine'] = True
        if 'track_running_stats' in init_params:
            init_params['track_running_stats'] = True

        new_op = type(op)(**init_params)
        return new_op

    xnas_object.update_edges(
        add_op_index, scope=xnas_object.OPTIMIZER_SCOPE, private_edge_data=False
    )

    xnas_object.update_edges(
        update_ops, scope=xnas_object.OPTIMIZER_SCOPE, private_edge_data=True
    )


def convert_xnas_to_str(xnas_object):
    """
    Converts xnas object to string representation.
    """

    ops_to_nb201 = {
        "AvgPool1x1": "avg_pool_3x3",
        "ReLUConvBN1x1": "nor_conv_1x1",
        "ReLUConvBN3x3": "nor_conv_3x3",
        "Identity": "skip_connect",
        "Zero": "none",
    }

    cell = xnas_object.edges[2, 3].op
    edge_op_dict = {
        (i, j): ops_to_nb201[cell.edges[i, j]["op"].get_op_name] for i, j in cell.edges
    }
    op_edge_list = [
        "{}~{}".format(edge_op_dict[(i, j)], i - 1)
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)
