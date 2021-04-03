"""Onnx Model Tools."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***	Copyright Dell 2021, All Rights Reserved.
# ***
# ***	File Author: Dell, 2021年 03月 23日 星期二 12:42:57 CST
# ***
# ************************************************************************************/
#

import argparse
import os
import sys

import torch
import onnx
from onnx import numpy_helper

import pdb

# https://pytorch.org/docs/stable/onnx.html
# https://github.com/onnx/onnx
# https://github.com/onnx/onnx/blob/master/docs/IR.md

# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3
system_c_data_type_names = {
	0: "UNDEFINED",
	1: "float",
	2: "uint8_t",
	3: "int8_t",
	4: "uint16_t",
	5: "int16_t",
	6: "int32_t",
	7: "int64_t",
	8: "string",
	9: "bool",
	10: "FLOAT16",
	11: "double",
	12: "uint32_t",
	13: "uint64_t",
	14: "COMPLEX64",
	15: "COMPLEX128",
	16: "BFLOAT16"
}

system_t_data_type_names = {
	0: "?", # "UNDEFINED",
	1: "f", # "float",
	2: "?", # "uint8_t",
	3: "?", # "int8_t",
	4: "?", # "uint16_t",
	5: "?", # "int16_t",
	6: "i", # "int32_t",
	7: "?", # "int64_t",
	8: "?", # "string",
	9: "?",  # "bool",
	10: "?", # "FLOAT16",
	11: "?", # "double",
	12: "?", # "uint32_t",
	13: "?", # "uint64_t",
	14: "?", # "COMPLEX64",
	15: "?", # "COMPLEX128",
	16: "?", # "BFLOAT16"
}

system_attribute_functions = {}

def register_attribute_functions(name, afunc):
	system_attribute_functions[name] = afunc


def MaxPool(node, attr_type, attr_ints):
	pdb.set_trace()

	return ""
    # def __init__(self, lhs, rhs, vtable):
    #     Emitter.__init__(self, lhs, rhs, vtable)
    #     self.make_output_same_as_first_arg()

    #     self.validate_arg_return_count(1, 1)

    #     self.append_parameter('kernel_shape', templated=True)
    #     self.append_parameter('strides', templated=True)
    #     self.append_parameter('pads', templated=True, padding=True)

    #     self.name = "MaxPool%dd" % self.get_dim()
register_attribute_functions("MaxPool", MaxPool)

# nodes = onnx_model.graph.node
# dir(onnx_model)
# ['graph', 'ir_version', 'model_version', 'opset_import', 'producer_name', 'producer_version']
# dir(onnx_model.graph)
# ['initializer', 'input', 'name', 'node', 'output']

def get_forward_args(graph_input):
	'''
		t4::tensor2f AlexNetForward(const AlexNet& ctx, t4::tensor4f xinput_1)
	'''

	output = []
	for i in range(len(graph_input)):
		name = graph_input[i].name
		etype = system_t_data_type_names[graph_input[i].type.tensor_type.elem_type]
		ndims = str(graph_input[i].type.tensor_type.shape).count("dim {")
		output.append("t4::tensor{}{} x{}".format(ndims, etype, name))

	return ", ".join(output)


def get_forward_return(graph_output):
	'''
		t4::tensor2f AlexNetForward(const AlexNet& ctx, t4::tensor4f xinput_1)
	'''
	output = []
	for i in range(len(graph_output)):
		name = graph_output[i].name
		etype = system_t_data_type_names[graph_output[i].type.tensor_type.elem_type]
		ndims = str(graph_output[i].type.tensor_type.shape).count("dim {")
		output.append("t4::tensor{}{}".format(ndims, etype))

	return ", ".join(output)

def get_forward_declare(model, class_name):
	'''
		t4::tensor2f AlexNetForward(const AlexNet& ctx, t4::tensor4f xinput_1)
	'''
	output = "{} {}Forward(const {}& ctx, {})".format(
		get_forward_return(model.graph.output),
		class_name, class_name,
		get_forward_args(model.graph.input))
	return output

def get_node_attribues(node):
	# [name: "dilations"
	# ints: [1, 1]
	# type: INTS
	# , name: "group"
	# i: 1
	# type: INT
	# , name: "kernel_shape"
	# ints: [11, 11]
	# type: INTS
	# , name: "strides"
	# ints: [4, 4]
	# type: INTS
	# ]
	if not node.name in system_attribute_functions:
		return ""

	attr_ints = {}
	attr_type = {}
	for a in node.attribute:
		attr_ints[a.name] = a.ints
		attr_type[a.name] = a.type

	output = system_attribute_functions[node.name](node, attr_type, attr_ints)
	return output

def create_head_file(onnx_model, class_name):
	'''

	struct AlexNet
	{
		t4::tensor4f features_0_weight;
	  ...
		t4::tensor1f classifier_6_bias;
	};

	'''

	# weights = model.graph.initializer

	print("Create file {}.h ...".format(class_name))

	# Include header file
	output = []
	output.append('#include "tensor4.h"')
	output.append("")

	# Define class structure
	output.append("struct " + class_name + " {")
	for w in onnx_model.graph.initializer:
		output.append("	t4::tensor{}f {};".format(len(w.dims), w.name))
	output.append("};")
	output.append("")
	output.append("{} {}Load(const char* filename);\n".format(class_name, class_name))
	output.append("")

	# Define forward function
	output.append(get_forward_declare(onnx_model, class_name) + ";")

	with open(class_name + ".h", "w") as source_h:
		source_h.write("\n".join(output))
	print("OK.")


def create_bin_file(onnx_model, class_name):
	print("Create file {}.bin ...".format(class_name))
	print("OK.")

def create_cpp_file(onnx_model, class_name):
	'''
	#include "AlexNet.h"

	AlexNet AlexNetLoad(const char* filename)
	{
		AlexNet ctx;
		t4::model_dict dict = t4::load(filename);
		dict.load(ctx.features_0_weight, "features.0.weight", 64, 3, 11, 11);
		。。。
		return ctx;
	}
	'''
	print("Create file {}.cpp ...".format(class_name))

	output = []
	output.append('#include "{}.h"'.format(class_name))
	output.append("")
	output.append("{} {}Load(const char *filename)".format(class_name, class_name))

	# {
	output.append("{")
	output.append("	{} ctx;".format(class_name))
	output.append("	t4::model_dict dict = t4::load(filename);")
	for w in onnx_model.graph.initializer:
		wstr = ", ".join([str(e) for e in w.dims])
		output.append('	dict.looad(ctx.{}, "{}", {});'.format(w.name, w.name, wstr))
	output.append("	return ctx;")
	# }
	output.append("}")

	output.append("")
	output.append("")

	'''
	t4::tensor2f AlexNetForward(const AlexNet& ctx, t4::tensor4f xinput_1)
	{
		t4::tensor4f x17 = t4::Conv2d<11, 11, 4, 4, 2, 2, 1, 1>(xinput_1, ctx.features_0_weight, ctx.features_0_bias); //features.0
		t4::release(xinput_1);
		return x43;
	}
	'''
	weights = {}
	for w in onnx_model.graph.initializer:
		weights[w.name] = w.dims

	output.append(get_forward_declare(onnx_model, class_name))
	# {
	output.append("{")
	for node in onnx_model.graph.node:
		node_input = [("ctr." + e) if e in weights else ("x" + e) for e in node.input]
		node_output = [("ctr." + e) if e in weights else ("x" + e) for e in node.output]
		node_attrs = get_node_attribues(node)
		output.append("	auto {} = t4::{}<{}>({})".format(
			", ".join(node_output), node.op_type, node_attrs, ", ".join(node_input)) + ";")

	output.append("")
	output.append("	return {};".format(", ".join(node_output)))
	#}
	output.append("}")

	with open(class_name + ".cpp", "w") as source_cpp:
		source_cpp.write("\n".join(output))

	print("OK.")


if __name__ == '__main__':
	"""Onnx tools ..."""

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-m', '--model', help="onnx model file", default='test.onnx')
	parser.add_argument('-n', '--network', help="network name", default='XAlexNet')

	args = parser.parse_args()

	if not os.path.exists(args.model):
		print("Onnx model does not exist, stop.")
		sys.exit(-1)

	model = onnx.load(args.model)
	if os.path.exists("{}.h".format(args.network)):
		print("File {}.h exist, stop.".format(args.network))
		sys.exit(-1)
	else:
		create_head_file(model, args.network)
		create_bin_file(model, args.network)

	if os.path.exists("{}.cpp".format(args.network)):
		print("File {}.cpp exist, stop.".format(args.network))
		sys.exit(-1)
	else:
		create_cpp_file(model, args.network)
