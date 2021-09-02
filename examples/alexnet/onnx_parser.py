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
#
# nodes = onnx_model.graph.node
# dir(onnx_model)
# ['graph', 'ir_version', 'model_version', 'opset_import', 'producer_name', 'producer_version']
# dir(onnx_model.graph)
# ['initializer', 'input', 'name', 'node', 'output']


import argparse
import os
import sys

import torch
import onnx
from onnx import numpy_helper

import pdb

class GraphParser:
	'''
	make_graph(nodes,name,inputs,outputs,initializer=None,doc_string=None,value_info=[])
	    nodes: NodeProto list, e.g:	[node1,node2,node3,…]
	    name: String
	    inputs:	ValueInfoProto list
	    outputs: ValueInfoProto list
	    initializer: TensorProto list
	    doc_string: String
	    value_info: ValueInfoProto list
	'''

	def __init__(self, graph):
		self.graph = graph

		self.used_vars = set()

		self.var_type = {}
		self.var_shape = {}

		self.node_attr_type = {}
		self.node_attr_ints = {}

		self.const_type = {}
		self.const_dims = {}

	def parse(self):
		# int used_vars
		for o in self.graph.output:
			# etype = o.type.tensor_type.elem_type
			# ndims = o.type.tensor_type.shape
			self.used_vars.add(o.name)

		node_list = [n for n in self.graph.node]
		need_checking = True
		while need_checking:
			need_checking = False
			for n in node_list:
				if self.node_used(n):
					# Add all node.input to used_vars
					for i in n.input:
						self.used_vars.add(i)
					need_checking = True
					node_list.remove(n)
					break

		for i in self.graph.input:
			# name = i.name
			# etype = i.type.tensor_type.elem_type
			# shape = i.type.tensor_type.shape
			print("i -- ", i)
			name, dtype, shape = self.tensor_value_info_parse(i)
			self.var_type[name] = dtype
			self.var_shape[name] = shape
			print(name, dtype, shape)

		for o in self.graph.output:
			# name = graph_input[i].name
			# etype = system_t_data_type_names[graph_input[i].type.tensor_type.elem_type]
			# ndims = str(graph_input[i].type.tensor_type.shape).count("dim {")
			print("o --- ", o)
			name, dtype, shape = self.tensor_value_info_parse(o)
			self.var_type[name] = dtype
			self.var_shape[name] = shape
			print(name, dtype, shape)

		for w in self.graph.initializer:
			# TensorProto
			print(w.name, w.data_type, w.dims)
			self.const_type[w.name] = w.data_type
			self.const_dims[w.name] = w.dims

			# w_step = 16
			# w_name = "w_" + w.name.replace(".", "_") + "_buf"
			# print(f"uint8_t {w_name}[] = {{ ")
			# for i in range(0, len(w.raw_data), w_step):
			# 	w_bytes = ', '.join(['0x%02x'%b for b in w.raw_data[i : i + w_step]])
			# 	print(f"    {w_bytes},")
			# print(f"}};")

		for n in self.graph.node:
			print(n)
			attr_type, attr_ints = self.node_attribute_parse(n)
			self.node_attr_type[n.name] = attr_type
			self.node_attr_ints[n.name] = attr_ints

	def var_used(self, name):
		return name in self.used_vars

	def node_used(self, node):
		for o in node.output:
			if self.var_used(o):
				return True
		return False

	def tensor_value_info_parse(self, t):
		# make_tensor_value_info(name,elem_type,shape,doc_string="",shape_denotation=None) --> ValueInfoProto

		name = t.name
		etype = t.type.tensor_type.elem_type
		shape = t.type.tensor_type.shape

		return name, etype, shape

	def node_attribute_parse(self, node):
		# make_attribute(key,value,doc_string=None) --> AttributeProto
		attr_ints = {}
		attr_type = {}
		for a in node.attribute:
			if (len(a.ints)) == 0:
				attr_ints[a.name] = a.i
			else:
				attr_ints[a.name] = a.ints
			attr_type[a.name] = a.type
		return attr_type, attr_ints




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

system_node_operator_functions = {}

def register_attribute_functions(name, afunc):
	system_node_operator_functions[name] = afunc

def MaxPool(node, attr_type, attr_ints):
	ndim = len(attr_ints['kernel_shape'])
	parameters = []
	parameters += [str(i) for i in attr_ints['kernel_shape']]
	parameters += [str(i) for i in attr_ints['strides']]
	parameters += [str(i) for i in attr_ints['pads'][0:2]]

	output = []
	output.append("MaxPool{}d".format(ndim))
	output.append("<")
	output.append(", ".join(parameters))
	output.append(">")

	return "".join(output)

def Conv(node, attr_type, attr_ints):
	ndim = len(attr_ints['kernel_shape'])
	parameters = []
	parameters += [str(i) for i in attr_ints['kernel_shape']]
	parameters += [str(i) for i in attr_ints['strides']]
	parameters += [str(i) for i in attr_ints['pads'][0:2]]
	parameters += [str(i) for i in attr_ints['dilations']]

	output = []
	output.append("Conv{}d".format(ndim))
	output.append("<")
	output.append(", ".join(parameters))
	output.append(">")

	return "".join(output)


def AveragePool(node, attr_type, attr_ints):
	ndim = len(attr_ints['kernel_shape'])
	parameters = []
	parameters += [str(i) for i in attr_ints['kernel_shape']]
	parameters += [str(i) for i in attr_ints['strides']]

	output = []
	output.append("AveragePool{}d".format(ndim))
	output.append("<")
	output.append(", ".join(parameters))
	output.append(">")

	return "".join(output)

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

def get_node_operators(node):
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

	# if node.op_type == 'Clip':
	# 	pdb.set_trace()


	if not node.op_type in system_node_operator_functions:
		return node.op_type


	attr_ints = {}
	attr_type = {}
	for a in node.attribute:
		if (len(a.ints)) == 0:
			attr_ints[a.name] = a.i
		else:
			attr_ints[a.name] = a.ints
		attr_type[a.name] = a.type

	# attr_ints
	# {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}


	output = system_node_operator_functions[node.op_type](node, attr_type, attr_ints)
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
		pdb.set_trace()
		
	output.append("};")
	output.append("")
	output.append("{} {}Load(const char* filename);\n".format(class_name, class_name))
	output.append("")

	# Define forward function
	output.append(get_forward_declare(onnx_model, class_name) + ";")

	# with open(class_name + ".h", "w") as source_h:
	# 	source_h.write("\n".join(output))
	print("--------------------------------------------------------")
	print("\n".join(output))
	print("--------------------------------------------------------")

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
		node_function = get_node_operators(node)
		output.append("	auto {} = t4::{}({})".format(
			", ".join(node_output), node_function, ", ".join(node_input)) + ";")

	output.append("")
	output.append("	return {};".format(", ".join(node_output)))
	#}
	output.append("}")

	# with open(class_name + ".cpp", "w") as source_cpp:
	# 	source_cpp.write("\n".join(output))
	print("--------------------------------------------------------")
	print("\n".join(output))
	print("--------------------------------------------------------")

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


	register_attribute_functions("Conv", Conv)
	register_attribute_functions("MaxPool", MaxPool)
	register_attribute_functions("AveragePool", AveragePool)

	model = onnx.load(args.model)
	onnx_parser = GraphParser(model.graph)
	onnx_parser.parse()
	pdb.set_trace()

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
