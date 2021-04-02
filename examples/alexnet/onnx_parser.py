import torch
import onnx
from onnx import numpy_helper

import pdb

onnx_model = onnx.load("test.onnx")

weights = onnx_model.graph.initializer
weights_dims = {}
weights_vals = {}

for w in weights:
	weights_dims[w.name] = w.dims
	weights_vals[w.name] = numpy_helper.to_array(w)


def create_weight_struct(model, class_name):
	'''

	struct AlexNet
	{
		t4::tensor4f features_0_weight;
	  ...
		t4::tensor1f classifier_6_bias;
	};

	'''

	# Include header file
	output = '#include "tensor4.h"\n'
	output += "\n"

	# Define class structure
	output += "struct " + class_name + " {\n"
	for w in weights:
		output += "    t4::tensor{}f {}\n".format(len(w.dims), w.name)
	output += "};\n"
	output += "\n"
	output += "AlexNet AlexNetLoad(const char* filename);\n"
	output += "\n"

	# Define forward function
	output += "t4::tensor2f {}Forward(const {}& ctx, t4::tensor4f xinput_1);\n".format(class_name, class_name)

	return output

output = create_weight_struct(onnx_model, "AlexNet")
print(output)



