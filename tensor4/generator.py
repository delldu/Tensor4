# Copyright 2018 Stanislav Pidhorskyi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .trace_parser import Parser
from .emitters import emitters
from .var_table import VarTable
import struct
import numpy as np
import sys
import torch
import torch.onnx
import torch.onnx.utils
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
import pdb
import onnx

def export_onnx(model, input):
    """Export onnx model."""

    onnx_file_name = "test.onnx"
    model.eval()

    # 2. Model export
    print("Exporting onnx model to {}...".format(onnx_file_name))

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(model, 
                    input,
                    onnx_file_name,
                    input_names=input_names,
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    keep_initializers_as_inputs=False,
                    export_params=True)



class GeneratorException(Exception):
    def __init__(self, message):
        self.message = message


def generate(module, args=tuple(), kwargs=None):
    def write_h(x, *args):
        source_h.write(x % args)
        
    def write_cpp(x, *args):
        #sys.stdout.write(x % args)
        source_cpp.write(x % args)

    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)

    export_onnx(module, args)

    onnx_model = onnx.load("test.onnx")
    
    # onnx_model.graph
    # onnx.helper.printable_graph(onnx_model.graph)
    # nodes =onnx_model.graph.node
    # for i in nodes: print(i.name)
    # for i in nodes: print(i.name, i.op_type, i.input, i.output)
    # ---->
    # Conv_0 Conv ['input', 'features.0.weight', 'features.0.bias'] ['17']
    # Relu_1 Relu ['17'] ['18']
    # MaxPool_2 MaxPool ['18'] ['19']
    # Conv_3 Conv ['19', 'features.3.weight', 'features.3.bias'] ['20']
    # Relu_4 Relu ['20'] ['21']
    # MaxPool_5 MaxPool ['21'] ['22']
    # Conv_6 Conv ['22', 'features.6.weight', 'features.6.bias'] ['23']
    # Relu_7 Relu ['23'] ['24']
    # Conv_8 Conv ['24', 'features.8.weight', 'features.8.bias'] ['25']
    # Relu_9 Relu ['25'] ['26']
    # Conv_10 Conv ['26', 'features.10.weight', 'features.10.bias'] ['27']
    # Relu_11 Relu ['27'] ['28']
    # MaxPool_12 MaxPool ['28'] ['29']
    # AveragePool_13 AveragePool ['29'] ['30']
    # Flatten_14 Flatten ['30'] ['31']
    # Gemm_15 Gemm ['31', 'classifier.1.weight', 'classifier.1.bias'] ['32']
    # Relu_16 Relu ['32'] ['33']
    # Gemm_17 Gemm ['33', 'classifier.4.weight', 'classifier.4.bias'] ['34']
    # Relu_18 Relu ['34'] ['35']
    # Gemm_19 Gemm ['35', 'classifier.6.weight', 'classifier.6.bias'] ['output']

    # (Pdb) len(nodes) -- 20
    # nodes[0]
    # len(nodes[0].attribute) -- 5
    # (Pdb) nodes[0].attribute[0]
    # name: "dilations"
    # ints: 1
    # ints: 1
    # type: INTS

    # (Pdb) nodes[0]
    # input: "input"
    # input: "features.0.weight"
    # input: "features.0.bias"
    # output: "17"
    # name: "Conv_0"
    # op_type: "Conv"
    # attribute {
    #   name: "dilations"
    #   ints: 1
    #   ints: 1
    #   type: INTS
    # }
    # attribute {
    #   name: "group"
    #   i: 1
    #   type: INT
    # }
    # attribute {
    #   name: "kernel_shape"
    #   ints: 11
    #   ints: 11
    #   type: INTS
    # }
    # attribute {
    #   name: "pads"
    #   ints: 2
    #   ints: 2
    #   ints: 2
    #   ints: 2
    #   type: INTS
    # }
    # attribute {
    #   name: "strides"
    #   ints: 4
    #   ints: 4
    #   type: INTS
    # }

    # weights = onnx_model.graph.initializer
    # from onnx import numpy_helper
    # len(weights) -- 16
    # numpy_helper.to_array(weights[0]).shape -- (4096,)
    # weights[15].name

    # pdb.set_trace()
    trace, out = torch.jit._get_trace_graph(module, args, kwargs)
    trace = torch.onnx.utils._optimize_graph(trace, OperatorExportTypes.ONNX)

    #print(str(trace))
    p = Parser()
    result = p.parse(str(trace))
    if result is None:
        raise Exception('Parsing error')

    inputs, statements, return_vars = result

    vtable = VarTable(module)
    pdb.set_trace()

    module_name = module.__class__.__name__

    # assign types for input vars
    for var_name, var_type, var_init in inputs:
        vtable.set_var_type(var_name, var_type, len(var_init))

    # create emitters
    # will populate vtable.init_list with parameters
    emitter_list = []
    for lhs, rhs in statements:
        op = rhs[0]
        if op not in emitters:
            raise GeneratorException('%s does not have an Emitter' % op)
        e = emitters[op](lhs, rhs, vtable)
        emitter_list.append(e)

    with open(module_name + ".cpp", "w") as source_cpp, open(module_name + ".h", "w") as source_h:
        write_h('#include "tensor4.h"' + '\n' * 3)
        write_cpp('#include "%s"' % (module_name + ".h") + '\n' * 3)

        write_h('struct %s\n{\n' % vtable.class_name)

        arguments = []

        not_used_vars = vtable.get_clean_list()

        for var_name, var_type, var_init in inputs:
            if var_name in vtable.init_list:
                var_cname = vtable.to_c_name(var_name).replace('ctx.', '')

                decl_str = "\t%s %s;\n" % (vtable.get_var_type(var_name), var_cname)

                write_h(decl_str)
            else:
                if vtable.is_var_used(var_name):
                    arguments.append(var_name)

        write_h('};' + '\n' * 3)

        declaration = '%s %sLoad(const char* filename)' % (vtable.class_name, vtable.class_name)
        write_h(declaration + ";\n\n")
        write_cpp(declaration + "\n{\n")
        write_cpp('\t%s ctx;\n', vtable.class_name)
        write_cpp('\tt4::model_dict dict = t4::load(filename);\n')

        for var_name, var_type, var_init in inputs:
            if var_name in vtable.init_list:
                var_cname = vtable.to_c_name(var_name)
                string = "\tdict.load(%s, \"%s\", %s);\n" % (var_cname, vtable.init_list[var_name][0], ', '.join([str(p) for p in var_init]))

                write_cpp(string)

        write_cpp('\treturn ctx;\n}' + '\n' * 3)
        
        if len(return_vars) == 1:
            return_var = '%s ' % vtable.get_var_type(return_vars[0])
        else:
            return_var = 'std::tuple<%s> ' % ', '.join([vtable.get_var_type(x) for x in return_vars])
        write_cpp(return_var)
        write_h(return_var)
        
        declaration = '%sForward(const %s& ctx, %s)' % (
              vtable.class_name, vtable.class_name, ', '.join([vtable.get_var_type(x) + ' ' + vtable.to_c_name(x) for x in arguments]))
        write_h(declaration + ';\n')
        write_cpp(declaration + '\n{\n')

        for e in emitter_list:
            string = e.emit()
            free_list = vtable.get_clean_list()
            if len(free_list) > 0:
                if string is not None:
                    string = string.replace('t4::Relu(', 't4::ReluInplace(')
                    string = string.replace('t4::BatchNormalization(', 't4::BatchNormalizationInplace(')
                    write_cpp('\t%s', string)
                write_cpp("\tt4::release(")
                string = ", ".join([vtable.to_c_name(x) for x in free_list])
                write_cpp("%s);\n", string)
            elif string is not None:
                write_cpp('\t%s', string)

        if len(return_vars) == 1:
            write_cpp('\treturn %s;\n', vtable.to_c_name(return_vars[0]))
        else:
            write_cpp('\treturn std::make_tuple(%s);\n' % ', '.join([vtable.to_c_name(x) for x in return_vars]))

        write_cpp('}\n')

        vtable.write_blob(vtable.class_name + '.bin')

    return out
