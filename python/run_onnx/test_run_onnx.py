import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import time
import sys
import os
import argparse

type_table = {
    'tensor(int64)' : np.int64,
    'tensor(int32)' : np.int32,
    'tensor(float)' : np.float32
}

def get_numpy_type(tensor_type):
    return_type = np.float32
    if tensor_type in type_table.keys():
        return_type = type_table[tensor_type]
    return return_type


def load_model(model_file, ep_name):
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    #Run the model on the backend
    session = onnxruntime.InferenceSession(model_file, sess_options = so)
    ep_name = ep_name + "ExecutionProvider"
    session.set_providers([ep_name])
    return session


def copy_onnx_file(model_file, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    file_name = os.path.basename(model_file)
    dst_file = dst_dir + '/' + file_name
    cmd = 'cp ' + model_file + ' ' + dst_file
    os.system(cmd)


def wrapup_inputs(session, default_batch_size):
    param_info = session.get_inputs()
    input_num = len(param_info)
    
    input_data = {}
    for in_idx in range(input_num):
        name = param_info[in_idx].name
        print("Input parameter: {}".format(name))
        dims = param_info[in_idx].shape

        input_type = param_info[in_idx].type
        print("shape: type = {}, dim = {}".format(input_type, dims))

        # check dynamic input shape
        is_dynamic_shape = False
        for index in range(len(dims)):
            if isinstance(dims[index], str) or dims[index] == 'None':
                is_dynamic_shape = True
                dims[index] = default_batch_size

        if is_dynamic_shape == True:
            print('Dynamic input shape, change shape to: {}'.format(dims))

        np_type = get_numpy_type(input_type)
        if np_type == np.int32 or np_type == np.int64:
            print("integer type")
            input_data[name] = np.ones(dims).astype(np_type)
        else:
            print("type = {}".format(np_type))
            input_data[name] = np.random.random(dims).astype(np_type)
    return input_data


def write_tensor_to_file(data, out_dir, index, is_input):
    # convert numpy array to onnx tensor
    tensor = numpy_helper.from_array(data)
    data_str = tensor.SerializeToString()
    name_prefix = out_dir + '/'
    if not os.path.isdir(name_prefix):
        os.mkdir(name_prefix)
    if is_input:
        name_prefix = name_prefix + 'input_'
    else:
        name_prefix = name_prefix + 'output_'

    filename = name_prefix + str(index) + '.pb'
    file = open(filename, 'wb')
    file.write(data_str)
    file.close()


def write_inputs_to_files(input_data, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    index = 0
    for key, val in input_data.items():
        write_tensor_to_file(val, out_dir, index, True)
        index = index + 1


def write_outputs_to_files(output_data, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    index = 0
    for val in output_data:
        write_tensor_to_file(val, out_dir, index, False)
        index = index + 1


def run_inference(session, input_data):
    outputs = session.run([], input_data)
    return outputs


def run_once(session, batch_size):
    input_data = wrapup_inputs(session, batch_size)
    param_info = session.get_inputs()
    input_num = len(param_info)
    
    print("Input:")
    for in_idx in range(input_num):
        name = param_info[in_idx].name
        print("\nInput parameter: {}".format(name))
        dims = param_info[in_idx].shape
        input_type = param_info[in_idx].type
        print("shape: type = {}, dim = {}".format(input_type, dims))
        print("input_data:\n{}".format(input_data[name]))

    out_data = run_inference(session, input_data)

    print("\n\n==========================")
    print("Output:")
    for out_index in range(len(out_data)):
        print("output[{}]'s shape = {}".format(out_index, out_data[out_index].shape))
        print("output[{}] = ".format(out_index))
        print(out_data[out_index])


def create_cases(model_file, session, batch_size, out_dir, case_num):
    #copy onnx file
    print("Copy {} to {}".format(model_file, out_dir))
    copy_onnx_file(model_file, out_dir)

    for i in range(case_num):
        print("Create case {}".format(i))
        data_dir = out_dir + '/test_data_set_' + str(i)
        input_data = wrapup_inputs(session, batch_size)
        write_inputs_to_files(input_data, data_dir)

        out_data = run_inference(session, input_data)

        # write output data to files
        write_outputs_to_files(out_data, data_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an onnx model")
    parser.add_argument('--batch_size', type=int, metavar='batch_size', default=1, help='Specify the batch size used in the model')
    parser.add_argument('model', type=str, metavar='model_file', help='onnx file name of the model')
    parser.add_argument('--ep', type=str, metavar='ep_name', default="MIGraphX", help='Name of the execution provider, CPU or MIGraphX')
    parser.add_argument('--create_test', action='store_true', help='Creat a unit test for the run')
    parser.add_argument('--case_dir', type=str, metavar='case_dir', default='ort_test', help='folder where the created test is stored')
    parser.add_argument('--case_num', type=int, metavar='case_num', default=1, help='Number of cases')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    batch_size = args.batch_size
    model_file = args.model
    ep_name = args.ep
    out_dir = args.case_dir
    case_num = args.case_num

    # create a session
    session = load_model(model_file, ep_name)

    # copy model from source to distination
    if args.create_test:
        print("Test case write to folder: {}".format(out_dir))
        create_cases(model_file, session, batch_size, out_dir, case_num)

    run_once(session, batch_size)

if __name__ == "__main__":
    main()

