import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import time
import sys

#type_table = {
#    TensorProto.INT64 : np.int64,
#    TensorProto.INT32 : np.int32,
#    TensorProto.FLOAT : np.float32
#}

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

    #Get input_name
    inputs = session.get_inputs()
    num_inputs = len(inputs)
    print("Model {} has {} inputs".format(model_file, num_inputs))

    return session

def run_inference(session, dim_size):
    #Get input_name
    inputs = session.get_inputs()
    num_inputs = len(inputs)

    #Wrap up inputs
    input_dict = {}
    for input_index in range(num_inputs):
        name = inputs[input_index].name
        print("name = {}".format(name))
        shape = inputs[input_index].shape
        print("shape = {}".format(shape))
        print("batch_size = {}".format(shape[0]))

        # check dynamic shape
        for index in range(len(shape)):
            if isinstance(shape[index], str):
                shape[index] = dim_size
        print("shape = {}".format(shape))

        input_type = inputs[input_index].type
        print(input_type)

        np_type = get_numpy_type(input_type)

        # handle dynamic shape
        is_dynamic_shape = False
        for i in range(len(shape)):
            if shape[i] == 'None':
                is_dynamic_shape = True
                shape[i] = dim_size

        if is_dynamic_shape == True:
            print('Dynamic input shape, change shape to: {}'.format(shape))

        if np_type == np.int32 or np_type == np.int64:
            print("integer type")
            input_dict[name] = np.ones(shape).astype(np_type)
        else:
            print("type = {}".format(np_type))
            input_dict[name] = np.random.random(shape).astype(np_type)

    for keys, values in input_dict.items():
        print(keys)
        print(values)

    outputs = session.run([], input_dict)
    num_outputs = len(outputs)
    for out_index in range(num_outputs):
        print("output[{}]'s shape = {}".format(out_index, outputs[out_index].shape))
        print("output[{}] = ".format(out_index))
        print(outputs[out_index])

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_onnx.py file.onnx ep_name:[MIGraphX, CPU]")
        exit()

    model_file = sys.argv[1]
    ep_name = sys.argv[2]

    session = load_model(model_file, ep_name)
    run_inference(session, 4)

if __name__ == "__main__":
    main()

