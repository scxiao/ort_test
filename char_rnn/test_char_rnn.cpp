// cxx include
#include <iostream>
#include <map>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <climits>

// onnx runtime include
#include <core/session/onnxruntime_cxx_api.h>
// #include <core/framework/allocator.h>
#include <core/providers/migraphx/migraphx_provider_factory.h>

std::string get_type_name(ONNXTensorElementDataType type)
{
    static std::map<ONNXTensorElementDataType, std::string> table = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "bool"}, 
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "half"}, 
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "float"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "double"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "int8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "int16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "int32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "int64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "uint8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "uint16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "uint32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "uint64"}
    };

    if (table.count(type) > 0) {
        return table.at(type);
    }

    return "unknown";
}

template<class T>
void print_vec(std::vector<T>& vec)
{
    std::cout << "{";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        if (it != vec.begin())
            std::cout << ", ";
        std::cout << *it;
    }

    std::cout << "}" << std::endl;
}

template<class T>
void set_default_dims(std::vector<T>& vec)
{
    for (auto& v : vec) {
        if (v < 0) v = 1;
    }
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " onnxfile" << std::endl;
        return 0;
    }


    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions sess_options;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sess_options, 0));

    Ort::Session sess{env, argv[1], sess_options};
    const char * input_names[20]; 

    std::vector<std::vector<float>> input_data;
    std::vector<std::vector<int64_t>> in_dims;
    input_data.resize(sess.GetInputCount());
    in_dims.resize(sess.GetInputCount());
    std::vector<Ort::Value> inputs;
    std::cout << "Input names: " << std::endl;
    for (size_t i = 0; i < sess.GetInputCount(); ++i)
    {
        input_names[i] = sess.GetInputName(i, ort_alloc);
        std::cout << "input " << i << "\'s name: " << input_names[i];
        Ort::TypeInfo info = sess.GetInputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        in_dims[i] = tensor_info.GetShape();
        set_default_dims(in_dims[i]);
        auto onnx_type = tensor_info.GetElementType();
        std::cout << ", type: " << get_type_name(onnx_type) << ", shape = ";
        print_vec(in_dims[i]);

        auto elem_num = std::accumulate(in_dims[i].begin(), in_dims[i].end(), 1, std::multiplies<int64_t>());
        input_data[i].resize(elem_num);
        std::generate(input_data[i].begin(), input_data[i].end(), []() { return rand() / (float(INT_MAX)); });
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, input_data[i].data(), 
                    input_data[i].size(), in_dims[i].data(), in_dims[i].size()));
    }
    std::cout << std::endl;

    std::vector<Ort::Value> outputs;
    const char * output_names[20];
    std::vector<std::vector<float>> output_data;
    std::vector<std::vector<int64_t>> out_dims;
    auto output_num = sess.GetOutputCount();
    output_data.resize(output_num);
    out_dims.resize(output_num);
    std::cout << "Output names:" << std::endl;
    for (size_t i = 0; i < sess.GetOutputCount(); ++i)
    {
        output_names[i] = sess.GetOutputName(i, ort_alloc);
        std::cout << "Out " << i << "'s name: " << input_names[i];
        Ort::TypeInfo info = sess.GetOutputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        out_dims[i] = tensor_info.GetShape();
        set_default_dims(out_dims[i]);
        auto onnx_type = tensor_info.GetElementType();
        std::cout << ", type: " << get_type_name(onnx_type) << ", shape = ";
        print_vec(out_dims[i]);

        std::size_t elem_num = std::accumulate(out_dims[i].begin(), out_dims[i].end(), 1, std::multiplies<int64_t>());
        output_data[i].resize(elem_num);
        outputs.push_back(Ort::Value::CreateTensor<float>(memory_info, output_data[i].data(), output_data[i].size(),
                    out_dims[i].data(), out_dims[i].size()));
    }
    std::cout << std::endl;

    std::cout << "Start inference ...." << std::endl;
    sess.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names,
            outputs.data(), outputs.size());
    std::cout << "Finished inference ...." << std::endl;

    std::cout << "outputs: " << std::endl;
    std::size_t i = 0;
    for (auto& output : output_data)
    {
        std::cout << "Output_" << i++ << " = " << std::endl;
        print_vec(output);
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}

