# Build and execution instructions

You can following the
[instruction](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/MIGraphX-ExecutionProvider.md)
to build a docker container using the MIGraphX EP. Then inside the container, this example can be built and executed as 
follows:
```
mkdir build
cd build
cmake ..
make
./run_onnx ../onnx/char_rnn.onnx
```

