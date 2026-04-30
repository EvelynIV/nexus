python -m grpc_tools.protoc \
    -I src \
    --python_out=src \
    --grpc_python_out=src \
    --mypy_out=src \
    --mypy_grpc_out=src \
src/nexus/protos/asr/ux_speech.proto
