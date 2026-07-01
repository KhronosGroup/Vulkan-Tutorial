import onnx
import sys
import numpy as np

# Patch missing onnx.mapping for newer ONNX versions
if not hasattr(onnx, 'mapping'):
    class Mapping:
        pass
    onnx.mapping = Mapping()
    # Basic mapping needed by onnx-graphsurgeon and onnx2tf
    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE = {
        onnx.TensorProto.FLOAT: np.dtype('float32'),
        onnx.TensorProto.UINT8: np.dtype('uint8'),
        onnx.TensorProto.INT8: np.dtype('int8'),
        onnx.TensorProto.UINT16: np.dtype('uint16'),
        onnx.TensorProto.INT16: np.dtype('int16'),
        onnx.TensorProto.INT32: np.dtype('int32'),
        onnx.TensorProto.INT64: np.dtype('int64'),
        onnx.TensorProto.STRING: np.dtype('O'),
        onnx.TensorProto.BOOL: np.dtype('bool'),
        onnx.TensorProto.FLOAT16: np.dtype('float16'),
        onnx.TensorProto.DOUBLE: np.dtype('float64'),
        onnx.TensorProto.UINT32: np.dtype('uint32'),
        onnx.TensorProto.UINT64: np.dtype('uint64'),
        onnx.TensorProto.BFLOAT16: np.dtype('float32'), # Close enough for mapping
    }
    onnx.mapping.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.items()}

# Now import onnx2tf and run it
from onnx2tf import main
if __name__ == "__main__":
    main()
