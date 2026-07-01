import numpy as np
import os

def main():
    # MNIST input is 1x1x28x28
    # The model expects float32
    x = np.random.rand(1, 1, 28, 28).astype(np.float32)
    np.save("test_input.npy", x)
    print("Generated test_input.npy with shape 1x1x28x28")

if __name__ == "__main__":
    main()
