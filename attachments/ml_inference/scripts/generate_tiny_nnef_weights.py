import nnef
import numpy as np
import os

def main():
    os.makedirs('models/tiny_nnef', exist_ok=True)
    
    # weights: [8, 16]
    weights = np.random.randn(8, 16).astype(np.float32)
    # bias: [1, 8]
    bias = np.random.randn(1, 8).astype(np.float32)
    
    with open('models/tiny_nnef/weights.dat', 'wb') as f:
        nnef.write_tensor(f, weights)
        
    with open('models/tiny_nnef/bias.dat', 'wb') as f:
        nnef.write_tensor(f, bias)
        
    print("Generated weights.dat and bias.dat in models/tiny_nnef")

if __name__ == "__main__":
    main()
