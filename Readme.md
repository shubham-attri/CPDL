# Neural Network Implementation in C++

This project implements a neural network from scratch in C++, inspired by Andrej Karpathy's micrograd. The implementation focuses on building an autograd engine and neural network components from first principles.

## Features

- **Autograd Engine**: Core implementation of automatic differentiation
  - Value class with support for basic operations (+, *, tanh)
  - Automatic gradient computation through backward propagation
  - Computation graph visualization capabilities

- **Neural Network Components**:
  - Neuron class implementing a single artificial neuron
  - Layer class for grouping neurons
  - Neural Network class for deep learning architectures

## Implementation Details

### Value Class
The `Value` class is the fundamental building block, implementing:
- Forward pass computation
- Backward propagation of gradients
- Topological sorting of computation graphs
- Debug visualization support

### Visualization
The project includes visualization tools to help understand:
- Forward pass computation steps
- Gradient flow during backpropagation
- Computation graph structure
### Value Class Example
Running the program with the `visualizeValue()` function demonstrates a basic Value class implementation with debug visualization enabled:


## Example Usage

### Running the Program
1. Clone the repository
2. Build using CMake:
   ```bash
   mkdir build
   cd build
   cmake -DDEBUG=ON ..   # Enable debug visualization
   # or cmake -DDEBUG=OFF .. for release build
   make
   ```
3. Run the executable:
   ```bash
   ./neural_network
   ```
   When built with DEBUG=ON, the program will generate visualization files for the computation graphs.


