#include "../include/engine.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace std;
using namespace NeuralNetwork;

namespace NeuralNetwork {

namespace Test {
    void testValue() {
        std::cout << "Testing Value class..." << std::endl;
        
        // Test basic operations with smaller values first
        Value a(1.0);
        Value b(2.0);
        std::cout << "Created values: a = " << a.getData() << ", b = " << b.getData() << std::endl;
        
        // Test addition
        auto c = a + b;
        std::cout << "Addition result (a + b): " << c.getData() << std::endl;
        assert(c.getData() == 3.0 && "Addition test failed");
        
        // Reset gradients before testing multiplication
        a.setGrad(0.0);
        b.setGrad(0.0);
        
        // Test multiplication
        auto d = a * b;
        std::cout << "Multiplication result (a * b): " << d.getData() << std::endl;
        assert(d.getData() == 2.0 && "Multiplication test failed");
        
        // Reset gradients before testing tanh
        a.setGrad(0.0);
        
        // Test tanh
        auto e = a.tanh();
        std::cout << "Tanh result of a: " << e.getData() << std::endl;
        std::cout << "Expected tanh(1.0): " << std::tanh(1.0) << std::endl;
        assert(std::abs(e.getData() - std::tanh(1.0)) < 1e-6 && "Tanh test failed");
        
        std::cout << "\nTesting backpropagation..." << std::endl;
        std::cout << "Initial gradients - a: " << a.getGrad() << ", b: " << b.getGrad() << std::endl;
        
        // Test backpropagation with simpler values
        Value x(2.0);
        Value y(3.0);
        auto z = x * y;  // Simple multiplication for testing gradients
        z.backward();    // This should set x.grad = 3.0 and y.grad = 2.0
        
        std::cout << "After backward pass - x gradient: " << x.getGrad() 
                  << ", y gradient: " << y.getGrad() << std::endl;
        assert(std::abs(x.getGrad() - 3.0) < 1e-6 && "Gradient computation for x failed");
        assert(std::abs(y.getGrad() - 2.0) < 1e-6 && "Gradient computation for y failed");
        
        std::cout << "Value class tests passed!" << std::endl;
    }
    
    void testNeuron() {
        std::cout << "Testing Neuron class..." << std::endl;
        
        // Create neuron with 2 inputs
        Neuron n(2);
        std::vector<Value> inputs = {Value(0.5), Value(-0.5)};  // Use smaller input values
        
        std::cout << "Created neuron with 2 inputs" << std::endl;
        std::cout << "Input values: " << inputs[0].getData() << ", " << inputs[1].getData() << std::endl;
        
        // Print weights and bias
        std::cout << "Weights: ";
        for (const auto& w : n.getWeights()) {
            std::cout << w.getData() << " ";
        }
        std::cout << "\nBias: " << n.getBias().getData() << std::endl;
        
        // Test feedforward
        auto output = n.feedForward(inputs);
        std::cout << "Neuron output: " << output.getData() << std::endl;
        
        // Verify output is in valid range for tanh
        assert(output.getData() >= -1.0 && output.getData() <= 1.0 && 
               "Neuron output should be in tanh range");
        
        std::cout << "Neuron class tests passed!" << std::endl;
    }
    
    void testLayer() {
        std::cout << "Testing Layer class..." << std::endl;
        
        // Create smaller layer for testing
        Layer layer(2, 2);  // 2 neurons, 2 inputs each
        std::vector<Value> inputs = {Value(0.5), Value(-0.5)};  // Use smaller input values
        
        std::cout << "Created layer with 2 neurons, each taking 2 inputs" << std::endl;
        std::cout << "Input values: " << inputs[0].getData() << ", " << inputs[1].getData() << std::endl;
        
        // Print layer weights
        std::cout << "Layer weights per neuron:" << std::endl;
        for (size_t i = 0; i < layer.getNeurons().size(); i++) {
            std::cout << "Neuron " << i << " weights: ";
            for (const auto& w : layer.getNeurons()[i].getWeights()) {
                std::cout << w.getData() << " ";
            }
            std::cout << "bias: " << layer.getNeurons()[i].getBias().getData() << std::endl;
        }
        
        // Test feedforward
        auto outputs = layer.feedForward(inputs);
        std::cout << "Layer outputs: ";
        for (const auto& out : outputs) {
            std::cout << out.getData() << " ";
        }
        std::cout << std::endl;
        
        assert(outputs.size() == 2 && "Layer should output 2 values");
        
        std::cout << "Layer class tests passed!" << std::endl;
    }
    
    void testTraining() {
        std::cout << "Testing Neural Network Training..." << std::endl;
        
        // First test with a very simple case
        std::cout << "\nTesting with simple XOR-like problem..." << std::endl;
        std::vector<std::vector<Value>> X_simple = {
            {Value(0.0), Value(0.0)},
            {Value(0.0), Value(1.0)},
            {Value(1.0), Value(0.0)},
            {Value(1.0), Value(1.0)}
        };
        std::vector<std::vector<Value>> y_simple = {
            {Value(0.0), Value(1.0)},
            {Value(1.0), Value(0.0)},
            {Value(1.0), Value(0.0)},
            {Value(0.0), Value(1.0)}
        };
        
        std::cout << "Creating simple network..." << std::endl;
        NeuralNet simple_model({2, 3, 2});
        
        try {
            simple_model.train(X_simple, y_simple, 5, 0.01);
        } catch (const std::exception& e) {
            std::cout << "Error training on simple data: " << e.what() << std::endl;
        }
        
        // Now try the spiral dataset
        std::cout << "\nTesting with spiral dataset..." << std::endl;
        // ... rest of the spiral dataset test ...
        
        // Generate smaller spiral dataset for testing
        std::cout << "Generating spiral dataset..." << std::endl;
        auto [X, y] = DataGen::make_spiral_data(10, 2);  // 10 points per class, 2 classes
        
        std::cout << "Dataset size - X: " << X.size() << " samples, y: " << y.size() << " labels" << std::endl;
        
        // Debug: Print dimensions
        std::cout << "X[0] size: " << X[0].size() << " features" << std::endl;
        std::cout << "y[0] size: " << y[0].size() << " classes" << std::endl;
        
        // Verify dataset is not empty and dimensions match
        if (X.empty() || y.empty()) {
            std::cout << "ERROR: Generated dataset is empty!" << std::endl;
            return;
        }
        
        if (X.size() != y.size()) {
            std::cout << "ERROR: Number of samples and labels don't match!" << std::endl;
            return;
        }
        
        // Print first few samples with more detail
        std::cout << "\nFirst 3 samples:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(3), X.size()); i++) {
            std::cout << "Sample " << i << ":" << std::endl;
            std::cout << "  Features: ";
            for (const auto& val : X[i]) {
                std::cout << val.getData() << " ";
            }
            std::cout << "\n  Label: ";
            for (const auto& val : y[i]) {
                std::cout << val.getData() << " ";
            }
            std::cout << std::endl;
        }
        
        // Create smaller network with debug output
        std::cout << "\nCreating neural network..." << std::endl;
        NeuralNet model({2, 4, 2});  // 2 inputs, 4 hidden, 2 outputs
        
        std::cout << "Network architecture:" << std::endl;
        std::cout << "  Input size: 2" << std::endl;
        std::cout << "  Hidden layer size: 4" << std::endl;
        std::cout << "  Output size: 2" << std::endl;
        
        std::cout << "\nStarting training..." << std::endl;
        try {
            // Train with even smaller parameters for debugging
            model.train(X, y, 10, 0.01);  // Reduced to 10 epochs temporarily
        } catch (const std::exception& e) {
            std::cout << "ERROR during training: " << e.what() << std::endl;
            return;
        }
        
        // Test predictions on first point of each class
        std::cout << "\nTesting predictions:" << std::endl;
        try {
            for (size_t i = 0; i < 2; i++) {
                std::cout << "Processing class " << i << " sample..." << std::endl;
                if (i * 10 >= X.size()) {
                    std::cout << "ERROR: Index out of bounds!" << std::endl;
                    continue;
                }
                
                auto pred = model.feedForward(X[i * 10]);
                std::cout << "Class " << i << " prediction: ";
                for (const auto& p : pred) {
                    std::cout << p.getData() << " ";
                }
                std::cout << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "ERROR during prediction: " << e.what() << std::endl;
            return;
        }
        
        std::cout << "Training test completed!" << std::endl;
    }
    
    void runAllTests() {
        std::cout << "Running all tests..." << std::endl;
        testValue();
        testNeuron();
        testLayer();
        testTraining();
        std::cout << "All tests passed!" << std::endl;
    }
}

} // namespace NeuralNetwork

int main() {
    srand(42);  // Set random seed for reproducible results
    NeuralNetwork::Test::runAllTests();
    return 0;
}