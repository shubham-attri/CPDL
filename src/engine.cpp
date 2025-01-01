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
        
        Value x(2.0);  // Create input value
        Value y(3.0);  // Create input value
        
        // Test multiplication
        Value z = x * y;  // Should compute z = x * y = 6.0
        
        // Print initial values
        std::cout << "Initial values:" << std::endl;
        std::cout << "x = " << x.getData() << ", grad = " << x.getGrad() << std::endl;
        std::cout << "y = " << y.getData() << ", grad = " << y.getGrad() << std::endl;
        std::cout << "z = " << z.getData() << ", grad = " << z.getGrad() << std::endl;
        
        // Test backward pass
        z.backward();  // This should set x.grad = 3.0 and y.grad = 2.0
        
        // Print final gradients
        std::cout << "After backward pass:" << std::endl;
        std::cout << "x.grad = " << x.getGrad() << " (should be 3.0)" << std::endl;
        std::cout << "y.grad = " << y.getGrad() << " (should be 2.0)" << std::endl;
        
        // Verify gradients
        assert(std::abs(x.getGrad() - 3.0) < 1e-6 && "Gradient computation for x failed");
        assert(std::abs(y.getGrad() - 2.0) < 1e-6 && "Gradient computation for y failed");
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
        
        // First test with a simple case
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
        
        NeuralNet simple_model({2, 3, 2});
        simple_model.train(X_simple, y_simple, 5, 0.01);
        
        // Test with spiral dataset
        auto [X, y] = DataGen::make_spiral_data(10, 2);
        NeuralNet model({2, 4, 2});
        model.train(X, y, 10, 0.01);
    }
    
    void runAllTests() {
        std::cout << "Running all tests..." << std::endl;
        // testValue();
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