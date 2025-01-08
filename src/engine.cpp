#include "../include/engine.h"
#include <cmath>
#include <cassert>


using namespace std;
using namespace NeuralNetwork;

namespace NeuralNetwork {

// namespace Test {
//     void testValue() {
//         std::cout << "Testing Value class..." << std::endl;
        
//         Value x(2.0);  // Create input value
//         Value y(3.0);  // Create input value
        
//         // Test multiplication
//         Value z = x * y;  // Should compute z = x * y = 6.0
        
//         // Print initial values
//         std::cout << "Initial values:" << std::endl;
//         std::cout << "x = " << x.getData() << ", grad = " << x.getGrad() << std::endl;
//         std::cout << "y = " << y.getData() << ", grad = " << y.getGrad() << std::endl;
//         std::cout << "z = " << z.getData() << ", grad = " << z.getGrad() << std::endl;
        
//         // Test backward pass
//         z.backward();  // This should set x.grad = 3.0 and y.grad = 2.0
        
//         // Print final gradients
//         std::cout << "After backward pass:" << std::endl;
//         std::cout << "x.grad = " << x.getGrad() << " (should be 3.0)" << std::endl;
//         std::cout << "y.grad = " << y.getGrad() << " (should be 2.0)" << std::endl;
        
//         // Verify gradients
//         assert(std::abs(x.getGrad() - 3.0) < 1e-6 && "Gradient computation for x failed");
//         assert(std::abs(y.getGrad() - 2.0) < 1e-6 && "Gradient computation for y failed");
//     }
    
//     void testNeuron() {
//         std::cout << "Testing Neuron class..." << std::endl;
        
//         // Create neuron with 2 inputs
//         Neuron n(2);
//         std::vector<Value> inputs = {Value(0.5), Value(-0.5)};  // Use smaller input values
        
//         std::cout << "Created neuron with 2 inputs" << std::endl;
//         std::cout << "Input values: " << inputs[0].getData() << ", " << inputs[1].getData() << std::endl;
        
//         // Print weights and bias
//         std::cout << "Weights: ";
//         for (const auto& w : n.getWeights()) {
//             std::cout << w.getData() << " ";
//         }
//         std::cout << "\nBias: " << n.getBias().getData() << std::endl;
        
//         // Test feedforward
//         auto output = n.feedForward(inputs);
//         std::cout << "Neuron output: " << output.getData() << std::endl;
        
//         // Verify output is in valid range for tanh
//         assert(output.getData() >= -1.0 && output.getData() <= 1.0 && 
//                "Neuron output should be in tanh range");
        
//         std::cout << "Neuron class tests passed!" << std::endl;
//     }
    
//     void testLayer() {
//         std::cout << "Testing Layer class..." << std::endl;
        
//         // Create smaller layer for testing
//         Layer layer(2, 2);  // 2 neurons, 2 inputs each
//         std::vector<Value> inputs = {Value(0.5), Value(-0.5)};  // Use smaller input values
        
//         std::cout << "Created layer with 2 neurons, each taking 2 inputs" << std::endl;
//         std::cout << "Input values: " << inputs[0].getData() << ", " << inputs[1].getData() << std::endl;
        
//         // Print layer weights
//         std::cout << "Layer weights per neuron:" << std::endl;
//         for (size_t i = 0; i < layer.getNeurons().size(); i++) {
//             std::cout << "Neuron " << i << " weights: ";
//             for (const auto& w : layer.getNeurons()[i].getWeights()) {
//                 std::cout << w.getData() << " ";
//             }
//             std::cout << "bias: " << layer.getNeurons()[i].getBias().getData() << std::endl;
//         }
        
//         // Test feedforward
//         auto outputs = layer.feedForward(inputs);
//         std::cout << "Layer outputs: ";
//         for (const auto& out : outputs) {
//             std::cout << out.getData() << " ";
//         }
//         std::cout << std::endl;
        
//         assert(outputs.size() == 2 && "Layer should output 2 values");
        
//         std::cout << "Layer class tests passed!" << std::endl;
//     }
    
//     void testTraining() {
//         std::cout << "Testing Neural Network Training..." << std::endl;
        
//         // First test with a simple case
//         std::vector<std::vector<Value>> X_simple = {
//             {Value(0.0), Value(0.0)},
//             {Value(0.0), Value(1.0)},
//             {Value(1.0), Value(0.0)},
//             {Value(1.0), Value(1.0)}
//         };
//         std::vector<std::vector<Value>> y_simple = {
//             {Value(0.0), Value(1.0)},
//             {Value(1.0), Value(0.0)},
//             {Value(1.0), Value(0.0)},
//             {Value(0.0), Value(1.0)}
//         };
        
//         NeuralNet simple_model({2, 3, 2});
//         simple_model.train(X_simple, y_simple, 5, 0.01);
        
//         // Test with spiral dataset
//         auto [X, y] = DataGen::make_spiral_data(10, 2);
//         NeuralNet model({2, 4, 2});
//         model.train(X, y, 10, 0.01);
//     }
    
//     void runAllTests() {
//         std::cout << "Running all tests..." << std::endl;
//         // testValue();
//         testNeuron();
//         testLayer();
//         testTraining();
//         std::cout << "All tests passed!" << std::endl;
//     }
// }

} // namespace NeuralNetwork

void visualizeValue() {
    // Create inputs and weights
    Value x1(2.0);
    Value w1(-3.0);
    Value x2(0.0);
    Value w2(1.0);
    Value b(6.881373787);

    std::cout << "\n=== Forward Pass ===" << std::endl;
    
    // First multiplication: x1 * w1
    Value mul1 = x1 * w1;
    std::cout << "1. mul1 = x1 * w1: " << x1.getData() << " * " << w1.getData() 
              << " = " << mul1.getData() 
              #ifdef DEBUG_MODE
              << " (op=" << mul1.getOp() << ")"
              #endif
              << std::endl;
    std::cout << "   mul1 children: " << mul1.getPrev()[0]->getData() 
              << ", " << mul1.getPrev()[1]->getData() << std::endl;
    
    // Second multiplication: x2 * w2  
    Value mul2 = x2 * w2;
    std::cout << "2. mul2 = x2 * w2: " << x2.getData() << " * " << w2.getData() 
              << " = " << mul2.getData() << " (op=" << mul2.getOp() << ")" << std::endl;
    std::cout << "   mul2 children: " << mul2.getPrev()[0]->getData() 
              << ", " << mul2.getPrev()[1]->getData() << std::endl;

    // Sum the products
    Value sum_products = mul1 + mul2;
    std::cout << "3. sum_products = mul1 + mul2: " << mul1.getData() << " + " << mul2.getData() 
              << " = " << sum_products.getData() << " (op=" << sum_products.getOp() << ")" << std::endl;
    std::cout << "   sum_products children: " << sum_products.getPrev()[0]->getData() 
              << ", " << sum_products.getPrev()[1]->getData() << std::endl;
    
    // Add bias
    Value pre_activation = sum_products + b;
    std::cout << "4. pre_activation = sum_products + b: " << sum_products.getData() << " + " << b.getData() 
              << " = " << pre_activation.getData() << " (op=" << pre_activation.getOp() << ")" << std::endl;
    std::cout << "   pre_activation children: " << pre_activation.getPrev()[0]->getData() 
              << ", " << pre_activation.getPrev()[1]->getData() << std::endl;
    
    // Apply tanh activation
    Value out = pre_activation.tanh();
    std::cout << "5. out = tanh(pre_activation): tanh(" << pre_activation.getData() << ") = " << out.getData() 
              << " (op=" << out.getOp() << ")" << std::endl;
    std::cout << "   out child: " << out.getPrev()[0]->getData() << std::endl;

    std::cout << "\n=== Starting Backward Pass ===" << std::endl;
    std::cout << "Initial gradients:" << std::endl;
    std::cout << "out: grad=" << out.getGrad() << std::endl;
    std::cout << "pre_activation: grad=" << pre_activation.getGrad() << std::endl;
    std::cout << "sum_products: grad=" << sum_products.getGrad() << std::endl;
    std::cout << "mul1: grad=" << mul1.getGrad() << std::endl;
    std::cout << "mul2: grad=" << mul2.getGrad() << std::endl;
    
    out.backward();
    
    std::cout << "\n=== Gradient Flow (Reverse Order) ===" << std::endl;
    std::cout << "Step 1 - Tanh:" << std::endl;
    std::cout << "  out (tanh): grad = " << out.getGrad() << std::endl;
    std::cout << "  pre_activation: grad = " << pre_activation.getGrad() << std::endl;
    
    std::cout << "\nStep 2 - Pre-activation Addition:" << std::endl;
    std::cout << "  sum_products: grad = " << sum_products.getGrad() << std::endl;
    std::cout << "  b: grad = " << b.getGrad() << std::endl;
    
    std::cout << "\nStep 3 - Sum Products Addition:" << std::endl;
    std::cout << "  mul1: grad = " << mul1.getGrad() << std::endl;
    std::cout << "  mul2: grad = " << mul2.getGrad() << std::endl;
    
    std::cout << "\nStep 4 - First Multiplication:" << std::endl;
    std::cout << "  x1: grad = " << x1.getGrad() << std::endl;
    std::cout << "  w1: grad = " << w1.getGrad() << std::endl;
    
    std::cout << "\nStep 5 - Second Multiplication:" << std::endl;
    std::cout << "  x2: grad = " << x2.getGrad() << std::endl;
    std::cout << "  w2: grad = " << w2.getGrad() << std::endl;
}

void visualizeNeuron() {
    std::cout << "\n=== Neuron Visualization ===" << std::endl;
    
    // Create a neuron with 2 inputs
    Neuron neuron(2);
    
    // Get the weights and bias as Value objects
    const auto& weights = neuron.getWeights();
    const auto& bias = neuron.getBias();
    
    // Create input Values
    Value x1(1.0);
    Value x2(0.5);
    std::vector<Value> inputs = {x1, x2};
    
    // Print initial configuration
    std::cout << "Initial Values:" << std::endl;
    std::cout << "x1 = " << x1.getData() << " (op=" << x1.getOp() << ")" << std::endl;
    std::cout << "x2 = " << x2.getData() << " (op=" << x2.getOp() << ")" << std::endl;
    std::cout << "w1 = " << weights[0].getData() << " (op=" << weights[0].getOp() << ")" << std::endl; 
    std::cout << "w2 = " << weights[1].getData() << " (op=" << weights[1].getOp() << ")" << std::endl;
    std::cout << "b = " << bias.getData() << " (op=" << bias.getOp() << ")" << std::endl;

    // Forward pass with visualization
    Value output = neuron.feedForward(inputs);
    
    // Print computation graph details
    std::cout << "\nComputation Graph:" << std::endl;
    std::cout << "1. First multiplication: " << x1.getData() << " * " << weights[0].getData() 
              << " = " << (x1 * weights[0]).getData() << " (op=" << (x1 * weights[0]).getOp() << ")" << std::endl;
              
    std::cout << "2. Second multiplication: " << x2.getData() << " * " << weights[1].getData()
              << " = " << (x2 * weights[1]).getData() << " (op=" << (x2 * weights[1]).getOp() << ")" << std::endl;
              
    std::cout << "3. Sum and bias addition: " << output.getData() << " (op=" << output.getOp() << ")" << std::endl;
    
    // Backward pass
    std::cout << "\nBackward Pass:" << std::endl;
    // output.backward();
    
    std::cout << "Final Gradients:" << std::endl;
    std::cout << "∂E/∂x1 = " << x1.getGrad() << std::endl;
    std::cout << "∂E/∂x2 = " << x2.getGrad() << std::endl;
    std::cout << "∂E/∂w1 = " << weights[0].getGrad() << std::endl;
    std::cout << "∂E/∂w2 = " << weights[1].getGrad() << std::endl;
    std::cout << "∂E/∂b = " << bias.getGrad() << std::endl;
}



int main() {
    using namespace NeuralNetwork;
    
    // visualizeValue();    
    visualizeNeuron();
    return 0;
}