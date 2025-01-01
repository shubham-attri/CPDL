#include <iostream>
#include <cmath>
#include <cassert>

using namespace std;

namespace NeuralNetwork {

namespace Test {
    void testValue() {
        std::cout << "Testing Value class..." << std::endl;
        
        Value a(2.0);
        Value b(3.0);
        
        auto c = a + b;
        assert(c.getData() == 5.0 && "Addition test failed");
        
        auto d = a * b;
        assert(d.getData() == 6.0 && "Multiplication test failed");
        
        auto e = a.tanh();
        assert(std::abs(e.getData() - std::tanh(2.0)) < 1e-6 && "Tanh test failed");
        
        d.backward();
        assert(a.getGrad() == 3.0 && "Gradient computation for a failed");
        assert(b.getGrad() == 2.0 && "Gradient computation for b failed");
        
        std::cout << "Value class tests passed!" << std::endl;
    }
    
    // void testNeuron() {
    //     std::cout << "Testing Neuron class..." << std::endl;
        
    //     Neuron n(2);
    //     std::vector<Value> inputs = {Value(1.0), Value(2.0)};
        
    //     auto output = n.feedForward(inputs);
    //     assert(output.getData() >= -1.0 && output.getData() <= 1.0 && 
    //            "Neuron output should be in tanh range");
        
    //     std::cout << "Neuron class tests passed!" << std::endl;
    // }
    // `
    // void testLayer() {
    //     std::cout << "Testing Layer class..." << std::endl;
        
    //     Layer layer(3, 2);
    //     std::vector<Value> inputs = {Value(1.0), Value(2.0)};
        
    //     auto outputs = layer.feedForward(inputs);
    //     assert(outputs.size() == 3 && "Layer should output 3 values");
        
    //     std::cout << "Layer class tests passed!" << std::endl;
    // }
    
    // void runAllTests() {
    //     std::cout << "Running all tests..." << std::endl;
    //     testValue();
    //     testNeuron();
    //     testLayer();
    //     std::cout << "All tests passed!" << std::endl;
    // }
}

} // namespace NeuralNetwork

int main() {
    NeuralNetwork::Test::testValue();
    return 0;
}
