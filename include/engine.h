#pragma once

#include <vector>
#include <functional>
#include <set>
#include <cmath>
#include <iostream>
#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>

namespace NeuralNetwork {

class Value {
private:
    // double data;  // Forward value
    // double grad=0;  // Gradient value 
    std::vector<std::shared_ptr<Value>> _prev;  // Change to shared_ptr to manage lifetime
    #ifdef DEBUG_MODE
    char op = 'n';  // Operation type for debugging ('n' for none/input)
    #endif
    std::function<void()> _backward;  // Gradient computation function
    
    void build_topo(std::vector<Value*>& topo, std::set<Value*>& visited) noexcept {
        if (visited.find(this) == visited.end()) {
            visited.insert(this);
            for (const auto& child : _prev) {
                child->build_topo(topo, visited);
            }
            topo.push_back(this);
        }
    }

public:
    double data;  // Forward value
    double grad=0;  // Gradient value 
    // Constructor that creates a Value with given data and empty backward function
    // noexcept indicates this constructor won't throw exceptions
    explicit Value(double data) noexcept 
        : data(data), _backward([](){}) {}

    // Copy constructor - needed to allow Values to be copied, which happens when:
    // - Passing Values by value to functions
    // - Storing Values in containers like vectors
    // - Creating new Values from existing ones (e.g. in operator+ return)
    Value(const Value&) = default;
    
    // Move constructor - enables efficient transfer of Values without copying, used when:
    // - Returning Values from functions (like operator+)
    // - Moving Values into containers
    // - std::move() is used explicitly
    Value(Value&&) noexcept = default;
    
    // Copy assignment - needed when assigning one Value to another, like:
    // Value a(1.0); Value b(2.0); a = b;
    // Common in neural network weight updates
    Value& operator=(const Value&) = default;
    
    // Move assignment - enables efficient Value reassignment without copying when:
    // - Assigning temporary Values: a = std::move(b)
    // - Reassigning Values in containers
    Value& operator=(Value&&) noexcept = default;
    
    // Destructor - needed to properly clean up Value resources
    // Default is fine since we use smart pointers/containers
    ~Value() = default;

    // We use += for gradients because a value might be used multiple times in the graph
    // For example, if a=b in a*b or a+b, we need to accumulate gradients from both paths
    Value operator+(const Value& other) const {
        Value out(data + other.data);
        out._prev = {std::make_shared<Value>(*this), std::make_shared<Value>(other)};
        
        #ifdef DEBUG_MODE
        out.op = '+';
        #endif
        
        // For addition, gradient flows directly: ∂out/∂x = 1
        out._backward = [prev = out._prev]() {
            prev[0]->grad += 1.0;  // Each input gets gradient of 1.0 times upstream gradient
            prev[1]->grad += 1.0;
        };
        return out;
    }

    Value operator*(const Value& other) const {
        Value out(data * other.data);
        out._prev = {std::make_shared<Value>(*this), std::make_shared<Value>(other)};
        
        #ifdef DEBUG_MODE
        out.op = '*';
        #endif
        
        // For multiplication: ∂out/∂a = b, ∂out/∂b = a
        double a_data = data;
        double b_data = other.data;
        out._backward = [prev = out._prev, a_data, b_data]() {
            prev[0]->grad += b_data;  // First input gets other's value times upstream gradient
            prev[1]->grad += a_data;  // Second input gets first's value times upstream gradient
        };
        return out;
    }

    Value tanh() const {
        double t = std::tanh(data);
        Value out(t);
        out._prev = {std::make_shared<Value>(*this)};
        
        #ifdef DEBUG_MODE
        out.op = 't';
        #endif
        
        // For tanh: ∂out/∂x = 1 - tanh²(x)
        out._backward = [prev = out._prev, t]() {
            prev[0]->grad += (1 - t*t);  // Derivative of tanh times upstream gradient
        };
        return out;
    }

    // New operators for double
    Value operator+(double other) const {
        return *this + Value(other);  // Convert double to Value and use existing operator
    }

    Value operator*(double other) const {
        return *this * Value(other);
    }

    // Assignment operators
    Value& operator+=(const Value& other) {
        *this = *this + other;
        return *this;
    }

    Value& operator+=(double other) {
        *this = *this + Value(other);
        return *this;
    }

    Value& operator*=(const Value& other) {
        *this = *this * other;
        return *this;
    }

    Value& operator*=(double other) {
        *this = *this * Value(other);
        return *this;
    }

    // Overloads for int
    Value operator+(int other) const {
        return *this + Value(static_cast<double>(other));
    }

    Value operator*(int other) const {
        return *this * Value(static_cast<double>(other));
    }

    Value& operator+=(int other) {
        *this = *this + Value(static_cast<double>(other));
        return *this;
    }

    Value& operator*=(int other) {
        *this = *this * Value(static_cast<double>(other));
        return *this;
    }

    // Friend operators for int
    friend Value operator+(int lhs, const Value& rhs) {
        return Value(static_cast<double>(lhs)) + rhs;
    }

    friend Value operator*(int lhs, const Value& rhs) {
        return Value(static_cast<double>(lhs)) * rhs;
    }

    // Friend operators for commutative operations (allows 2.0 + Value)
    friend Value operator+(double lhs, const Value& rhs) {
        return Value(lhs) + rhs;
    }

    friend Value operator*(double lhs, const Value& rhs) {
        return Value(lhs) * rhs;
    }

    // Basic getters
    double getData() const { return data; }
    double getGrad() const { return grad; }
    #ifdef DEBUG_MODE
    char getOp() const { return op; }
    #endif
    const std::vector<std::shared_ptr<Value>>& getPrev() const { return _prev; }

    void backward() {
        // 1. Build topological order of computation graph
        std::vector<Value*> topo;
        std::set<Value*> visited;
        build_topo(topo, visited);
        
        // 2. Initialize output gradient to 1.0
        grad = 1.0;
        
        // 3. Backpropagate through nodes in reverse topological order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    // Add stream operator as a friend function
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        return os << v.getData();
    }
};

class Neuron {
private:
    std::vector<Value> w;  // Vector of weights, one per input
    Value b;               // Bias term
    
    // Helper to convert numeric inputs to Value objects
    template<typename T>
    static std::vector<Value> toValues(const std::vector<T>& inputs) {
        std::vector<Value> result;
        result.reserve(inputs.size());
        for (const auto& x : inputs) {
            result.emplace_back(static_cast<double>(x));
        }
        return result;
    }
    
public:
    // Constructor takes number of inputs (nin) and initializes:
    // - A bias term randomly between -1 and 1
    // - nin weights randomly between -1 and 1 
    Neuron(size_t nin) : b(Value(static_cast<double>(rand()) / RAND_MAX * 2 - 1)) {
        for (size_t i = 0; i < nin; i++) {
            // For each input, create a random weight between -1 and 1
            w.push_back(Value(static_cast<double>(rand()) / RAND_MAX * 2 - 1));
        }
    }

    // Convenience overload for initializer list inputs
    Value feedForward(std::initializer_list<double> x) {
        return feedForward(std::vector<double>(x));
    }

    // Template overload for different numeric types
    template<typename T>
    Value feedForward(const std::vector<T>& x) {
        return feedForward(toValues(x)); // Convert to Values and forward
    }

    // Main feedforward implementation that:
    // 1. Validates input size matches weights
    // 2. Computes weighted sum of inputs plus bias
    // 3. Applies tanh activation
    Value feedForward(const std::vector<Value>& x) {
        if (x.size() != w.size()) {
            throw std::invalid_argument("Input size must match number of weights");
        }
        
        std::cout << "Computing weighted sum..." << std::endl;
        
        // Create the first product properly
        Value mul = w[0] * x[0];
        Value act = mul;  // Keep the multiplication result alive
        
        // Add remaining products
        for (size_t i = 1; i < w.size(); i++) {
            std::cout << "Adding product " << i << ": " << w[i].getData() << " * " << x[i].getData() << std::endl;
            Value mul_i = w[i] * x[i];  // Store intermediate multiplication
            act = act + mul_i;          // Store intermediate addition
        }
        
        std::cout << "Adding bias: " << b.getData() << std::endl;
        Value with_bias = act + b;  // Store bias addition
        
        std::cout << "Applying tanh activation..." << std::endl;
        return with_bias.tanh();  // Final activation
    }

    // Getters for weights and bias
    const std::vector<Value>& getWeights() const { return w; }
    const Value& getBias() const { return b; }
};

class Layer {
private:
    std::vector<Neuron> neurons;
    size_t n_inputs;  // Number of inputs per neuron

public:
    // Constructor takes number of inputs per neuron and number of neurons
    Layer(size_t nin, size_t nout) : n_inputs(nin) {
        // Create nout neurons, each taking nin inputs
        for (size_t i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
        }
    }

    // Feed forward through all neurons
    std::vector<Value> feedForward(const std::vector<Value>& inputs) {
        if (inputs.size() != n_inputs) {
            throw std::invalid_argument("Input size must match number of inputs per neuron");
        }

        std::vector<Value> outputs;
        outputs.reserve(neurons.size());

        // Feed input through each neuron
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.feedForward(inputs));
        }

        return outputs;
    }

    // Convenience method for numeric inputs
    template<typename T>
    std::vector<Value> feedForward(const std::vector<T>& x) {
        std::vector<Value> inputs;
        inputs.reserve(x.size());
        for (const auto& val : x) {
            inputs.emplace_back(static_cast<double>(val));
        }
        return feedForward(inputs);
    }

    // Getters
    const std::vector<Neuron>& getNeurons() const { return neurons; }
    size_t getInputSize() const { return n_inputs; }
    size_t getOutputSize() const { return neurons.size(); }

    // Get all parameters (weights and biases) for optimization
    std::vector<std::reference_wrapper<Value>> getParameters() {
        std::vector<std::reference_wrapper<Value>> params;
        for (auto& neuron : neurons) {
            // Add weights
            for (auto& w : const_cast<std::vector<Value>&>(neuron.getWeights())) {
                params.push_back(std::ref(w));
            }
            // Add bias
            params.push_back(std::ref(const_cast<Value&>(neuron.getBias())));
        }
        return params;
    }
};

class MLP {
private:
    std::vector<Layer> layers;

public:
    // Constructor takes a vector of sizes (e.g., {2,3,1} = 2 inputs -> 3 hidden -> 1 output)
    MLP(const std::vector<size_t>& layer_sizes) {
        // Create layers based on sizes
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
        }
    }

    // Forward pass through all layers
    std::vector<Value> feedForward(const std::vector<Value>& inputs) {
        std::vector<Value> current = inputs;
        
        // Pass through each layer
        for (auto& layer : layers) {
            current = layer.feedForward(current);
        }
        
        return current;
    }

    // Convenience method for numeric inputs
    template<typename T>
    std::vector<Value> feedForward(const std::vector<T>& x) {
        std::vector<Value> inputs;
        inputs.reserve(x.size());
        for (const auto& val : x) {
            inputs.emplace_back(static_cast<double>(val));
        }
        return feedForward(inputs);
    }

    // Get all parameters for optimization
    std::vector<std::reference_wrapper<Value>> getParameters() {
        std::vector<std::reference_wrapper<Value>> params;
        for (auto& layer : layers) {
            auto layer_params = layer.getParameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    // Zero all gradients before backward pass
    void zeroGrad() {
        for (auto& param : getParameters()) {
            param.get().grad = 0;
        }
    }

    // Simple SGD optimizer step
    void step(double learning_rate) {
        for (auto& param : getParameters()) {
            param.get().data += -learning_rate * param.get().grad;
        }
    }

    // Getters
    const std::vector<Layer>& getLayers() const { return layers; }
    size_t getInputSize() const { return layers.front().getInputSize(); }
    size_t getOutputSize() const { return layers.back().getOutputSize(); }
};

} // namespace NeuralNetwork