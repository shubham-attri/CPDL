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
    std::function<void(Value&)> _backward;  // Changed signature to take Value& parameter
    
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
        : data(data), _backward([](Value&){}) {}

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
        
        out._backward = [prev=out._prev](Value& out) {
            prev[0]->grad += out.grad;  // Use the actual stored values
            prev[1]->grad += out.grad;
        };
        return out;
    }

    Value operator*(const Value& other) const {
        Value out(data * other.data);
        out._prev = {std::make_shared<Value>(*this), std::make_shared<Value>(other)};
        
        #ifdef DEBUG_MODE
        out.op = '*';
        #endif
        
        out._backward = [prev=out._prev](Value& out) {
            prev[0]->grad += prev[1]->data * out.grad;  // Use values from _prev
            prev[1]->grad += prev[0]->data * out.grad;
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
        
        out._backward = [prev=out._prev, t](Value& out) {
            prev[0]->grad += (1.0 - t*t) * out.grad;  // Use the stored value
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
        
        // 2. Initialize output gradient to 1.0 if not already set
        if (grad == 0.0) {
            grad = 1.0;
        }
        
        // 3. Backpropagate through nodes in reverse topological order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward(**it);  // Pass the Value object to the lambda
        }
    }

    // Add stream operator as a friend function
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        return os << v.getData();
    }
};

class Neuron {
private:
    std::vector<std::shared_ptr<Value>> w;  // Store shared_ptrs
    std::shared_ptr<Value> b;               // Store shared_ptr
    
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
    Neuron(size_t nin) : b(std::make_shared<Value>(static_cast<double>(rand()) / RAND_MAX * 2 - 1)) {
        for (size_t i = 0; i < nin; i++) {
            w.push_back(std::make_shared<Value>(static_cast<double>(rand()) / RAND_MAX * 2 - 1));
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
        
        // Create the first product properly - dereference the shared_ptr
        Value mul = *w[0] * x[0];  // Changed: w[0] -> *w[0]
        Value act = mul;
        
        // Add remaining products
        for (size_t i = 1; i < w.size(); i++) {
            std::cout << "Adding product " << i << ": " << w[i]->getData() << " * " << x[i].getData() << std::endl;
            Value mul_i = *w[i] * x[i];  // Changed: w[i] -> *w[i]
            act = act + mul_i;
        }
        
        std::cout << "Adding bias: " << b->getData() << std::endl;
        Value with_bias = act + *b;  // Already fixed
        
        std::cout << "Applying tanh activation..." << std::endl;
        return with_bias.tanh();
    }

    // Getters for weights and bias
    const std::vector<std::shared_ptr<Value>>& getWeights() const { return w; }
    const std::shared_ptr<Value>& getBias() const { return b; }
};

class Layer {
private:
    std::vector<Neuron> neurons;
    size_t n_inputs;
    std::vector<std::reference_wrapper<Value>> parameters;  // Store persistent references

public:
    Layer(size_t nin, size_t nout) : n_inputs(nin) {
        neurons.reserve(nout);
        for (size_t i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
            // Store parameters when creating neurons
            for (auto& w : const_cast<std::vector<std::shared_ptr<Value>>&>(neurons.back().getWeights())) {
                parameters.push_back(std::ref(*w));
            }
            parameters.push_back(std::ref(*const_cast<std::shared_ptr<Value>&>(neurons.back().getBias())));
        }
    }

    // Return the stored parameters
    const std::vector<std::reference_wrapper<Value>>& getParameters() const {
        return parameters;
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
};

class MLP {
private:
    std::vector<Layer> layers;
    std::vector<std::reference_wrapper<Value>> parameters;  // Store all parameters

public:
    // Constructor takes a vector of sizes (e.g., {2,3,1} = 2 inputs -> 3 hidden -> 1 output)
    MLP(const std::vector<size_t>& layer_sizes) {
        // Create layers based on sizes
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
            // Store parameters from each layer
            const auto& layer_params = layers.back().getParameters();
            parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
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

    // Return stored parameters instead of creating new ones
    const std::vector<std::reference_wrapper<Value>>& getParameters() const {
        return parameters;
    }

    // Update zeroGrad to use stored parameters
    void zeroGrad() {
        for (auto& param : parameters) {
            param.get().grad = 0;
        }
    }

    // Update step to use stored parameters
    void step(double learning_rate) {
        for (auto& param : parameters) {
            param.get().data += -learning_rate * param.get().grad;
        }
    }

    // Getters
    const std::vector<Layer>& getLayers() const { return layers; }
    size_t getInputSize() const { return layers.front().getInputSize(); }
    size_t getOutputSize() const { return layers.back().getOutputSize(); }
};

} // namespace NeuralNetwork