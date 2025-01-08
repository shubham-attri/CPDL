#pragma once

#include <vector>
#include <functional>
#include <set>
#include <cmath>
#include <iostream>

namespace NeuralNetwork {

class Value {
private:
    double data;
    double grad = 0;
    std::vector<Value*> _prev;  // Children in computation graph
    #ifdef DEBUG_MODE
    char op = 'n'; // for debugging and visualising the computation graph
    #endif
    std::function<void()> _backward;  // Changed to void() like micrograd

    // Helper function for topological sort
    void build_topo(std::vector<Value*>& topo, std::set<Value*>& visited) noexcept {
        if (visited.find(this) == visited.end()) {
            visited.insert(this);
            for (Value* child : _prev) { // for each child, build the topo
                child->build_topo(topo, visited);
            }
            topo.push_back(this); // add this node to the topo
        }
    }

public:
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
        out._prev = {const_cast<Value*>(this), const_cast<Value*>(&other)};
        
        #ifdef DEBUG_MODE
        out.op = '+';
        #endif
        
        out._backward = [&out, a=const_cast<Value*>(this), 
                        b=const_cast<Value*>(&other)]() {
            a->grad += out.grad;
            b->grad += out.grad;
        };
        return out;
    }

    Value operator*(const Value& other) const {
        Value out(data * other.data);
        out._prev = {const_cast<Value*>(this), const_cast<Value*>(&other)};
        out.op = '*'; // for debugging and visualising the computation graph

        
        out._backward = [&out, a=const_cast<Value*>(this), 
                        b=const_cast<Value*>(&other)]() {
            a->grad += b->data * out.grad;  // ∂(a*b)/∂a = b
            b->grad += a->data * out.grad;  // ∂(a*b)/∂b = a
        };
        return out;
    }

    Value tanh() const {
        double t = std::tanh(data);
        Value out(t);
        out._prev = {const_cast<Value*>(this)};
        out.op = 't';
        
        out._backward = [&out, a=const_cast<Value*>(this), t]() {
            a->grad += (1 - t*t) * out.grad;  // ∂tanh(x)/∂x = 1 - tanh²(x)
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
    const std::vector<Value*>& getPrev() const { return _prev; }

    void backward() {
        // Build topological order
        std::vector<Value*> topo;
        std::set<Value*> visited;
        build_topo(topo, visited);
        
        // Initialize output gradient
        grad = 1.0;
        
        // Backpropagate in reverse order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();  // Call backward function for each node
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
        
        // Compute weighted sum of inputs
        Value act = w[0] * x[0];
        
        // Add remaining weight * input products
        for (size_t i = 1; i < w.size(); i++) {
            act = act + w[i] * x[i];
        }
        
        // Add bias term and apply tanh activation
        act = act + b;
        return act.tanh();
    }

    // Getters for weights and bias
    const std::vector<Value>& getWeights() const { return w; }
    const Value& getBias() const { return b; }
};


} // namespace NeuralNetwork