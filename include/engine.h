#pragma once

#include <vector>
#include <functional>
#include <set>
#include <memory>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>

namespace NeuralNetwork {

// Forward declarations
class Value;
class Neuron;
class Layer;
class NeuralNet;

// Value class for automatic differentiation
class Value {
private:
    double data; // value of the node
    mutable double grad = 0; // gradient of the node and initialised to 0 
    std::vector<Value*> prev; // previous nodes
    std::function<void()> backward_fn; // backward function

public:
    // Constructor: Creates a Value node with given data and optional children nodes
    // @param data: The numerical value to store
    // @param children: Vector of pointers to child nodes in computation graph (default empty)
    explicit Value(double data, std::vector<Value*> children = {}) 
        : data(data), grad(0.0), prev(children) {}

    // Addition operator: Creates a new Value representing (this + other)
    // Sets up gradient computation for backpropagation
    // @param other: The Value to add to this
    // @return: New Value containing the sum and gradient function
    Value operator+(const Value& other) const {
        // Forward pass: just add the values
        Value out(data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)});
        
        // Backward pass: define how gradients flow
        out.backward_fn = [this, &other, out_ptr = &out]() {
            grad += out_ptr->grad;
            other.grad += out_ptr->grad;
        };
        return out;
    }

    // Multiplication operator: Creates a new Value representing (this * other)
    // Sets up gradient computation for backpropagation
    // @param other: The Value to multiply with this
    // @return: New Value containing the product and gradient function
    Value operator*(const Value& other) const {
        // Forward pass: just multiply the values
        Value out(data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)});

        // Backward pass: define how gradients flow
        double this_data = data;
        double other_data = other.data;
        out.backward_fn = [this, &other, this_data, other_data, out_ptr = &out]() {
            grad += other_data * out_ptr->grad;
            other.grad += this_data * out_ptr->grad;
        };
        return out;
    }

    void backward() {
        // These will store our computation graph in topologically sorted order
        std::vector<Value*> topo;
        std::set<Value*> visited;
        
        // Define a recursive function to build the topological sort
        std::function<void(Value*)> build_topo = [&](Value* v) {
            if (!v || visited.find(v) != visited.end()) {
                return;  // Skip null pointers and already visited nodes
            }
            visited.insert(v);                           // Mark as visited
            for (Value* child : v->prev) {              // Visit all children first
                if (child) {                            // Check for null pointers
                    build_topo(child);
                }
            }
            topo.push_back(v);                          // Add node after its children
        };
        
        build_topo(this);           // Start building from current node
        grad = 1.0;                 // Set gradient of output node to 1.0
        
        // Traverse nodes in reverse order to compute gradients
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (*it && (*it)->backward_fn) {            // Check for null pointers
                (*it)->backward_fn();    // Apply chain rule via stored backward function
            }
        }
    }

    // Returns the data value stored in this Value object
    double getData() const { return data; }
    // Returns the gradient value stored in this Value object
    double getGrad() const { return grad; }
    // Sets the gradient value to the specified value
    void setGrad(double g) { grad = g; }

    // Applies the hyperbolic tangent (tanh) activation function to this Value
    // @return: A new Value containing tanh(data) and gradient computation
    Value tanh() const {
        // Compute tanh of the input data
        double t = std::tanh(data);
        
        // Create new Value node with the tanh result
        Value out(t, {const_cast<Value*>(this)});
        
        // Set up gradient computation for backpropagation
        out.backward_fn = [t, this, out_ptr = &out]() {
            // Update gradient using chain rule:
            // ∂out/∂x = (1 - tanh²(x)) * ∂L/∂out 
            grad += (1 - t * t) * out_ptr->grad;
        };
        return out;
    }
};

// Loss functions namespace
namespace Loss {
    // Mean Squared Error (MSE) loss
    // Mean Squared Error (MSE) loss function
    // @param predictions: Vector of predicted values from the network
    // @param targets: Vector of target/ground truth values to compare against
    // @return: A Value object containing the MSE loss value
    inline Value mse(const std::vector<Value>& predictions, const std::vector<Value>& targets) {
        // Verify predictions and targets have same size
        assert(predictions.size() == targets.size());
        // Log start of MSE computation
        std::cout << "Computing MSE loss for " << predictions.size() << " values" << std::endl;
        
        Value loss(0.0);
        try {
            for (size_t i = 0; i < predictions.size(); i++) {
                // Use the prediction directly (it's already in the computation graph)
                const Value& pred = predictions[i];
                // Create a constant Value for the target (no gradient needed)
                Value diff = pred + Value(-targets[i].getData());  // Convert target to scalar
                Value squared = diff * diff;
                loss = loss + squared;
                
                std::cout << "  pred: " << pred.getData() 
                         << ", target: " << targets[i].getData() 
                         << ", diff: " << diff.getData() 
                         << ", squared: " << squared.getData() << std::endl;
            }
            loss = loss * Value(1.0 / predictions.size());
            std::cout << "Final MSE loss: " << loss.getData() << std::endl;
            return loss;
        } catch (const std::exception& e) {
            std::cout << "Error computing MSE loss: " << e.what() << std::endl;
            throw;
        }
    }
}

// Single neuron class that represents one unit in a neural network
// It contains weights for each input, a bias term, and stores its output
class Neuron {
private:
    std::vector<Value> weights;  // Vector of weights, one for each input connection
    Value bias;                  // Bias term that gets added to weighted sum
    Value output;               // Stores the neuron's output after activation

public:
    // Constructor for a Neuron that takes the number of input connections as a parameter
    explicit Neuron(size_t num_inputs) : bias(Value(0.0)), output(Value(0.0)) {
        // Initialize weights with random values between -1 and 1
        for (size_t i = 0; i < num_inputs; i++) {
            // Random initialization using uniform distribution
            double rand_weight = ((double)rand() / RAND_MAX) * 2 - 1;  // Range [-1, 1]
            weights.push_back(Value(rand_weight));
        }
        // Initialize bias with random value between -1 and 1
        double rand_bias = ((double)rand() / RAND_MAX) * 2 - 1;
        bias = Value(rand_bias);
    }

    // Performs the feedforward computation for this neuron
    // @param inputs: Vector of input values to process
    // @return: The output Value after applying weights, bias, and activation function
    Value feedForward(const std::vector<Value>& inputs) {
        // Validate input size matches number of weights
        if (inputs.size() != weights.size()) {
            throw std::invalid_argument("Number of inputs must match number of weights");
        }

        // Use std::transform and std::accumulate for cleaner weighted sum computation
        Value act = std::accumulate(
            inputs.begin(), inputs.end(),  // Input range
            Value(0.0),                    // Initial value
            [this, i = 0](Value& sum, const Value& input) mutable {
                return sum + (weights[i++] * input);
            }
        );
        
        // Add bias and apply activation function
        act = (act + bias).tanh();
        
        // Store and return result
        output = act;
        return output;
    }

    void zero_grad() {
        for (auto& w : weights) {
            w.setGrad(0.0);
        }
        bias.setGrad(0.0);
    }

    void update_parameters(double learning_rate) {
        for (auto& w : weights) {
            w = Value(w.getData() - learning_rate * w.getGrad());
        }
        bias = Value(bias.getData() - learning_rate * bias.getGrad());
    }

    // Getter methods for accessing neuron's internal state
    std::vector<Value>& getWeights() { return weights; }
    Value& getBias() { return bias; }
    Value& getOutput() { return output; }
};

// Layer class
// Layer class represents a collection of neurons that form one layer of the neural network
class Layer {
private:
    std::vector<Neuron> neurons;  // Collection of neurons in this layer

public:
    // Constructor creates a layer with specified number of neurons, each with given number of inputs
    // @param num_neurons: Number of neurons in this layer
    // @param num_inputs_per_neuron: Number of inputs each neuron accepts
    Layer(size_t num_neurons, size_t num_inputs_per_neuron) {
        for (size_t i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_inputs_per_neuron);
        }
    }

    // Processes inputs through all neurons in the layer
    // @param inputs: Vector of input values shared by all neurons
    // @return: Vector of output values, one from each neuron
    std::vector<Value> feedForward(const std::vector<Value>& inputs) {
        std::vector<Value> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.feedForward(inputs));
        }
        return outputs;
    }

    // Resets gradients to zero for all neurons in the layer
    // This is called before each backward pass to clear accumulated gradients
    void zero_grad() {
        for (auto& neuron : neurons) {
            neuron.zero_grad();
        }
    }

    // Updates the parameters (weights and biases) of all neurons in the layer
    // using gradient descent with the specified learning rate
    // @param learning_rate: Step size for gradient descent update
    void update_parameters(double learning_rate) {
        for (auto& neuron : neurons) {
            neuron.update_parameters(learning_rate);
        }
    }

    // Getter for accessing the layer's neurons
    std::vector<Neuron>& getNeurons() { return neurons; }
};

// Neural Network class represents the complete neural network structure
class NeuralNet {
private:
    std::vector<Layer> layers;     // Collection of layers forming the network

public:
    // Constructor creates a neural network with specified topology
    // @param topology: Vector where each element specifies number of neurons in that layer
    //                 First element is input size, last is output size
    NeuralNet(const std::vector<size_t>& topology) {
        // Create layers based on topology
        // Start from i=1 since first element is input size
        for (size_t i = 1; i < topology.size(); ++i) {
            layers.emplace_back(topology[i], topology[i-1]);
        }
    }

    // Processes inputs through all layers of the network
    // @param inputs: Initial input values to the network
    // @return: Final output values after processing through all layers
    std::vector<Value> feedForward(const std::vector<Value>& inputs) {
        std::vector<Value> current_inputs = inputs;
        for (auto& layer : layers) {
            current_inputs = layer.feedForward(current_inputs);
        }
        return current_inputs;
    }

    // Training function
    void train(const std::vector<std::vector<Value>>& X_train,
               const std::vector<std::vector<Value>>& y_train,
               size_t epochs,
               double learning_rate) {
        std::cout << "Starting epoch loop..." << std::endl;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "Epoch " << epoch << " started" << std::endl;
            Value total_loss(0.0);
            
            std::cout << "Processing " << X_train.size() << " samples" << std::endl;
            for (size_t i = 0; i < X_train.size(); ++i) {
                // Forward pass
                std::cout << "Sample " << i << ": Forward pass" << std::endl;
                auto predictions = feedForward(X_train[i]);
                
                // Debug predictions
                std::cout << "Predictions: ";
                for (const auto& p : predictions) {
                    std::cout << p.getData() << " ";
                }
                std::cout << "\nTargets: ";
                for (const auto& t : y_train[i]) {
                    std::cout << t.getData() << " ";
                }
                std::cout << std::endl;
                
                // Compute loss
                std::cout << "Sample " << i << ": Computing loss" << std::endl;
                auto loss = Loss::mse(predictions, y_train[i]);
                std::cout << "Loss value: " << loss.getData() << std::endl;
                
                // Zero gradients before backward pass
                std::cout << "Sample " << i << ": Zeroing gradients" << std::endl;
                for (auto& layer : layers) {
                    layer.zero_grad();
                }
                
                // Backward pass
                std::cout << "Sample " << i << ": Starting backward pass" << std::endl;
                loss.setGrad(1.0);
                std::cout << "Loss gradient: " << loss.getGrad() << std::endl;
                try {
                    loss.backward();
                } catch (const std::exception& e) {
                    std::cout << "Error during backward pass: " << e.what() << std::endl;
                    throw;
                }
                
                // Update parameters
                std::cout << "Sample " << i << ": Updating parameters" << std::endl;
                try {
                    for (auto& layer : layers) {
                        layer.update_parameters(learning_rate);
                    }
                } catch (const std::exception& e) {
                    std::cout << "Error during parameter update: " << e.what() << std::endl;
                    throw;
                }
                
                total_loss = total_loss + loss;
            }
            
            // Print epoch statistics
            double avg_loss = total_loss.getData() / X_train.size();
            std::cout << "Epoch " << epoch << " completed. Average loss: " << avg_loss << std::endl;
        }
    }
};

// Data generation for testing
namespace DataGen {
    // Generate a spiral dataset for classification tasks
    // Returns a pair of vectors containing input features (X) and one-hot encoded labels (y)
    // @param points_per_class: Number of data points to generate for each class
    // @param num_classes: Number of distinct classes/spirals to generate
    // @return: Pair of vectors {X, y} where X contains 2D coordinates and y contains one-hot labels
    inline std::pair<std::vector<std::vector<Value>>, std::vector<std::vector<Value>>> 
    make_spiral_data(int points_per_class, int num_classes) {
        // Initialize vectors to store features (X) and labels (y)
        std::vector<std::vector<Value>> X;  // Will store 2D coordinates of points
        std::vector<std::vector<Value>> y;  // Will store one-hot encoded class labels
        
        // Generate points for each class
        for (int c = 0; c < num_classes; c++) {
            // Generate multiple points along the spiral for this class
            for (int i = 0; i < points_per_class; i++) {
                // Calculate radius - increases linearly from 0 to 1 along the spiral
                double r = (double)i / points_per_class;
                // Calculate angle with some random noise - creates spiral pattern
                double t = c * 4 + 4 * r + ((double)rand() / RAND_MAX) * 2;
                
                // Convert polar coordinates (r,t) to Cartesian coordinates (x,y)
                double x = r * std::sin(t * 2.5);        // x coordinate on spiral
                double y_coord = r * std::cos(t * 2.5);  // y coordinate on spiral
                
                // Add the point to feature vector X as Value objects
                X.push_back({Value(x), Value(y_coord)});
                
                // Create one-hot encoded label vector (all 0s except 1 at class index)
                std::vector<Value> one_hot(num_classes, Value(0.0));
                one_hot[c] = Value(1.0);  // Set 1.0 at index corresponding to current class
                y.push_back(one_hot);     // Add label vector to labels
            }
        }
        
        // Return the features and labels as a pair
        return {X, y};
    }
}

} // namespace NeuralNetwork