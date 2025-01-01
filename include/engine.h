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
    std::vector<std::shared_ptr<Value>> prev; // previous nodes
    std::function<void()> backward_fn; // backward function
    static size_t node_count;  // For debugging: track number of nodes
    size_t node_id;           // For debugging: unique identifier for each node
    bool is_leaf;  // Flag to mark leaf nodes (no gradients needed)

public:
    // Constructor: Creates a Value node with given data and optional children nodes
    // @param data: The numerical value to store
    // @param children: Vector of pointers to child nodes in computation graph (default empty)
    explicit Value(double data, std::vector<std::shared_ptr<Value>> children = {}) 
        : data(data), grad(0.0), prev(children), 
         is_leaf(children.empty()) {
        node_id = ++node_count;
        
        // Validate all child pointers
        for (size_t i = 0; i < children.size(); i++) {
            if (!children[i]) {
                throw std::runtime_error("Null child pointer in Value constructor");
            }
        }
    }

    // Copy constructor to ensure proper node ownership
    Value(const Value& other) 
        : data(other.data), grad(0.0), is_leaf(other.is_leaf) {
        node_id = ++node_count;
        // Deep copy the prev vector
        for (const auto& p : other.prev) {
            prev.push_back(std::make_shared<Value>(*p));
        }
    }

    // Addition operator: Creates a new Value representing (this + other)
    // Sets up gradient computation for backpropagation
    // @param other: The Value to add to this
    // @return: New Value containing the sum and gradient function
    Value operator+(const Value& other) const {
        Value out(data + other.data, {std::make_shared<Value>(*this), std::make_shared<Value>(other)});
        
        out.backward_fn = [out_ptr = &out]() {
            out_ptr->prev[0]->grad += out_ptr->grad;
            out_ptr->prev[1]->grad += out_ptr->grad;
        };
        return out;
    }

    // Multiplication operator: Creates a new Value representing (this * other)
    // Sets up gradient computation for backpropagation
    // @param other: The Value to multiply with this
    // @return: New Value containing the product and gradient function
    Value operator*(const Value& other) const {
        // Store pointers to original nodes
        auto this_ptr = std::make_shared<Value>(data);  // Just store the value
        auto other_ptr = std::make_shared<Value>(other.data);
        this_ptr->grad = grad;  // Copy current gradients
        other_ptr->grad = other.grad;
        
        Value out(data * other.data, {this_ptr, other_ptr});
        
        double this_data = data;
        double other_data = other.data;
        
        // Capture original nodes by reference to update their gradients
        out.backward_fn = [this_ptr, other_ptr, this_data, other_data, 
                          orig_this = this, orig_other = &other]() {
            this_ptr->grad += other_data;
            other_ptr->grad += this_data;
            const_cast<Value*>(orig_this)->grad = this_ptr->grad;
            const_cast<Value*>(orig_other)->grad = other_ptr->grad;
        };
        return out;
    }

    void backward() {
        if (!backward_fn) {
            std::cout << "Warning: No backward function for node " << node_id << std::endl;
            return;
        }
        
        try {
            backward_fn();
        } catch (const std::exception& e) {
            std::cout << "Error in backward pass for node " << node_id << ": " << e.what() << std::endl;
            throw;
        }
    }

    // Returns the data value stored in this Value object
    double getData() const { return data; }
    // Returns the gradient value stored in this Value object
    double getGrad() const { return grad; }
    // Sets the gradient value to the specified value
    void setGrad(double g) { grad = g; }
    const std::vector<std::shared_ptr<Value>>& getPrev() const { return prev; }

    // Applies the hyperbolic tangent (tanh) activation function to this Value
    // @return: A new Value containing tanh(data) and gradient computation
    Value tanh() const {
        // Compute tanh of the input data
        double t = std::tanh(data);
        
        // Create new Value node with the tanh result
        Value out(t, {std::make_shared<Value>(*this)});
        
        // Set up gradient computation for backpropagation
        out.backward_fn = [t, this, out_ptr = &out]() {
            // Update gradient using chain rule:
            // ∂out/∂x = (1 - tanh²(x)) * ∂L/∂out 
            grad += (1 - t * t) * out_ptr->grad;
        };
        return out;
    }

    // Add method to get node info for debugging
    std::string getDebugString() const {
        return "Node_" + std::to_string(node_id) + 
               "(data=" + std::to_string(data) + 
               ", grad=" + std::to_string(grad) + 
               ", prev_count=" + std::to_string(prev.size()) + ")";
    }

    // Debug method to print computation graph
    void printGraph(std::string prefix = "") const {
        std::cout << prefix << getDebugString() << std::endl;
        for (const auto& child : prev) {
            if (child) {
                child->printGraph(prefix + "  ");
            } else {
                std::cout << prefix + "  NULL CHILD" << std::endl;
            }
        }
    }
};

// Initialize static member
size_t Value::node_count = 0;

// Loss functions namespace
namespace Loss {
    // Mean Squared Error (MSE) loss
    // Mean Squared Error (MSE) loss function
    // @param predictions: Vector of predicted values from the network
    // @param targets: Vector of target/ground truth values to compare against
    // @return: A Value object containing the MSE loss value
    inline Value mse(const std::vector<Value>& predictions, const std::vector<Value>& targets) {
        assert(predictions.size() == targets.size());
        
        // Step 1: Collection of terms
        std::vector<Value> terms;
        terms.reserve(predictions.size());
        
        // Step 2: Computing squared differences
        for (size_t i = 0; i < predictions.size(); i++) {
            double target_val = targets[i].getData();
            Value diff = predictions[i] + Value(-target_val, {});
            Value squared = diff * diff;
            terms.push_back(squared);
        }
        
        // Step 3: Binary tree reduction
        while (terms.size() > 1) {
            std::vector<Value> next_level;
            for (size_t i = 0; i < terms.size(); i += 2) {
                if (i + 1 < terms.size()) {
                    Value sum = terms[i] + terms[i + 1];
                    next_level.push_back(sum);
                } else {
                    next_level.push_back(terms[i]);
                }
            }
            terms = std::move(next_level);
        }
        
        // Step 4: Final scaling
        Value scale(1.0 / predictions.size());
        Value final_loss = terms[0] * scale;
        
        std::cout << "Loss: " << final_loss.getData() << std::endl;
        
        return final_loss;
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
        std::cout << "\nFeedforward Debug:" << std::endl;
        std::vector<Value> current_inputs = inputs;
        
        Value act(0.0);  // Declare the accumulator variable!
        
        for (size_t i = 0; i < weights.size(); i++) {
            std::cout << "Processing weight " << i << ":" << std::endl;
            std::cout << "  Input: " << current_inputs[i].getDebugString() << std::endl;
            std::cout << "  Weight: " << weights[i].getDebugString() << std::endl;
            
            Value product = weights[i] * current_inputs[i];
            std::cout << "  Product: " << product.getDebugString() << std::endl;
            
            if (i == 0) {
                act = product;
            } else {
                act = act + product;
            }
            std::cout << "  Accumulated: " << act.getDebugString() << std::endl;
        }
        
        act = (act + bias).tanh();
        std::cout << "Final output: " << act.getDebugString() << std::endl;
        return act;
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
    const std::vector<Value>& getWeights() const { return weights; }
    const Value& getBias() const { return bias; }
    const Value& getOutput() const { return output; }
    // Non-const versions for when we need to modify
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
    const std::vector<Neuron>& getNeurons() const { return neurons; }
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
        std::cout << "\n=== Starting Training ===" << std::endl;
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            Value epoch_loss(0.0);
            std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << std::endl;
            
            for (size_t i = 0; i < X_train.size(); ++i) {
                std::cout << "\n--- Sample " << i + 1 << "/" << X_train.size() << " ---" << std::endl;
                
                // Debug forward pass
                std::cout << "1. Starting forward pass..." << std::endl;
                auto predictions = feedForward(X_train[i]);
                std::cout << "   Predictions size: " << predictions.size() << std::endl;
                for (size_t p = 0; p < predictions.size(); p++) {
                    std::cout << "   Prediction " << p << ": " << predictions[p].getData() << std::endl;
                }
                
                // Debug loss computation
                std::cout << "2. Computing loss..." << std::endl;
                std::cout << "   Target size: " << y_train[i].size() << std::endl;
                Value loss = Loss::mse(predictions, y_train[i]);
                std::cout << "   Loss computed: " << loss.getData() << std::endl;
                
                // Debug gradient initialization
                std::cout << "3. Initializing gradients..." << std::endl;
                std::cout << "   Number of layers: " << layers.size() << std::endl;
                for (size_t l = 0; l < layers.size(); l++) {
                    std::cout << "   Layer " << l << " neurons: " << layers[l].getNeurons().size() << std::endl;
                    layers[l].zero_grad();
                }
                
                // Debug backward pass
                std::cout << "4. Starting backward pass..." << std::endl;
                loss.backward();
                std::cout << "   Backward pass complete" << std::endl;
                
                // Debug gradient state after backward pass
                std::cout << "5. Checking gradients after backward pass..." << std::endl;
                for (size_t l = 0; l < layers.size(); l++) {
                    auto& layer = layers[l];
                    std::cout << "   Layer " << l << ":" << std::endl;
                    for (size_t n = 0; n < layer.getNeurons().size(); n++) {
                        auto& neuron = layer.getNeurons()[n];
                        std::cout << "     Neuron " << n << " gradients:" << std::endl;
                        
                        // Debug weights
                        const auto& weights = neuron.getWeights();
                        std::cout << "       Weights (" << weights.size() << "): ";
                        for (const auto& w : weights) {
                            std::cout << w.getGrad() << " ";
                        }
                        std::cout << std::endl;
                        
                        // Debug bias
                        std::cout << "       Bias grad: " << neuron.getBias().getGrad() << std::endl;
                    }
                }
                
                // Debug parameter updates
                std::cout << "6. Starting parameter updates..." << std::endl;
                try {
                    for (size_t l = 0; l < layers.size(); l++) {
                        std::cout << "   Updating Layer " << l << std::endl;
                        auto& layer = layers[l];
                        for (size_t n = 0; n < layer.getNeurons().size(); n++) {
                            std::cout << "     Updating Neuron " << n << std::endl;
                            auto& neuron = layer.getNeurons()[n];
                            neuron.update_parameters(learning_rate);
                        }
                    }
                    std::cout << "   Parameter updates complete" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "EXCEPTION during parameter update: " << e.what() << std::endl;
                    throw;
                } catch (...) {
                    std::cout << "UNKNOWN EXCEPTION during parameter update!" << std::endl;
                    throw;
                }
                
                epoch_loss = epoch_loss + loss;
            }
            
            std::cout << "\nEpoch " << epoch + 1 << " complete. "
                      << "Average loss: " << epoch_loss.getData() / X_train.size() 
                      << std::endl;
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