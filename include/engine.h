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
    std::function<void(double)> backward_fn; // backward function
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
        // Share the same children nodes
        prev = other.prev;
        backward_fn = other.backward_fn;
    }

    // Destructor to clean up memory
    ~Value() {
        prev.clear();
    }
    
    // Assignment operator
    Value& operator=(const Value& other) {
        if (this != &other) {
            prev.clear();
            data = other.data;
            grad = 0.0;
            is_leaf = other.is_leaf;
            node_id = ++node_count;
            prev = other.prev;
            backward_fn = other.backward_fn;
        }
        return *this;
    }

    // Addition operator: Creates a new Value representing (this + other)
    // Sets up gradient computation for backpropagation
    // @param other: The Value to add to this
    // @return: New Value containing the sum and gradient function
    Value operator+(const Value& other) const {
        Value out(data + other.data);
        
        std::cout << "\nAddition operation:" << std::endl;
        std::cout << "  Left operand: " << data << " (node " << node_id << ")" << std::endl;
        std::cout << "  Right operand: " << other.data << " (node " << other.node_id << ")" << std::endl;
        
        // Create shared_ptr to the original nodes
        auto this_shared = std::make_shared<Value>(data);
        this_shared->grad = grad;
        this_shared->prev = prev;
        this_shared->node_id = node_id;
        this_shared->is_leaf = is_leaf;
        this_shared->backward_fn = backward_fn;
        
        auto other_shared = std::make_shared<Value>(other.data);
        other_shared->grad = other.grad;
        other_shared->prev = other.prev;
        other_shared->node_id = other.node_id;
        other_shared->is_leaf = other.is_leaf;
        other_shared->backward_fn = other.backward_fn;
        
        out.prev = {this_shared, other_shared};
        
        out.backward_fn = [this_shared, other_shared](double out_grad) {
            std::cout << "  Addition backward pass:" << std::endl;
            std::cout << "    Before update - left: " << this_shared->grad 
                      << ", right: " << other_shared->grad << std::endl;
            
            this_shared->grad += out_grad;
            other_shared->grad += out_grad;
            
            // Propagate gradients to children
            if (this_shared->backward_fn) {
                this_shared->backward_fn(this_shared->grad);
            }
            if (other_shared->backward_fn) {
                other_shared->backward_fn(other_shared->grad);
            }
            
            std::cout << "    After update - left: " << this_shared->grad 
                      << ", right: " << other_shared->grad << std::endl;
        };
        return out;
    }

    // Multiplication operator: Creates a new Value representing (this * other)
    // Sets up gradient computation for backpropagation
    // @param other: The Value to multiply with this
    // @return: New Value containing the product and gradient function
    Value operator*(const Value& other) const {
        Value out(data * other.data);
        
        std::cout << "\nMultiplication operation details:" << std::endl;
        std::cout << "  Original nodes:" << std::endl;
        std::cout << "    this: node_" << node_id << " (data=" << data 
                  << ", grad=" << grad << ", leaf=" << is_leaf << ")" << std::endl;
        std::cout << "    other: node_" << other.node_id << " (data=" << other.data 
                  << ", grad=" << other.grad << ", leaf=" << other.is_leaf << ")" << std::endl;
        
        auto this_shared = std::make_shared<Value>(data);
        this_shared->grad = grad;
        this_shared->prev = prev;
        this_shared->node_id = node_id;
        this_shared->is_leaf = is_leaf;
        this_shared->backward_fn = backward_fn;
        
        auto other_shared = std::make_shared<Value>(other.data);
        other_shared->grad = other.grad;
        other_shared->prev = other.prev;
        other_shared->node_id = other.node_id;
        other_shared->is_leaf = other.is_leaf;
        other_shared->backward_fn = other.backward_fn;
        
        out.prev = {this_shared, other_shared};
        
        out.backward_fn = [this_shared, other_shared](double out_grad) {
            std::cout << "  Backward pass for multiplication:" << std::endl;
            std::cout << "    Left node " << this_shared->node_id << " before: grad=" << this_shared->grad << std::endl;
            std::cout << "    Right node " << other_shared->node_id << " before: grad=" << other_shared->grad << std::endl;
            
            this_shared->grad += other_shared->data * out_grad;
            other_shared->grad += this_shared->data * out_grad;
            
            // Propagate gradients to children
            if (this_shared->backward_fn) {
                this_shared->backward_fn(this_shared->grad);
            }
            if (other_shared->backward_fn) {
                other_shared->backward_fn(other_shared->grad);
            }
            
            std::cout << "    Left node after: grad=" << this_shared->grad << std::endl;
            std::cout << "    Right node after: grad=" << other_shared->grad << std::endl;
        };
        return out;
    }

    void backward() {
        if (grad == 0.0) {
            grad = 1.0;
        }
        
        std::cout << "\nBackward pass for node " << node_id << ":" << std::endl;
        std::cout << "  Data: " << data << std::endl;
        std::cout << "  Gradient before: " << grad << std::endl;
        std::cout << "  Number of children: " << prev.size() << std::endl;
        std::cout << "  Node type: " << (is_leaf ? "Leaf" : "Internal") << std::endl;
        
        std::cout << "  Children data:";
        for (const auto& child : prev) {
            std::cout << "\n    Node " << child->node_id 
                      << ": data=" << child->getData() 
                      << ", grad=" << child->getGrad();
        }
        std::cout << std::endl;
        
        if (backward_fn) {
            std::cout << "  Executing backward function" << std::endl;
            backward_fn(grad);
            std::cout << "  Backward function complete" << std::endl;
        } else {
            std::cout << "  No backward function defined" << std::endl;
        }
        
        std::cout << "  Gradient after backward_fn: " << grad << std::endl;
        
        for (auto it = prev.rbegin(); it != prev.rend(); ++it) {
            std::cout << "  Propagating to child node " << (*it)->node_id 
                      << " (data=" << (*it)->getData() 
                      << ", current_grad=" << (*it)->getGrad() << ")" << std::endl;
            if (*it) {
                (*it)->backward();
            }
        }
        std::cout << "  Backward pass complete for node " << node_id << std::endl;
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
        double t = std::tanh(data);
        std::cout << "\nTanh operation:" << std::endl;
        std::cout << "  Input node: " << getDebugString() << std::endl;
        std::cout << "  Tanh value: " << t << std::endl;
        
        auto this_shared = std::make_shared<Value>(*this);
        this_shared->grad = grad;
        this_shared->prev = prev;
        this_shared->node_id = node_id;
        this_shared->is_leaf = is_leaf;
        this_shared->backward_fn = backward_fn;
        
        Value out(t, {this_shared});
        
        std::cout << "  Created output node: " << out.getDebugString() << std::endl;
        std::cout << "  Output children: " << out.getPrev().size() << std::endl;
        std::cout << "  Child node: " << this_shared->getDebugString() << std::endl;
        
        out.backward_fn = [t, this_ptr = out.prev[0]](double out_grad) {
            std::cout << "  Tanh backward pass:" << std::endl;
            std::cout << "    Input grad before: " << this_ptr->grad << std::endl;
            std::cout << "    Derivative: " << (1 - t * t) << std::endl;
            std::cout << "    Incoming gradient: " << out_grad << std::endl;
            this_ptr->grad += (1 - t * t) * out_grad;
            if (this_ptr->backward_fn) {
                this_ptr->backward_fn(this_ptr->grad);
            }
            std::cout << "    Input grad after: " << this_ptr->grad << std::endl;
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
        std::cout << "\n=== Starting MSE Computation ===" << std::endl;
        std::cout << "Number of predictions: " << predictions.size() << std::endl;
        assert(predictions.size() == targets.size());
        
        // Step 1: Collection of terms
        std::vector<Value> terms;
        terms.reserve(predictions.size());
        
        // Step 2: Computing squared differences
        for (size_t i = 0; i < predictions.size(); i++) {
            std::cout << "\nMSE computation for index " << i << ":" << std::endl;
            std::cout << "  Prediction node: " << predictions[i].getDebugString() << std::endl;
            std::cout << "  Target value: " << targets[i].getData() << std::endl;
            
            double target_val = targets[i].getData();
            Value diff = predictions[i] + Value(-target_val, {});
            std::cout << "  Created difference node: " << diff.getDebugString() << std::endl;
            std::cout << "  Difference children: " << diff.getPrev().size() << std::endl;
            
            Value squared = diff * diff;
            std::cout << "  Created squared node: " << squared.getDebugString() << std::endl;
            std::cout << "  Squared children: " << squared.getPrev().size() << std::endl;
            
            terms.push_back(squared);
        }
        
        // Step 3: Binary tree reduction
        while (terms.size() > 1) {
            std::cout << "\nBinary reduction step:" << std::endl;
            std::cout << "  Terms before reduction:" << std::endl;
            for (size_t i = 0; i < terms.size(); i++) {
                std::cout << "    Term " << i << ": " << terms[i].getDebugString() << std::endl;
            }
            
            std::vector<Value> next_level;
            for (size_t i = 0; i < terms.size(); i += 2) {
                if (i + 1 < terms.size()) {
                    Value sum = terms[i] + terms[i + 1];
                    std::cout << "  Created sum node: " << sum.getDebugString() << std::endl;
                    std::cout << "  Sum children: " << sum.getPrev().size() << std::endl;
                    next_level.push_back(sum);
                } else {
                    next_level.push_back(terms[i]);
                }
            }
            terms = std::move(next_level);
            std::cout << "  Terms after reduction: " << terms.size() << std::endl;
        }
        
        // Step 4: Final scaling
        std::cout << "\nFinal scaling step:" << std::endl;
        std::cout << "  Final term node: " << terms[0].getDebugString() << std::endl;
        std::cout << "  Final term children: " << terms[0].getPrev().size() << std::endl;
        
        Value scale(1.0 / predictions.size());
        Value final_loss = terms[0] * scale;
        
        std::cout << "\n=== MSE Computation Complete ===" << std::endl;
        std::cout << "Final loss node: " << final_loss.getDebugString() << std::endl;
        std::cout << "Loss children: " << final_loss.getPrev().size() << std::endl;
        std::cout << "Computation graph depth: " << std::endl;
        final_loss.printGraph("  ");
        
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
        Value act(0.0);
        for (size_t i = 0; i < weights.size(); i++) {
            std::cout << "\nNeuron computation step " << i << ":" << std::endl;
            std::cout << "  Input: " << inputs[i].getData() 
                      << " * Weight: " << weights[i].getData() << std::endl;
            Value product = weights[i] * inputs[i];
            std::cout << "  Product result: " << product.getData() << std::endl;
            std::cout << "  Accumulated before: " << act.getData() << std::endl;
            if (i == 0) {
                act = product;
            } else {
                act = act + product;
            }
            std::cout << "  Accumulated after: " << act.getData() << std::endl;
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
        std::cout << "\nLayer forward pass:" << std::endl;
        for (size_t i = 0; i < neurons.size(); i++) {
            std::cout << "  Neuron " << i << " computation:" << std::endl;
            auto out = neurons[i].feedForward(inputs);
            std::cout << "  Output: " << out.getData() << std::endl;
            outputs.push_back(out);
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