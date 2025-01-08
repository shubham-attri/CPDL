#include "../include/engine.h"
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>

using namespace std;
using namespace NeuralNetwork;

void testLayer();
void testMLP();


// Helper function to generate unique IDs for DOT nodes
std::string getDotId(const void* ptr) {
    std::stringstream ss;
    ss << "node_" << ptr;
    return ss.str();
}

// Helper function to create DOT node with formatting
std::string createDotNode(const void* ptr, const std::string& label, const std::string& shape = "box") {
    std::stringstream ss;
    ss << getDotId(ptr) << " [label=\"" << label << "\" shape=" << shape << "];\n";
    return ss.str();
}

// Helper function to get node color based on operation type and gradient value
std::string getNodeColor(const Value* v) {
    #ifdef DEBUG_MODE
    // Color based on operation type
    switch(v->getOp()) {
        case '+': return "\"#FFB6C1\"";  // Light pink for addition
        case '*': return "\"#98FB98\"";  // Pale green for multiplication
        case 't': return "\"#87CEEB\"";  // Sky blue for tanh
        default: return "\"#D3D3D3\"";   // Light gray for inputs
    }
    #else
    // Color based on gradient magnitude when DEBUG_MODE is off
    double grad_mag = std::abs(v->getGrad());
    if (grad_mag > 1.0) return "\"#FF6B6B\"";  // Strong red for high gradients
    if (grad_mag > 0.5) return "\"#FFB6C1\"";  // Light red for medium gradients
    if (grad_mag > 0.1) return "\"#FFF0F0\"";  // Very light red for small gradients
    return "\"#F8F8F8\"";  // Almost white for tiny/zero gradients
    #endif
}

// Helper function to get node shape based on node type
std::string getNodeShape(const Value* v) {
    if (v->getPrev().empty()) return "ellipse";  // Input nodes
    #ifdef DEBUG_MODE
    switch(v->getOp()) {
        case 't': return "diamond";  // Activation functions
        case '+': return "box";      // Addition
        case '*': return "octagon";  // Multiplication
        default: return "box";
    }
    #else
    return "box";
    #endif
}

void generateDotFile(const Value& root, const std::string& filename, bool isNeuron = false) {
    std::cout << "Starting DOT file generation for " << filename << std::endl;
    
    std::ofstream dot_file(filename);
    if (!dot_file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    std::cout << "Writing graph header..." << std::endl;
    dot_file << "digraph ComputationalGraph {\n";
    dot_file << "  rankdir=TB;\n";
    dot_file << "  bgcolor=\"white\";\n";
    dot_file << "  node [style=filled, fontname=\"Arial\"];\n";
    dot_file << "  edge [color=\"#666666\"];\n";
    
    std::cout << "Initializing data structures..." << std::endl;
    std::set<const Value*> visited;
    std::map<const Value*, std::string> node_labels;
    std::vector<std::set<const Value*>> layers;
    
    // First pass: collect nodes and their depths
    std::map<const Value*, int> node_depths;
    std::function<int(const Value*)> compute_depth = [&](const Value* v) -> int {
        if (!v) return -1;  // Handle null pointers
        if (node_depths.find(v) != node_depths.end()) {
            return node_depths[v];
        }
        
        if (v->getPrev().empty()) {
            node_depths[v] = 0;
            return 0;
        }
        
        int max_child_depth = -1;
        for (const auto& child : v->getPrev()) {
            if (child) {  // Check for null
                max_child_depth = std::max(max_child_depth, compute_depth(child.get()));
            }
        }
        
        node_depths[v] = max_child_depth + 1;
        return node_depths[v];
    };
    
    // Compute depths starting from root
    std::cout << "Computing node depths..." << std::endl;
    int max_depth = compute_depth(&root);
    
    // Initialize layers vector
    layers.resize(max_depth + 1);
    
    // Assign nodes to layers based on their depths
    std::cout << "Assigning nodes to layers..." << std::endl;
    for (const auto& [node, depth] : node_depths) {
        layers[depth].insert(node);
    }
    
    std::cout << "Creating subgraphs for " << layers.size() << " layers..." << std::endl;
    for (size_t i = 0; i < layers.size(); i++) {
        std::cout << "Processing layer " << i << " with " << layers[i].size() << " nodes..." << std::endl;
        dot_file << "  subgraph cluster_layer_" << i << " {\n";
        dot_file << "    style=invis;\n";
        dot_file << "    rank=same;\n";
        
        for (const Value* v : layers[i]) {
            std::stringstream label;
            label << "{";
            #ifdef DEBUG_MODE
            label << "op: " << v->getOp() << "|";
            #endif
            label << "data: " << std::fixed << std::setprecision(4) << v->getData();
            label << "|grad: " << std::fixed << std::setprecision(4) << v->getGrad();
            label << "}";
            
            node_labels[v] = getDotId(v);
            
            dot_file << "    " << getDotId(v) << " ["
                    << "label=\"" << label.str() << "\", "
                    << "shape=\"record\", "
                    << "fillcolor=" << getNodeColor(v) << ", "
                    << "penwidth=2"
                    << "];\n";
        }
        dot_file << "  }\n";
    }
    
    std::cout << "Creating edges..." << std::endl;
    for (const auto& [node, _] : node_labels) {
        for (const auto& child : node->getPrev()) {
            if (child && node_labels.find(child.get()) != node_labels.end()) {
                dot_file << "  " << node_labels[child.get()] << " -> " << node_labels[node];
                dot_file << " [label=\"grad: " << std::fixed << std::setprecision(4) 
                        << child->getGrad() << "\"];\n";
            }
        }
    }
    
    std::cout << "Finalizing graph..." << std::endl;
    dot_file << "}\n";
    dot_file.close();
    std::cout << "DOT file generation complete for " << filename << std::endl;
}

void visualizeValue() {
    std::cout << "\n=== Value Class Computational Graph Visualization ===" << std::endl;
    
    // Create inputs and weights
    Value x1(2.0);
    Value w1(-3.0);
    std::cout << "Created input values: x1=" << x1.getData() << ", w1=" << w1.getData() << std::endl;
    
    // Compute multiplication
    Value mul1 = x1 * w1;
    std::cout << "Computed multiplication: " << mul1.getData() << std::endl;
    
    // Generate DOT file before backward pass
    std::cout << "Generating forward pass graph..." << std::endl;
    generateDotFile(mul1, "value_graph_forward.dot");
    
    // Backward pass
    std::cout << "Running backward pass..." << std::endl;
    mul1.backward();
    
    // Generate DOT file after backward pass
    std::cout << "Generating backward pass graph..." << std::endl;
    generateDotFile(mul1, "value_graph_backward.dot");
    
    std::cout << "Visualization complete!" << std::endl;
}

void visualizeNeuron() {
    std::cout << "\n=== Neuron Computational Graph Visualization ===" << std::endl;
    
    // Create a neuron with 2 inputs
    Neuron neuron(2);
    std::cout << "Created neuron with 2 inputs" << std::endl;
    
    const auto& weights = neuron.getWeights();
    const auto& bias = neuron.getBias();
    
    // Print initial weights and bias
    std::cout << "Initial weights: [";
    for (const auto& w : weights) {
        std::cout << w.getData() << " ";
    }
    std::cout << "], bias: " << bias.getData() << std::endl;
    
    // Create input values
    std::vector<Value> inputs = {Value(1.0), Value(0.5)};
    std::cout << "Created input values: [" << inputs[0].getData() << ", " << inputs[1].getData() << "]" << std::endl;
    
    try {
        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        Value output = neuron.feedForward(inputs);
        std::cout << "Forward pass output: " << output.getData() << std::endl;
        
        // Generate DOT file before backward pass
        std::cout << "Generating forward pass graph..." << std::endl;
        generateDotFile(output, "neuron_graph_forward.dot", true);
    
    // Backward pass
        std::cout << "Running backward pass..." << std::endl;
        std::cout << "Initial output gradient: " << output.getGrad() << std::endl;
        output.backward();
        
        // Print gradients after backward pass
        std::cout << "Gradients after backward pass:" << std::endl;
        std::cout << "Input gradients: [";
        for (const auto& input : inputs) {
            std::cout << input.getGrad() << " ";
        }
        std::cout << "]\nWeight gradients: [";
        for (const auto& w : weights) {
            std::cout << w.getGrad() << " ";
        }
        std::cout << "]\nBias gradient: " << bias.getGrad() << std::endl;
        
        // Generate DOT file after backward pass
        std::cout << "Generating backward pass graph..." << std::endl;
        generateDotFile(output, "neuron_graph_backward.dot", true);
        
        std::cout << "Visualization complete!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during neuron visualization: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown error during neuron visualization" << std::endl;
        throw;
    }
}

bool generatePNG(const std::string& dot_file, const std::string& png_file) {
    std::string command = "dot -Tpng " + dot_file + " -o " + png_file;
    return system(command.c_str()) == 0;
}


int visuliaztion() {
    try {

        


        std::cout << "Starting visualization..." << std::endl;
        
        std::cout << "\n=== Value Class Demonstration ===" << std::endl;
        visualizeValue();
        
        std::cout << "\n=== Neuron Class Demonstration ===" << std::endl;
        visualizeNeuron();
        
        std::cout << "\nAll visualizations completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

int main() {

    
    // visuliaztion();
    // testLayer();
    // testMLP();

    // Neuron neuron(2);
    // std::vector<Value> inputs = {Value(1.0), Value(0.5)};
    // Value output = neuron.feedForward(inputs);
    // std::cout << "Neuron output: " << output.getData() << std::endl;

    // Create a simple MLP for binary classification
    std::cout << "\n=== Simple Neural Network Training Example ===" << std::endl;
    MLP mlp({1, 2, 1}); // 1 input, 2 hidden neurons, 1 output

    // Simple training data: y = 2x
    std::vector<std::vector<double>> X = {{1}, {2}, {3}};  // Input values
    std::vector<double> y = {2, 4, 6};                     // Target values (2x)

    // Just 3 epochs to see the progression
    int epochs = 3;
    double learning_rate = 0.1;

    std::cout << "\nWatching the network learn y = 2x:" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << ":" << std::endl;
        
        // Train on each example
        for (size_t i = 0; i < X.size(); i++) {
            auto pred = mlp.feedForward(X[i]);
            double target = y[i];
            
            std::cout << "Input: " << X[i][0] << ", Target: " << target 
                     << ", Prediction: " << std::fixed << std::setprecision(3) 
                     << pred[0].getData() << std::endl;

            // Backward pass and update
            mlp.zeroGrad();
            pred[0].grad = 2.0 * (pred[0].getData() - target);
            pred[0].backward();
            mlp.step(learning_rate);
        }
    }

    return 0;
}

void testLayer() {
    std::cout << "\n=== Layer Test Visualization ===" << std::endl;
    
    // Create a layer with 2 inputs and 3 neurons
    Layer layer(2, 3);
    
    // Create some test inputs
    std::vector<Value> inputs = {Value(1.0), Value(0.5)};
    
    // Feed forward
    std::vector<Value> outputs = layer.feedForward(inputs);
    
    // Print results
    std::cout << "Layer test:" << std::endl;
    std::cout << "Input size: " << layer.getInputSize() << std::endl;
    std::cout << "Output size: " << layer.getOutputSize() << std::endl;
    std::cout << "Outputs: ";
    for (const auto& out : outputs) {
        std::cout << out.getData() << " ";
    }
    std::cout << std::endl;
    
    // Generate forward pass visualization
    std::cout << "Generating forward pass graph..." << std::endl;
    generateDotFile(outputs[0], "layer_graph_forward.dot");  // Visualize first neuron's computation
    
    // Test backward pass
    std::cout << "Running backward pass..." << std::endl;
    outputs[0].backward();
    
    // Generate backward pass visualization
    std::cout << "Generating backward pass graph..." << std::endl;
    generateDotFile(outputs[0], "layer_graph_backward.dot");
    
    // Test parameter access
    auto params = layer.getParameters();
    std::cout << "Total parameters: " << params.size() << std::endl;
}

void testMLP() {
    std::cout << "\n=== MLP Test Visualization ===" << std::endl;
    
    // Create MLP with architecture: 2 -> 3 -> 1
    MLP mlp({2, 3, 1});
    
    // Test inputs
    std::vector<Value> inputs = {Value(1.0), Value(-1.0)};
    
    // Forward pass
    std::cout << "MLP test:" << std::endl;
    std::cout << "Input size: " << mlp.getInputSize() << std::endl;
    std::cout << "Output size: " << mlp.getOutputSize() << std::endl;
    
    auto outputs = mlp.feedForward(inputs);
    std::cout << "Outputs: ";
    for (const auto& out : outputs) {
        std::cout << out.getData() << " ";
    }
    std::cout << std::endl;
    
    // Generate forward pass visualization
    std::cout << "Generating forward pass graph..." << std::endl;
    generateDotFile(outputs[0], "mlp_graph_forward.dot");
    
    // Test gradient computation
    std::cout << "Running backward pass..." << std::endl;
    outputs[0].backward();
    
    // Generate backward pass visualization
    std::cout << "Generating backward pass graph..." << std::endl;
    generateDotFile(outputs[0], "mlp_graph_backward.dot");
    
    // Test parameter access
    auto params = mlp.getParameters();
    std::cout << "Total parameters: " << params.size() << std::endl;
    
    // Test optimizer step
    double learning_rate = 0.01;
    mlp.step(learning_rate);
    
    // Zero gradients
    mlp.zeroGrad();
}