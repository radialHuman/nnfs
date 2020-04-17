// https://www.youtube.com/watch?v=lGLto9Xd7bU
/*
Coding a layer
*/
#[derive(Debug)]
struct NodeInput {
    input: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
}

impl NodeInput {
    pub fn dot_product(&self) -> f64 {
        let output: f64 = self
            .input
            .iter()
            .zip(self.weights.iter())
            .map(|(x, y)| x * y)
            .sum();
        output + self.bias
    }
}

pub fn two() {
    // new code, using struct
    let a_node = NodeInput {
        // every neuron has just one bias while every input has one weight
        // these inputs can be from data source / from another layer in the n/w
        input: vec![1., 2., 3., 2.5],
        weights: vec![0.2, 0.8, -0.5, 1.],
        bias: 2.,
    };
    println!(
        "Dot product for one neuron with 4 inputs is {}",
        a_node.dot_product()
    );

    // for 3 neurons with 4 inputs, there will be different weights and bias
    let b_node = NodeInput {
        input: vec![1., 2., 3., 2.5], // input remains the same
        weights: vec![0.5, -0.91, 0.26, -0.5],
        bias: 3.,
    };
    let c_node = NodeInput {
        input: vec![1., 2., 3., 2.5], // input remains the same
        weights: vec![-0.26, -0.27, 0.17, 0.87],
        bias: 0.5,
    };
    let mut a_layer: Vec<NodeInput> = vec![a_node, b_node, c_node];

    // output of this layer will be like array of input to the layer
    let output = a_layer
        .iter()
        .map(|x| x.dot_product())
        .collect::<Vec<f64>>();
    println!("Output of this layer is {:?}", output);
}

/*
OUTPUT
Dot product for one neuron with 4 inputs is 4.8
Output of this layer is [4.8, 1.21, 2.385]
*/
