// https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
/*
Intro and Neuron code
*/

pub fn one() {
    // output from previous layer becomes input to another layer
    // node takes in input, bais and wights
    let input = vec![1.2, 5.1, 2.1];
    let weights = vec![3.1, 2.1, 8.7];
    let bias: f64 = 3.;
    // the output will be calculated by dot product of input and weights added to bias
    let dot_product: f64 = input.iter().zip(weights.iter()).map(|(x, y)| x * y).sum();
    let output = dot_product + bias;
    println!("Dot product is {}", output);
}

/*
OUTPUT
Dot product is 35.7
*/
