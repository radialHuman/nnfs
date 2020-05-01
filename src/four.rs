// https://www.youtube.com/watch?v=TEWy9vZcxW4

/*
Batches and OOPS
*/
use math::round;

pub fn four() {
    // BATCHES
    /*
    > Parallelization for calculating
    > Better on GPU as many 100s of cores available
    > Helps with generalization
    > To avoid overfitting at all points 1/1

    Batch size of 32 and 64 is very common
    Weights are random value initialized at the begining between -1 and 1 to avoid explosion issue
    it is normalized and scaled
    bias can be 0s, if this doesn work then due to diminishing issues change bias to a non zero value
    */

    // In this example, the input of one layer describes the status fo server from various sensors at a given time
    let input = vec![
        vec![1., 2., 3., 2.5],
        vec![2., 5., -1., 2.],
        vec![-1.5, 2.7, 3.3, -0.8],
    ]; // input in batches
    let weights1 = vec![
        vec![0.2, 0.8, -0.5, 1.],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87],
    ];
    let bias1 = vec![2., 3., 0.5];

    // transposing weights to get the matrix multiplication
    // let weights_t = transpose(&weights);
    // println!("Transpose of {:?} = {:?}", weights, weights_t);
    let mat_mul1 = matrix_product(&input, &weights1);
    // println!("input * weights1= {:?}", mat_mul);

    // adding the bias
    let mut layer1_output = vec![];
    for i in mat_mul1 {
        layer1_output.push(vector_addition(&i, &bias1));
    }
    println!("Layer 1 output (input * weights1+ bias1) = ",);
    print_a_matrix(&layer1_output);

    // second layer with different weights and bias
    let weights2 = vec![
        vec![0.1, -0.14, 0.5],
        vec![-0.5, 0.12, -0.33],
        vec![-0.44, 0.73, -0.13],
    ];
    let bias2 = vec![-1., 2., -0.5];

    // output of layer 1 become input for layer 2
    let mat_mul2 = matrix_product(&layer1_output, &weights2);
    let mut layer2_output: Vec<Vec<f64>> = vec![];
    for i in mat_mul2 {
        layer2_output.push(vector_addition(&i, &bias2));
    }

    println!("Layer 2 output (layer 1 output * weights2 + bias2) =",);
    print_a_matrix(&layer2_output);
}

pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut output = vec![];
    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            output.push(matrix[i][j]);
        }
    }
    shape_changer(&mut output, 3, 4)
}

pub fn shape_changer(list: &Vec<f64>, columns: usize, rows: usize) -> Vec<Vec<f64>> {
    /*Changes a list to desired shape matrix*/
    let mut l = list.clone();
    let mut output = vec![vec![]; rows];
    for i in 0..rows {
        output[i] = l[..columns].iter().cloned().collect();
        // remove the ones pushed to putput
        l = l[columns..].iter().cloned().collect();
    }
    output
}

pub fn matrix_product(input: &Vec<Vec<f64>>, weights: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if input.len() == weights.len() {
        let mut output: Vec<f64> = vec![];
        for i in input.iter() {
            for j in weights.iter() {
                output.push(dot_product(&i, &j));
            }
        }
        // println!("{:?}", output);
        shape_changer(&output, input.len(), weights.len())
    } else {
        println!("The matrix is invalid");
        input.clone()
    }
}

pub fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

pub fn vector_addition(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    if a.len() == b.len() {
        let mut output = vec![];
        for i in 0..a.len() {
            output.push(round::ceil(a[i], 3) + round::ceil(b[i], 3));
        }
        output
    } else {
        println!("The vector is invalid",);
        a.clone()
    }
}

pub fn print_a_matrix(matrix: &Vec<Vec<f64>>) {
    for i in matrix.iter() {
        println!("{:?}", i);
    }
    println!("");
    println!("");
}
/*
OUTPUT
Layer 1 output (input * weights1+ bias1) =
[4.8, 1.21, 2.385]
[8.9, -1.8099999999999996, 0.2]
[1.411, 1.051, 0.026000000000000023]


Layer 2 output (layer 1 output * weights2 + bias2) =
[0.504, -1.041, -2.0380000000000003]
[0.244, -2.7329999999999997, -5.763]
[-0.993, 1.413, -0.356]


*/
