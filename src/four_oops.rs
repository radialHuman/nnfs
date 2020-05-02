// https://www.youtube.com/watch?v=TEWy9vZcxW4

/*
Batches and OOPS
*/

use math::round;
use rand::*;

// to create weights using random values than typying it in
struct LayerDetails {
    n_inputs: usize,
    n_neurons: i32,
}
impl LayerDetails {
    pub fn create_weights(&self) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut weight: Vec<Vec<f64>> = vec![];
        // this gives transposed weights
        for _ in 0..self.n_inputs {
            weight.push(
                (0..self.n_neurons)
                    .map(|_| round::ceil(rng.gen_range(-1., 1.), 3))
                    .collect(),
            );
        }
        weight
    }
    pub fn create_bias(&self) -> Vec<f64> {
        let bias = vec![1.; self.n_neurons as usize];
        bias
    }
    pub fn output_of_layer(
        &self,
        input: &Vec<Vec<f64>>,
        weights: &Vec<Vec<f64>>,
        bias: &Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let mat_mul = transpose(&matrix_product(&input, &weights));
        // println!("input * weights = {:?}", mat_mul);
        let mut output: Vec<Vec<f64>> = vec![];
        for i in mat_mul {
            // println!("i*w {:?}, bias {:?}", &i, &bias);
            output.push(vector_addition(&i, &bias));
        }
        transpose(&output)
    }
}

pub fn four_oops() {
    let X = vec![
        vec![1., 2., 3., 2.5],
        vec![2., 5., -1., 2.],
        vec![-1.5, 2.7, 3.3, -0.8],
    ]; // input in batches
       // OOPS
    let l1 = LayerDetails {
        n_inputs: 4,  // number of features/columns in X
        n_neurons: 5, // can be anything depending the situation
    };
    println!("The input of layer 1 are = {}x{}", X.len(), X[0].len());
    print_a_matrix(&X);
    let generated_weights1 = l1.create_weights();
    println!(
        "The weights of layer 1 are = {}x{}",
        generated_weights1.len(),
        generated_weights1[0].len()
    );
    print_a_matrix(&generated_weights1);
    // print_a_matrix(&transpose(&generated_weights1));
    let generated_bias1 = l1.create_bias();
    println!(
        "The bias of layer 1 is  = 1x{}\n{:?}\n",
        generated_bias1.len(),
        &generated_bias1
    );

    let generated_output1 =
        transpose(&l1.output_of_layer(&X, &generated_weights1, &generated_bias1));
    println!(
        "The output of layer 1 are = {}x{}",
        generated_output1.len(),
        generated_output1[0].len()
    );
    print_a_matrix(&generated_output1);

    // layer 2
    println!(
        "=> Number of input for the next layer is {}\n",
        generated_output1[0].len()
    );
    let l2 = LayerDetails {
        n_inputs: generated_output1[0].len(), // has to be that of the previous layer
        n_neurons: 2,                         // can be anything depending the situation
    };
    let generated_weights2 = l2.create_weights();
    let generated_bias2 = l2.create_bias();
    println!(
        "The inputs of layer 2 are = {}x{}",
        generated_output1.len(),
        generated_output1[0].len()
    );
    print_a_matrix(&generated_output1);
    println!(
        "The weights of layer 2 are = {}x{}",
        generated_weights2.len(),
        generated_weights2[0].len()
    );
    print_a_matrix(&generated_weights2);
    println!(
        "The bias of layer 2 is = 1x{}\n{:?}\n",
        generated_bias2.len(),
        &generated_bias2
    );
    let generated_output2 =
        transpose(&l2.output_of_layer(&generated_output1, &generated_weights2, &generated_bias2));
    println!(
        "The outputs of layer 2 are = {}x{}",
        generated_output2.len(),
        generated_output2[0].len()
    );
    print_a_matrix(&generated_output2);
}

pub fn vector_addition(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut output = vec![];
    for i in 0..a.len() {
        output.push(round::ceil(a[i], 3) + round::ceil(b[i], 3));
    }
    output
}

pub fn print_a_matrix(matrix: &Vec<Vec<f64>>) {
    for i in matrix.iter() {
        println!("{:?}", i);
    }
    println!("");
    println!("");
}

pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut output = vec![];
    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            output.push(matrix[i][j]);
        }
    }
    let x = matrix[0].len();
    shape_changer(&output, matrix.len(), x)
}
pub fn shape_changer(list: &Vec<f64>, columns: usize, rows: usize) -> Vec<Vec<f64>> {
    /*Changes a list to desired shape matrix*/
    // println!("{},{}", &columns, &rows);
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
    // println!(
    //     "Multiplication of {}x{} and {}x{}",
    //     input.len(),
    //     input[0].len(),
    //     weights.len(),
    //     weights[0].len()
    // );
    // println!("Weights transposed to",);
    let weights_t = transpose(&weights);
    // print_a_matrix(&weights_t);
    let mut output: Vec<f64> = vec![];
    for i in input.iter() {
        for j in weights_t.iter() {
            // println!("{:?}x{:?},", i, j);
            output.push(dot_product(&i, &j));
        }
    }
    // println!("{:?}", output);
    shape_changer(&output, input.len(), weights_t.len())
}

pub fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

/*
OUTPUT

The input of layer 1 are = 3x4
[1.0, 2.0, 3.0, 2.5]
[2.0, 5.0, -1.0, 2.0]
[-1.5, 2.7, 3.3, -0.8]


The weights of layer 1 are = 4x5
[-0.438, -0.295, 0.701, -0.887, 0.48]
[0.373, -0.359, 0.773, 0.084, 0.034]
[0.542, 0.92, -0.537, 0.896, -0.413]
[0.097, 0.625, 0.326, -0.638, -0.203]


The bias of layer 1 is  = 1x5
[1.0, 1.0, 1.0, 1.0, 1.0]

The output of layer 1 are = 3x5
[3.177, 1.375, -1.0539999999999998, 2.137, 0.0030000000000000027]
[4.3100000000000005, -0.19799999999999995, 7.456, 4.3759999999999994, 6.025]
[2.451, 1.641, -2.526, 3.01, -0.8280000000000001]


=> Number of input for the next layer is 5

The inputs of layer 2 are = 3x5
[3.177, 1.375, -1.0539999999999998, 2.137, 0.0030000000000000027]
[4.3100000000000005, -0.19799999999999995, 7.456, 4.3759999999999994, 6.025]
[2.451, 1.641, -2.526, 3.01, -0.8280000000000001]


The weights of layer 2 are = 5x2
[-0.99, -0.991]
[0.088, 0.651]
[0.984, 0.17]
[0.077, 0.632]
[0.55, 0.721]


The bias of layer 2 is = 1x2
[1.0, 1.0]

The outputs of layer 2 are = 3x2
[-2.895, 4.978]
[-0.07899999999999996, -3.9909999999999997]
[7.704, 0.516]

*/
