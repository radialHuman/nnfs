// https://www.youtube.com/watch?v=gmjzbpSVY1A&t=5s

/*
Activation function : relu, step and sigmoid (for hidden layers)
Data read in manually, generated using python code from video, create_data function incomplete
*/
use math::round;
use ndarray::Array;
use ndarray_rand::rand_distr::{Distribution, Normal};
use rand::*;
use std::fs;
use std::io;

// from four::four_oopsstruct
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
        let bias = vec![0.; self.n_neurons as usize];
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
        println!("Before activation it was {:?}", &output[0]);
        println!("After activation it was {:?}", activation_relu(&output[0]));
        transpose(&output)
    }
}

pub fn five() {
    /*ACTIVATION FUNCTION
     * Aim: If there are no AF, its y = x (linear function), this will not helpw ith non-linear relations
     */

    /* STEP FUNCTIONS
     * If the input is more than 0 then output is 1 else 0
     * Input here is the input*weight+bias
     * Output is either 1 or 0
     * Not useful in output layer
     * Not really useful in nn
     */

    /* SIGMOID FUNCTIONS
     * y = 1/(1+e^(-x))
     * Gives granular output
     * Useful for optimizing by reducing loss
     * Cant avoid vanishing gradient problem
     */

    /* RECTIFIED LINEAR UNIT
     * If its less than 0 its 0 else what ever x is
     * Its fast and no calculation involved
     * It works because, clipping it at 0 makes it good for non-linear fucntions
     * Weight moves line away from x axis
     * Bias offsets line along -x axis if positive
     * Decides when the neuron activates and when it does not
     * The combination of weights and baisa using this function can fit non-linear functions
     * More the neurons the better the fit
     */

    let input1 = vec![0., 2., -1., 3.3, -2.7, 1.1, 2.2, -100.];
    let output = activation_relu(&input1);
    println!("The output from relu for {:?} is {:?}", &input1, &output);

    // read in X
    let X_string = read_file("./src/spiral_data_X.txt").unwrap();
    let X_inter = &X_string
        .replace("[", "")
        .replace("]", "")
        .replace("\n", "")
        .replace(" ", "");
    let X_vector: Vec<_> = X_inter.split(",").collect();
    // println!("Input is {:?}", X_vector);
    let X_parsed: Vec<f64> = X_vector.iter().map(|x| x.parse().unwrap()).collect();
    let features = shape_changer(&X_parsed, 2, X_parsed.len() / 2);
    // println!("Input is {:?}", features);

    // read in y
    let y_string = read_file("./src/spiral_data_y.txt").unwrap();
    let y_inter = &y_string.replace("[", "").replace("]", "").replace(" ", "");
    let y_vector: Vec<_> = y_inter.split(",").collect();
    // println!("Input is {:?}", X_vector);
    let target: Vec<f64> = y_vector.iter().map(|x| x.parse().unwrap()).collect();
    // println!("Input is {:?}", target);

    // First layer
    let X = vec![
        vec![1., 2., 3., 2.5],
        vec![2., 5., -1., 2.],
        vec![-1.5, 2.7, 3.3, -0.8],
    ]; // input in batches
       // OOPS
    let l1 = LayerDetails {
        n_inputs: features[0].len(), // number of features/columns in X
        n_neurons: 5,                // can be anything depending the situation
    };
    println!("The input of layer 1 are = {}x{}", X.len(), X[0].len());
    print_a_matrix("The matrix is:", &X);
    let generated_weights1 = l1.create_weights();
    println!(
        "The weights of layer 1 are = {}x{}",
        generated_weights1.len(),
        generated_weights1[0].len()
    );
    print_a_matrix("Generated weights are:", &generated_weights1);
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
    print_a_matrix("Output before activation is:", &generated_output1);

    // output through activation
    let mut activated_output_1 = vec![];
    for i in generated_output1 {
        activated_output_1.push(activation_relu(&i));
    }
    print_a_matrix("Input to second layer is:", &activated_output_1);
}

pub fn read_file(path: &str) -> Result<String, io::Error> {
    fs::read_to_string(path)
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

pub fn shape_changer<T>(list: &Vec<T>, columns: usize, rows: usize) -> Vec<Vec<T>>
where
    T: std::clone::Clone,
{
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

pub fn print_a_matrix(string: &str, matrix: &Vec<Vec<f64>>) {
    println!("{}", string);
    for i in matrix.iter() {
        println!("{:?}", i);
    }
    println!("");
    println!("");
}

pub fn activation_relu(input: &Vec<f64>) -> Vec<f64> {
    input
        .iter()
        .map(|x| if *x > 0. { *x } else { 0. })
        .collect()
}

pub fn vector_addition(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut output = vec![];
    for i in 0..a.len() {
        output.push(round::ceil(a[i], 3) + round::ceil(b[i], 3));
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

// WIP
pub fn create_data(data_points: i32, classes: i32, dimension: usize) {
    // -> (Vec<f64>, Vec<u8>) {
    // https://github.com/cs231n/cs231n.github.io/blob/master/neural-networks-case-study.md
    // Lets generate a classification dataset that is not easily linearly separable. Our favorite example is the spiral dataset, which can be generated as follows:
    let mut X = vec![];
    let rows = data_points * classes;
    for _ in 0..=rows {
        X.push(vec![0.; dimension]);
    }
    let y = vec![0; rows as usize];

    for class_number in 0..classes {
        let ix = data_points * class_number..=data_points * (class_number + 1);
        let radius = Array::linspace(0., 1., data_points as usize);
        let mut np = vec![];
        let normal = Normal::new(0.0, 1.0).unwrap();
        for _ in 0..data_points {
            np.push(normal.sample(&mut rand::thread_rng()));
        }
        let normal_points: Vec<f64> = np.iter().map(|z| z * 0.2).collect();
        // println!("Normal distribution is {:?}", normal_points);
        let linspace = Array::linspace(
            class_number as f64 * 4.,
            (class_number as f64 + 1.) * 4.,
            data_points as usize,
        );
        // println!("Linspace is {:?}", linspace);
        let mut theta = vec![];
        for (n, i) in linspace.iter().enumerate() {
            theta.push(normal_points[n] + i);
        }
        // println!("Theta is {:?}", theta);
        let mut sin_value = vec![];
        let mut cos_value = vec![];
        for i in theta.iter() {
            sin_value.push((i * 2.5).sin() * &radius);
            cos_value.push((i * 2.5).cos() * &radius);
        }
        let sin_cos: Vec<_> = sin_value.iter().zip(cos_value.iter()).collect();
        // println!("Sin_Cos is {:?}", sin_cos);
        // for i in data_points * class_number..=data_points * (class_number + 1) {
        //     X[i as usize].push(sin_cos.iter().map(|x| vec![x.0, x.1]).collect());
        // }
        for (n, i) in X.iter().enumerate() {
            for (m, j) in sin_cos.iter().enumerate() {
                // X[n] = sin_cos[m] // TO BE CONTINUED
            }
        }
        // y[ix] = class_number;
    }

    //   t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    //   X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    //   y[ix] = j
}

/*
OUTPUT
The output from relu for [0.0, 2.0, -1.0, 3.3, -2.7, 1.1, 2.2, -100.0] is [0.0, 2.0, 0.0, 3.3, 0.0, 1.1, 2.2, 0.0]
The input of layer 1 are = 3x4
The matrix is:
[1.0, 2.0, 3.0, 2.5]
[2.0, 5.0, -1.0, 2.0]
[-1.5, 2.7, 3.3, -0.8]


The weights of layer 1 are = 2x5
Generated weights are:
[0.952, -0.968, 0.703, -0.176, 0.068]
[-0.908, -0.406, 0.066, 0.853, 0.515]


The bias of layer 1 is  = 1x5
[0.0, 0.0, 0.0, 0.0, 0.0]

Before activation it was [-0.864, 1.53, -3.966, 2.712, -0.876]
After activation it was [0.0, 1.53, 0.0, 2.712, 0.0]
The output of layer 1 are = 3x5
Output before activation is:
[-0.864, 1.53, -3.966, 2.712, -0.876]
[-1.78, 1.098, 1.736, -3.879, 2.568]
[0.835, -2.636, 3.913, 0.356, 1.289]


Input to second layer is:
[0.0, 1.53, 0.0, 2.712, 0.0]
[0.0, 1.098, 1.736, 0.0, 2.568]
[0.835, 0.0, 3.913, 0.356, 1.289]
*/
