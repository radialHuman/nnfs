// https://www.youtube.com/watch?v=TEWy9vZcxW4

/*
Batches and OOPS NEEDS WORK
*/
use math::round;
use rand::{rngs::StdRng, Rng, SeedableRng};

// to create weights using random values than typying it in
struct LayerDetails {
    n_inputs: i32,
    n_neurons: i32,
}
impl LayerDetails {
    pub fn create_weights(&self) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut weight: Vec<Vec<f64>> = vec![];
        // this gives transposed weights
        for i in 0..self.n_inputs {
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
        let mat_mul = matrix_product(&input, &weights);
        let mut output: Vec<Vec<f64>> = vec![];
        for i in mat_mul {
            output.push(vector_addition(&i, &bias));
        }
        output
    }
}

pub fn four() {
    // OOPS
    let l1 = LayerDetails {
        n_inputs: 3,  // number of features/columns in X
        n_neurons: 5, // can be anything depending the situation
    };
    // let generated_weights1 = l1.create_weights();
    // println!("The weights of layer 1 are = ");
    // print_a_matrix(&generated_weights1);
    // let generated_bias1 = l1.create_bias();
    // println!("The bias of layer 1 is = {:?}", &generated_bias1);
    let generated_output1 = l1.output_of_layer(&input, &l1.create_weights(), &l1.create_bias());
    println!("The output of layer 1 are = ");
    print_a_matrix(&l1.output_of_layer(&input, &l1.create_weights(), &l1.create_bias()));

    // layer 2
    let l2 = LayerDetails {
        n_inputs: 5,  // has to be that of the previous layer
        n_neurons: 2, // can be anything depending the situation
    };
    println!("The output of layer 2 are = ");
    print_a_matrix(&l2.output_of_layer(
        &generated_output1,
        &l1.create_weights(),
        &l1.create_bias(),
    ));
}
