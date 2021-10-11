
use neural_network::utils::{ InputAttributes, OutputAttributes, PerceptronOutput };
use neural_network::neural_network::NeuralNetwork;
use neural_network::perceptron::Perceptron;
use neural_network::activation_functions::*;

struct Input {
    x1: f32,
    x2: f32,
    x3: f32,
}

impl InputAttributes for Input {
    fn to_vec<'input>(&'input self) -> Vec<&'input f32> {
        vec![&self.x1, &self.x2, &self.x3]
    }

    fn num_attributes() -> usize {
        3
    }
}

impl Input {
    pub fn new(x1: f32, x2: f32, x3: f32) -> Input {
        Input {
            x1,
            x2,
            x3,
        }
    }
}

struct Output {
    y0: f32
}

impl OutputAttributes for Output {
    fn to_vec<'s>(&'s self) -> Vec<&'s f32> {
        vec![&self.y0]
    }
}

impl PerceptronOutput for Output {
    fn new(y0: f32) -> Output {
        Output {
            y0
        }
    }

    fn get_output(&self) -> f32 {
        self.y0
    }
}

fn main() {

    // Read input file
    // let mut nn: NeuralNetwork<Input, Output> = NeuralNetwork::new(1, 0.05);

    let inputs = vec![
        Input::new(0f32,1f32,0f32),
        Input::new(0f32, 0f32, 1f32),
        Input::new(1f32, 0f32, 0f32),
        Input::new(1f32, 1f32, 0f32),
        Input::new(1f32, 1f32, 1f32),
        Input::new(0f32, 1f32, 1f32),
        Input::new(0f32, 1f32, 0f32)
    ];
    let outputs = vec![
        Output::new(1f32),
        Output::new(0f32),
        Output::new(0f32),
        Output::new(1f32),
        Output::new(1f32),
        Output::new(0f32),
        Output::new(1f32),
    ];
    // nn.train(500000, inputs, outputs).expect("Error Occurred while training");

    // let output = nn.test_input(&Input::new(0f32,1f32,0f32)).expect("Value to be valid")[0];

    // println!("Expected Result: 1, actual result: {}", output);

    let mut p: Perceptron<Input, Output> = Perceptron::new(0.05);

    for _ in 0..25000 {
        let XWs = inputs.iter().map(|input| p.compute_dot_product(input));
        let z: Vec<f32> = XWs.map(|XW| sigmoid(XW)).collect();
        let error = z.iter().zip(outputs.iter()).map(|(z, label)| z - label.to_vec()[0]);
        let dcost = error;
        let dpred = z.iter().map(|&v| sigmoid_derivative(v));
        let z_del = dcost.zip(dpred).map(|(cost, pred)| cost*pred);

        for (partial_error, input) in z_del.zip(inputs.iter()) {
            p.update_weights(input.to_vec().iter().map(|&&input| input*partial_error).collect());
            p.update_bias(partial_error);
        }
    }

    for i in 0..inputs.len() {
        println!("Expected Output: {}", outputs[i].get_output());
        println!("Received Output: {}", p.compute_output(&inputs[i]).get_output());
    }
}
