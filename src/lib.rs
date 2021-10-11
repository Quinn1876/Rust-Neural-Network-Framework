/**
 * Source Material
 * https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
 */

pub mod activation_functions {
    pub fn sigmoid(x: f32) -> f32 {
        1f32 / (1f32 + (-x).exp())
    }

    pub fn sigmoid_derivative(x: f32) -> f32 {
        sigmoid(x) * (1f32-sigmoid(x))
    }
}

pub mod utils {
    pub trait InputAttributes {
        fn to_vec<'s>(&'s self) -> Vec<&'s f32>;
        fn num_attributes() -> usize;
    }

    pub trait OutputAttributes {
        fn to_vec<'s>(&'s self) -> Vec<&'s f32>;
    }

    pub trait PerceptronOutput {
        fn get_output(&self) -> f32;
        fn new(output: f32) -> Self;
    }
}

pub mod perceptron {
    use std::marker::PhantomData;
    use rand::prelude::*;
    use rand::distributions::Standard;

    use crate::utils;

    pub struct Perceptron<Inputs: utils::InputAttributes, Output: utils::PerceptronOutput> {
        weights: Vec<f32>,
        bias_term: f32,
        learning_rate: f32,
        input_phantom: PhantomData<Inputs>,
        output_phantom: PhantomData<Output>
    }

    impl<Inputs: utils::InputAttributes, Output: utils::PerceptronOutput> Perceptron<Inputs, Output>{
        pub fn new(learning_rate: f32) -> Perceptron<Inputs, Output> {
            let bias_term: f32 = StdRng::from_entropy().sample(Standard);
            let mut weights: Vec<f32> = Vec::with_capacity(Inputs::num_attributes());

            for _ in 0..Inputs::num_attributes() {
                weights.push(StdRng::from_entropy().sample(Standard));
            }
            Perceptron {
                bias_term,
                weights,
                learning_rate,
                input_phantom: PhantomData,
                output_phantom: PhantomData,
            }
        }

        pub fn compute_dot_product(&self, inputs: &Inputs) -> f32 {
            inputs
                .to_vec()
                .iter()
                .zip(self.weights.iter())
                .fold(self.bias_term, |acc, (&&input, &weight)| acc + (input*weight))
        }

        pub fn update_weights(&mut self, derivative_of_errors_wrt_weights: Vec<f32>) {
            for (weight, &error_term) in self.weights.iter_mut().zip(derivative_of_errors_wrt_weights.iter()) {
                *weight = *weight - self.learning_rate*error_term;
            }
        }

        pub fn update_bias(&mut self, partial_error: f32) {
            self.bias_term = self.bias_term - self.learning_rate*partial_error;
        }

        pub fn compute_output(&self, inputs: &Inputs) -> Output {
            Output::new(super::activation_functions::sigmoid(self.compute_dot_product(inputs)))
        }
    }
}

pub mod neural_network {
    use rand::distributions::{Distribution, Uniform};
    use std::marker::PhantomData;
    use crate::utils;

    pub struct NeuralNetwork<Inputs: utils::InputAttributes, Outputs: utils::OutputAttributes> {
        hidden_layer: Vec<hidden_node::HiddenNode>,
        learning_rate: f32,
        input_phantom: PhantomData<Inputs>,
        output_phantom: PhantomData<Outputs>
    }

    #[derive(Debug)]
    pub enum Error {
        InvalidTrainingData,
        RNGError,
        HiddenNodeError(hidden_node::Error)
    }

    impl<Inputs: utils::InputAttributes, Outputs: utils::OutputAttributes> NeuralNetwork<Inputs, Outputs> {
        pub fn new(num_hidden_nodes: usize, learning_rate: f32) -> NeuralNetwork<Inputs, Outputs> {
            let mut hidden_layer: Vec<hidden_node::HiddenNode> = Vec::with_capacity(num_hidden_nodes);
            for _ in 0..num_hidden_nodes {
                hidden_layer.push(hidden_node::HiddenNode::new(Inputs::num_attributes()));
            }
            NeuralNetwork {
                hidden_layer,
                learning_rate,
                input_phantom: PhantomData,
                output_phantom: PhantomData
            }
        }

        /**
         * The input_set should have the same length as the output_set
         * epochs is the number of training runs we will do on the model.
         * Each epoch, a set or inputs and corresponding outputs will be used to train the model.
         */
        pub fn train(&mut self, epochs: usize, input_set: Vec<Inputs>, output_set: Vec<Outputs>) -> Result<(), Error> {
            // Validate Values
            if input_set.len() != output_set.len() { return Err(Error::InvalidTrainingData); }

            let distribution = Uniform::from(0..input_set.len());
            let mut generator = rand::thread_rng();

            for epoch in 0..epochs {
                let set = distribution.sample(&mut generator);
                let input = input_set.get(set).ok_or(Error::RNGError)?;
                let output = output_set.get(set).ok_or(Error::RNGError)?;
                // let mut results: Vec<f32> = Vec::with_capacity(self.hidden_layer.len());

                // for node in &self.hidden_layer {
                //     let result = node.compute_output(input.to_vec()).map_err(|e| Error::HiddenNodeError(e))?;
                //     results.push(result);
                // }
                let results: Vec<f32> = self.test_input(input)?;

                let errors = results
                    .iter()
                    .zip(output.to_vec().into_iter())
                    .map(|(result, expected)| result - expected)
                    .collect::<Vec<f32>>();

                // println!("Error: {}", errors.get(0).expect("First element should exist"));

                let dcost = errors.iter();
                let dpred = results.iter().map(|result| crate::activation_functions::sigmoid_derivative(*result));
                let z_del = dcost
                    .zip(dpred)
                    .map(|(cost,pred)| cost * pred);

                let learning_rate = self.learning_rate;
                for (node, product) in self.hidden_layer
                    .iter_mut()
                    .zip(
                        z_del
                            .zip(input.to_vec().iter())
                            .map(|(z_del, &input)| input * z_del * learning_rate)

                    ) {
                        node.update_weights(product);
                    }
            }

            Ok(())
        }

        pub fn test_input(&self, input: &Inputs) -> Result<Vec<f32>, Error> {
            Ok(self
                .hidden_layer
                .iter()
                .map(|node| node.compute_output(input.to_vec()).map_err(|e| Error::HiddenNodeError(e)))
                .collect::<Result<Vec<f32>, Error>>()?)
        }
    }

    pub mod hidden_node {
        use rand::distributions::{Distribution, Uniform};
        use crate::activation_functions;

        #[derive(Debug)]
        pub enum Error {
            InconsistentInputSize{num_weights: usize, num_values: usize},
            MissingInput(usize)
        }

        pub struct HiddenNode {
            input_weights: Vec<f32>,
            bias_term: f32,
        }

        impl HiddenNode {
            pub fn new(num_inputs: usize) -> HiddenNode {
                let mut input_weights: Vec<f32> = Vec::with_capacity(num_inputs);
                let weight_distribution = Uniform::from(-100f32..100f32);
                let mut generator = rand::thread_rng();
                for _ in 0..num_inputs {
                    input_weights.push(weight_distribution.sample(&mut generator));
                }
                let bias_term = weight_distribution.sample(&mut generator);
                HiddenNode {
                    input_weights,
                    bias_term
                }
            }

            pub fn from_weights_and_bias(input_weights: Vec<f32>, bias_term: f32) -> HiddenNode {
                HiddenNode {
                    input_weights,
                    bias_term
                }
            }

            fn compute_dot_product(&self, input_values: Vec<&f32>) -> Result<f32, Error> {
                let mut sum = self.bias_term;
                if input_values.len() != self.input_weights.len() {
                    return Err(Error::InconsistentInputSize{num_weights: self.input_weights.len(), num_values: input_values.len()});
                }
                for (index, &weight) in self.input_weights.iter().enumerate() {
                    sum += weight*(*input_values.get(index).ok_or(Error::MissingInput(index))?);
                }
                Ok(sum)
            }

            pub fn compute_output(&self, input_values: Vec<&f32>) -> Result<f32, Error> {
                let output = activation_functions::sigmoid(self.compute_dot_product(input_values)?);
                Ok(output)
            }

            pub fn update_weights(&mut self, product: f32) {
                for weight in self.input_weights.iter_mut() {
                    *weight -= product;
                }
            }
        }
    }
}
