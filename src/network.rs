use std::sync::Arc;

use crate::{
    math::{ActivationFunction, Sigmoid},
    training::NetworkGradients,
};
use log;
use rand::Rng;

#[derive(Debug)]
pub struct NetworkBuilder {
    input_layer_size: usize,
    hidden_layer_sizes: Vec<usize>,
    activation_function: ActivationFunction,
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self {
            input_layer_size: 1,
            hidden_layer_sizes: vec![],
            activation_function: Arc::new(Sigmoid),
        }
    }
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(mut self, size: usize) -> Self {
        self.input_layer_size = size;
        log::debug!("Input size set to {size}");
        self
    }

    pub fn hidden(mut self, size: usize) -> Self {
        if self.hidden_layer_sizes.is_empty() {
            self.hidden_layer_sizes = vec![size];
        } else {
            self.hidden_layer_sizes.push(size);
        }
        log::debug!(
            "Added a hidden layer to the network. Now it has {} hidden layers, latest is size {size}",
            self.hidden_layer_sizes.len()
        );
        self
    }

    pub fn activation(mut self, activation_function: ActivationFunction) -> Self {
        self.activation_function = activation_function;
        log::debug!("Set activation function to {:?}", self.activation_function);
        self
    }

    pub fn finalize<const I: usize, const H: usize, const O: usize>(
        mut self,
    ) -> Network<I, { H + 1 }, O> {
        self.hidden_layer_sizes.push(O);
        log::debug!("Added the output layer with size: {O}");

        let mut layers: Vec<Vec<Neuron>> = vec![vec![]; self.hidden_layer_sizes.len()];
        let mut prev_layer_size = I;

        layers
            .iter_mut()
            .zip(self.hidden_layer_sizes)
            .enumerate()
            .for_each(|(i, (a, b))| {
                log::debug!("Layer {i} being filled with {b} neurons with prev_layer_size: {prev_layer_size}");
                for _ in 0..b {
                    a.push(Neuron::new(prev_layer_size, &self.activation_function));
                }
                prev_layer_size = b;
            });
        log::debug!("Hidden layers:\n{layers:#?}");
        Network {
            hidden_layers: layers.try_into().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
    pub(crate) activation_function: ActivationFunction,
}

impl Neuron {
    fn new(num_inputs: usize, activation_function: &ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = vec![0f64; num_inputs];
        rng.fill(weights.as_mut_slice());
        Self {
            weights,
            bias: rng.gen(),
            activation_function: activation_function.clone(),
        }
    }

    fn activate(&self, inputs: &[f64]) -> (f64, NeuronData) {
        let weighted_input_sum = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| i * w)
            .fold(self.bias, |acc, x| acc + x);

        let output = self.activation_function.activate(weighted_input_sum);

        let data = NeuronData {
            weighted_input_sum,
            output,
            inputs: inputs.to_vec(),
            weights: self.weights.clone(),
            bias: self.bias,
            activation_function: self.activation_function.clone(),
        };

        (output, data)
    }
}

#[derive(Debug, Clone)]
pub struct NeuronData {
    pub inputs: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub output: f64,
    pub weighted_input_sum: f64,
    pub activation_function: ActivationFunction,
}

pub struct NetworkResult<const O: usize> {
    pub output: [f64; O],
    pub layer_data: Vec<Vec<NeuronData>>,
}

#[derive(Debug, Clone)]
pub struct Network<const I: usize, const H: usize, const O: usize> {
    pub(crate) hidden_layers: [Vec<Neuron>; H],
}

impl<const I: usize, const H: usize, const O: usize> Network<I, H, O> {
    pub fn compute(&self, inputs: [f64; I]) -> NetworkResult<O> {
        let mut layer_result = inputs.to_vec();
        let mut layer_data = vec![];
        for layer in &self.hidden_layers {
            let comp_res = Self::compute_layer(&layer_result, layer);
            layer_result = comp_res.0;
            layer_data.push(comp_res.1);
        }

        NetworkResult {
            output: layer_result.try_into().unwrap(),
            layer_data,
        }
    }

    fn compute_layer(inputs: &[f64], layer: &[Neuron]) -> (Vec<f64>, Vec<NeuronData>) {
        layer.iter().map(|n| n.activate(inputs)).unzip()
    }

    pub(crate) fn apply_gradients(&mut self, gradients: &NetworkGradients, learn_rate: f64) {
        self.hidden_layers
            .iter_mut()
            .flatten()
            .zip(gradients.values.iter().flatten())
            .for_each(|(neuron, gradient)| {
                neuron
                    .weights
                    .iter_mut()
                    .zip(&gradient.weights)
                    .for_each(|(nw, gw)| {
                        *nw += gw * learn_rate;
                    });
                neuron.bias += gradient.bias * learn_rate;
            });
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn do_log() {
        std::env::set_var("RUST_LOG", "debug");
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn make_network() {
        do_log();
        let net = NetworkBuilder::new()
            .input(1)
            .hidden(1)
            .finalize::<1, 1, 1>();
        assert!(net.compute([1f64]).output[0] >= 0f64);
    }

    #[test]
    fn make_bigger_network() {
        do_log();
        let net = NetworkBuilder::new()
            .input(2)
            .hidden(2)
            .finalize::<2, 1, 2>();
        assert!(net.compute([1f64, 3.5]).output[0] >= 0f64);
    }
}
