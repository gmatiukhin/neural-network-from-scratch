use crate::{math, training::NetworkGradients};
use log;
use rand::Rng;

#[derive(Debug, Clone)]
pub(crate) struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = vec![0f64; num_inputs];
        rng.fill(weights.as_mut_slice());
        Self {
            weights,
            bias: rng.gen(),
        }
    }

    fn activate(&self, inputs: &[f64]) -> (f64, NeuronData) {
        let weighted_input_sum = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| i * w)
            .fold(0f64, |acc, x| acc + x)
            + self.bias;
        let output = math::sigmoid(weighted_input_sum);

        let data = NeuronData {
            weighted_input_sum,
            output,
            inputs: inputs.to_vec(),
            weights: self.weights.clone(),
            bias: self.bias,
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
}

pub struct NetworkBuilder {
    input_layer_size: usize,
    hidden_layer_sizes: Vec<usize>,
    output_layer_size: usize,
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self {
            input_layer_size: 1,
            hidden_layer_sizes: vec![],
            output_layer_size: 1,
        }
    }
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(mut self, size: usize) -> Self {
        log::debug!("Input size set to {size}");
        self.input_layer_size = size;
        self
    }

    pub fn hidden(mut self, size: usize) -> Self {
        if self.hidden_layer_sizes.is_empty() {
            self.hidden_layer_sizes = vec![size];
            log::debug!("Hidden layers now has single layer of size {size}");
        } else {
            self.hidden_layer_sizes.push(size);
            log::debug!(
                "Now has {} hidden layers, latest is size {size}",
                self.hidden_layer_sizes.len()
            );
        }
        self
    }

    pub fn output(mut self, size: usize) -> Self {
        log::debug!("Input size set to {size}");
        self.output_layer_size = size;
        self
    }

    pub fn finalize<const I: usize, const H: usize, const O: usize>(self) -> Network<I, H, O> {
        let mut layers: Vec<Vec<Neuron>> = vec![vec![]; H];
        let mut prev_layer_size = I;
        layers
            .iter_mut()
            .zip(self.hidden_layer_sizes)
            .for_each(|(a, b)| {
                log::debug!("Layer {a:?} being filled with {b} neurons with prev_layer_size: {prev_layer_size}");
                for _ in 0..b {
                    a.push(Neuron::new(prev_layer_size));
                }
                prev_layer_size = b;
            });
        log::debug!("hidden layers: {:?}", layers);
        Network {
            hidden_layers: layers.try_into().unwrap(),
        }
    }
}

#[derive(Debug)]
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
        env_logger::init();
    }

    #[test]
    fn make_network() {
        do_log();
        let net = NetworkBuilder::new()
            .input(1)
            .hidden(1)
            .output(1)
            .finalize::<1, 1, 1>();
        assert!(net.compute([1f64]).output[0] >= 0f64);
    }

    #[test]
    fn make_bigger_network() {
        do_log();
        let net = NetworkBuilder::new()
            .input(2)
            .hidden(3)
            .hidden(2)
            .output(2)
            .finalize::<2, 2, 2>();
        assert!(net.compute([1f64, 3.5]).output[0] >= 0f64);
    }
}
