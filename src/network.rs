use crate::math;
use log;
use rand::Rng;

#[derive(Debug, Clone)]
pub(crate) struct Neuron {
    pub(crate) inputs: Option<Vec<f64>>,
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
    pub(crate) output: Option<f64>,
    pub(crate) weighted_input_sum: Option<f64>,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = vec![0f64; num_inputs];
        rng.fill(weights.as_mut_slice());
        Self {
            weights,
            bias: rng.gen(),
            inputs: None,
            output: None,
            weighted_input_sum: None,
        }
    }

    fn activate(&mut self, inputs: &[f64]) -> f64 {
        let weighted_sum = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| i * w)
            .fold(0f64, |acc, x| acc + x)
            + self.bias;
        let output = math::relu(weighted_sum);

        self.weighted_input_sum = Some(weighted_sum);
        self.output = Some(output);
        self.inputs = Some(inputs.to_vec());

        output
    }
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
        Network {
            hidden_layers: layers.try_into().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Network<const I: usize, const H: usize, const O: usize> {
    hidden_layers: [Vec<Neuron>; H],
}

impl<const I: usize, const H: usize, const O: usize> Network<I, H, O> {
    pub fn compute(&mut self, inputs: [f64; I]) -> [f64; O] {
        let mut layer_result = inputs.to_vec();
        for layer in &mut self.hidden_layers {
            layer_result = Self::compute_layer(&layer_result, layer);
        }

        math::softmax::<O>(layer_result.try_into().expect("Vec of incorrect length"))
    }

    fn compute_layer(inputs: &[f64], layer: &mut [Neuron]) -> Vec<f64> {
        let output = layer
            .iter_mut()
            .map(|n| -> f64 { n.activate(inputs) })
            .collect::<Vec<f64>>();
        output
    }

    pub(crate) fn export_data(&self) -> Vec<Vec<Neuron>> {
        self.hidden_layers.to_vec()
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
        let mut net = NetworkBuilder::new()
            .input(1)
            .hidden(1)
            .output(1)
            .finalize::<1, 1, 1>();
        assert!(net.compute([1f64])[0] >= 0f64);
    }
}
