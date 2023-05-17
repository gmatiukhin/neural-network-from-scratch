#[derive(Debug, Clone, Copy)]
struct Node<const N: usize> {
    weights: [f64; N],
    bias: f64,
}

impl<const N: usize> Default for Node<N> {
    fn default() -> Self {
        Self {
            weights: [0f64; N],
            bias: 0f64,
        }
    }
}

/// Activation function
/// Rectified Linear Unit
fn activate(val: f64) -> f64 {
    val.max(0f64)
}

pub enum NeuronError {
    InputsMismatch,
}

trait Neuron {
    fn activate(&mut self, inputs: &[f64]) -> Result<f64, NeuronError>;
}

impl<const N: usize> Neuron for Node<N> {
    fn activate(&mut self, inputs: &[f64]) -> Result<f64, NeuronError> {
        if inputs.len() != self.weights.len() {
            return Err(NeuronError::InputsMismatch);
        }

        let val = activate(
            inputs
                .into_iter()
                .zip(self.weights)
                .map(|(i, w)| i * w)
                .fold(0f64, |acc, x| acc + x)
                + self.bias,
        );

        Ok(val)
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
        self.input_layer_size = size;
        self
    }

    pub fn hidden(mut self, size: usize) -> Self {
        if self.hidden_layer_sizes.is_empty() {
            self.hidden_layer_sizes = vec![size];
        } else {
            self.hidden_layer_sizes.push(size);
        }
        self
    }

    pub fn output(mut self, size: usize) -> Self {
        self.output_layer_size = size;
        self
    }

    pub fn finalize<const I: usize, const H: usize, const O: usize>(self) -> Network<I, H, O> {
        let mut layers: Vec<Vec<Box<dyn Neuron>>> = vec![vec![]; self.hidden_layer_sizes.len()];
        let mut prev_layer_size = 0;
        layers
            .iter_mut()
            .zip(self.hidden_layer_sizes)
            .map(|(a, b)| {
                for _ in 0..b {
                    a.push(Box::<Node<{ prev_layer_size }>>::new(Node::default()));
                }
                prev_layer_size = b;
            });
        todo!()
    }
}

pub struct Network<const I: usize, const H: usize, const O: usize> {
    hidden_layers: [Vec<Box<dyn Neuron>>; H],
    output_layer: [Box<dyn Neuron>; O],
}

fn softmax<const T: usize>(data: [f64; T]) -> [f64; T] {
    let exponents = data.into_iter().map(|el| el.exp());
    let sum = exponents.clone().fold(0f64, |acc, el| acc + el);
    exponents
        .map(|el| el / sum)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Arrays length do not match")
}

impl<const I: usize, const H: usize, const O: usize> Network<I, H, O> {
    pub fn compute(&mut self, inputs: [f64; I]) -> Result<[f64; O], NeuronError> {
        let mut inputs = inputs.to_vec();
        for layer in &mut self.hidden_layers {
            inputs = Self::compute_layer(&inputs, layer)?;
        }
        let outputs = Self::compute_layer(&inputs, &mut self.output_layer)?;

        let output_normalized = softmax(outputs.try_into().expect("Vec of incorrect length"));
        Ok(output_normalized)
    }

    fn compute_layer(
        inputs: &[f64],
        layer: &mut [Box<dyn Neuron>],
    ) -> Result<Vec<f64>, NeuronError> {
        let output = layer
            .iter_mut()
            .map(|n| -> Result<f64, NeuronError> { Ok(n.activate(inputs)?) })
            .collect::<Result<Vec<f64>, NeuronError>>()?;
        Ok(output)
    }
}
