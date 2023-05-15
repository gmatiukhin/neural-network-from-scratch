struct Neuron<const N: usize> {
    inputs: [f64; N],
    weights: [f64; N],
    bias: f64,
}

/// Activation function
/// Rectified Linear Unit
fn activate(val: f64) -> f64 {
    val.max(0f64)
}

impl<const N: usize> Neuron<N> {
    fn activate(&mut self) -> f64 {
        activate(
            self.inputs
                .into_iter()
                .zip(self.weights)
                .map(|(i, w)| i * w)
                .fold(0f64, |acc, x| acc + x)
                + self.bias,
        )
    }
}

pub struct ModelBuilder {
    hidden_layers: Option<Vec<Vec<Neuron>>>,
    output_layer: Option<Vec<Neuron>>,
}

impl ModelBuilder {
    fn new() {
        Self {
            hidden_layers: None,
            output_layer: None,
        }
    }
    fn hidden(mut self, size: usize) -> Self {
        match self.hidden_layers {
            Some(_) => todo!(),
            None => todo!(),
        }
    }
}

struct Model {}
