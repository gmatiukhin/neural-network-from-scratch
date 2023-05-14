struct Neuron<const N: usize> {
    inputs: [f64; N],
    weights: [f64; N],
    bias: f64,
}

/// Activation function
fn activate(val: f64) -> f64 {
    if val < 0f64 {
        0f64
    } else {
        val
    }
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
