use crate::{
    math,
    network::{Network, Neuron},
};

pub trait TrainableNetwork<const I: usize, const H: usize, const O: usize> {
    fn node_loss(got: f64, expected: f64) -> f64;
    fn node_loss_derivative(got: f64, expected: f64) -> f64;
    fn network_loss(&self, network_output: [f64; O], expected_output: [f64; O]) -> f64;
    fn update_gradients(&mut self, train_data: &TrainData<I, O>);
    fn train(&mut self, train_data: &TrainData<I, O>, _learn_rate: f64, err_margin: f64);
}

impl<const I: usize, const H: usize, const O: usize> TrainableNetwork<I, H, O>
    for Network<I, H, O>
{
    fn node_loss(got: f64, expected: f64) -> f64 {
        let err = got - expected;
        err * err
    }

    fn node_loss_derivative(got: f64, expected: f64) -> f64 {
        2f64 * (got - expected)
    }

    // NOTE: maybe declare self.compute in another trait
    // that way it is possible to make this a default implementation
    fn network_loss(&self, network_output: [f64; O], expected_output: [f64; O]) -> f64 {
        network_output
            .iter()
            .zip(expected_output)
            .fold(0f64, |acc, (&got, expected)| {
                acc + Self::node_loss(got, expected)
            })
    }

    fn update_gradients(&mut self, train_data: &TrainData<I, O>) {
        let _ = self.compute(train_data.input);

        let layer_data = self.export_data();
        let mut prev_layer_node_der_values = vec![];
        let mut gradients: Vec<Vec<f64>> = vec![];
        let mut prev_layer: Vec<Neuron> = vec![];

        // Backpropagation
        layer_data.iter().rev().enumerate().for_each(|(i, layer)| {
            let mut layer_gradients = vec![];
            layer.iter().enumerate().for_each(|(j, neuron)| {
                let neuron_der = if i == 0 {
                    // Output layer
                    // For every neuron in the output layer:
                    // multiply the derivative of its activation function by
                    // the derivative of the loss function
                    math::d_relu(neuron.weighted_input_sum.unwrap())
                        * Self::node_loss_derivative(
                            neuron.output.unwrap(),
                            train_data.expected_output[j],
                        )
                } else {
                    // Hidden layers
                    // For every neuron in the hidden layer
                    // multiply the derivative of its activation function by
                    // sum of the weights of the next layer neurons
                    // multiplied by their respective derivative values
                    math::d_relu(neuron.weighted_input_sum.unwrap())
                        * prev_layer
                            .iter()
                            .enumerate()
                            .flat_map(|(k, node)| {
                                node.weights
                                    .iter()
                                    .map(|weight| weight * gradients.last().unwrap()[k])
                                    .collect::<Vec<_>>()
                            })
                            .sum::<f64>()
                };
                layer_gradients.push(neuron_der);
                prev_layer_node_der_values.push(neuron_der);
            });
            prev_layer = layer.clone();
            gradients.push(layer_gradients);
        });
    }

    fn train(&mut self, train_data: &TrainData<I, O>, _learn_rate: f64, err_margin: f64) {
        let computation_result = self.compute(train_data.input);
        let mut prev_loss = self.network_loss(computation_result, train_data.expected_output);
        loop {
            let computation_result = self.compute(train_data.input);
            let loss = self.network_loss(computation_result, train_data.expected_output);
            if (loss - prev_loss).abs() <= err_margin {
                break;
            }
            prev_loss = loss;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainData<const I: usize, const O: usize> {
    pub input: [f64; I],
    pub expected_output: [f64; O],
}
