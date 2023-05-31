use log::debug;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use crate::{
    math,
    network::{Network, NeuronData},
};

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const I: usize, const O: usize> {
    pub input: [f64; I],
    pub expected_output: [f64; O],
}

fn update_gradients<const I: usize, const H: usize, const O: usize>(
    network_data: Vec<Vec<NeuronData>>,
    train_data: DataPoint<I, O>,
    gradients: Arc<Mutex<NetworkGradients>>,
) {
    let mut common_gradient_parts: Vec<Vec<f64>> = vec![];
    let mut prev_layer: Vec<NeuronData> = vec![];

    // Backpropagation
    for (i, layer) in network_data.iter().rev().enumerate() {
        let mut grad_parts = vec![];
        for (j, neuron) in layer.iter().enumerate() {
            let value = if i == 0 {
                // Output layer
                // For every neuron in the output layer:
                // multiply the derivative of its activation function by
                // the derivative of the loss function
                math::d_sigmoid(neuron.weighted_input_sum)
                    * node_loss_derivative(neuron.output, train_data.expected_output[j])
            } else {
                // Hidden layers
                // For every neuron in the hidden layer
                // multiply the derivative of its activation function by the sum of
                // the weights of the prevuous layer neurons
                // multiplied by their respective derivative values
                math::d_sigmoid(neuron.weighted_input_sum)
                    * prev_layer
                        .iter()
                        .zip(common_gradient_parts.last().unwrap())
                        .map(|(node, der)| node.weights[j] * der)
                        .sum::<f64>()
            };

            grad_parts.push(-value);
        }
        prev_layer = layer.to_vec();
        common_gradient_parts.push(grad_parts);
    }

    // log::debug!("partial={common_gradient_parts:?}");

    // {
    //     let mut gradients = gradients.lock().unwrap();
    //     for ((layer_gradients, layer_partial_gradient), layer_data) in gradients
    //         .values
    //         .iter_mut()
    //         .zip(common_gradient_parts.iter().rev())
    //         .zip(network_data)
    //     {
    //         for ((neuron_gradient, neuron_partial_gradient), neuron_data) in (*layer_gradients)
    //             .iter_mut()
    //             .zip(layer_partial_gradient)
    //             .zip(layer_data)
    //         {
    //             for (weight, input) in neuron_gradient.weights.iter_mut().zip(neuron_data.inputs) {
    //                 *weight += neuron_partial_gradient * input;
    //             }
    //             neuron_gradient.bias += neuron_partial_gradient;
    //         }
    //     }
    //
    //     // log::debug!("gradients=\n{:?}", gradients);
    // }

    {
        let mut gradients = gradients.lock().unwrap();
        gradients
            .values
            .iter_mut()
            .zip(common_gradient_parts.iter().rev())
            .zip(network_data)
            .for_each(|((layer_gradients, layer_partial_gradient), layer_data)| {
                (*layer_gradients)
                    .iter_mut()
                    .zip(layer_partial_gradient)
                    .zip(layer_data)
                    .for_each(
                        |((neuron_gradient, neuron_partial_gradient), neuron_data)| {
                            neuron_gradient
                                .weights
                                .iter_mut()
                                .zip(neuron_data.inputs)
                                .for_each(|(weight, input)| {
                                    *weight += neuron_partial_gradient * input;
                                });
                            neuron_gradient.bias += neuron_partial_gradient;
                        },
                    );
            });

        // log::debug!("gradients=\n{:?}", gradients);
    }
}

#[derive(Debug)]
pub(crate) struct Gradient {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}

#[derive(Debug)]
pub(crate) struct NetworkGradients {
    pub(crate) values: Vec<Vec<Gradient>>,
}

impl NetworkGradients {
    fn new<const I: usize, const H: usize, const O: usize>(network: &Network<I, H, O>) -> Self {
        let mut values = vec![];

        for l in &network.hidden_layers {
            let mut layer = vec![];
            for n in l {
                layer.push(Gradient {
                    weights: vec![0f64; n.weights.len()],
                    bias: 0f64,
                });
            }
            values.push(layer);
        }

        Self { values }
    }
}

pub fn train<const I: usize, const H: usize, const O: usize>(
    network: &mut Network<I, H, O>,
    train_data: Vec<DataPoint<I, O>>,
    learn_rate: f64,
) {
    let gradients = Arc::new(Mutex::new(NetworkGradients::new(network)));
    train_data.par_iter().for_each(|data| {
        let data = data.to_owned();
        let g = gradients.clone();
        let res = network.compute(data.input);
        update_gradients::<I, H, O>(res.layer_data, data, g);
    });

    network.apply_gradients(&gradients.lock().unwrap(), learn_rate);
}

fn node_loss(got: f64, expected: f64) -> f64 {
    let err = got - expected;
    err * err
}

fn node_loss_derivative(got: f64, expected: f64) -> f64 {
    2f64 * (got - expected)
}

fn network_loss<const O: usize>(network_output: [f64; O], expected_output: [f64; O]) -> f64 {
    network_output
        .iter()
        .zip(expected_output)
        .fold(0f64, |acc, (&got, expected)| acc + node_loss(got, expected))
}

pub fn avg_network_loss<const I: usize, const H: usize, const O: usize>(
    network: &Network<I, H, O>,
    test_data: &[DataPoint<I, O>],
) -> f64 {
    test_data
        .iter()
        .map(|data_point| {
            let res = network.compute(data_point.input);
            network_loss(res.output, data_point.expected_output)
        })
        .sum::<f64>()
        / (test_data.len() as f64)
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    use crate::network::NetworkBuilder;

    use super::*;

    fn do_log() {
        std::env::set_var("RUST_LOG", "debug");
        env_logger::init();
    }

    #[test]
    fn test_learning() {
        do_log();
        let mut network = NetworkBuilder::new()
            .input(2)
            .hidden(2)
            .output(2)
            .finalize::<2, 1, 2>();

        log::debug!("New hidden layers: {:?}", network.hidden_layers);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let (training_data, test_batch) = {
            let mut data = vec![];
            for _ in 0..1000 {
                let x: f64 = rng.gen();
                let y: f64 = rng.gen();

                let line = |x: f64| -> f64 { x };
                let expected_output = if y <= line(x) {
                    [1f64, 0f64]
                } else {
                    [0f64, 1f64]
                };

                data.push(DataPoint::<2, 2> {
                    input: [x, y],
                    expected_output,
                });
            }

            let mut train_data = data
                .chunks(50)
                .map(|el| el.to_vec())
                .collect::<Vec<Vec<_>>>();
            let test = train_data.pop().unwrap();
            (train_data, test)
        };
        // log::debug!(
        //     "data\ntraining: {:?}\ntest: {:?}",
        //     training_data,
        //     test_batch
        // );

        let init_loss = avg_network_loss(&network, &test_batch);
        log::info!("Initial loss: {}", init_loss);

        std::thread::sleep(std::time::Duration::from_secs(2));

        for _ in 0..10000 {
            for batch in &training_data {
                train(&mut network, batch.to_vec(), 0.001);
                // log::debug!("{network:?}");
                let avg_loss = avg_network_loss(&network, &test_batch);
                log::info!("Avg loss: {}", avg_loss);
                //log::debug!("Network: {:?}", network.hidden_layers);
                //thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        let fin_loss = avg_network_loss(&network, &test_batch);
        log::info!("Final loss: {}", fin_loss);
        log::debug!("Final network: {network:?}");
    }
}
