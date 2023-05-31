use std::sync::{Arc, Mutex};
use std::thread;

use crate::{
    math,
    network::{Network, NeuronData},
};

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const I: usize, const O: usize> {
    pub input: [f64; I],
    pub expected_output: [f64; O],
}

fn update_gradients<const I: usize, const O: usize>(
    network_data: Vec<Vec<NeuronData>>,
    train_data: DataPoint<I, O>,
    gradients: Arc<Mutex<NetworkGradients>>,
) {
    let mut common_gradient_parts: Vec<Vec<f64>> = vec![];
    let mut prev_layer: Vec<NeuronData> = vec![];

    // Backpropagation
    network_data
        .iter()
        .rev()
        .enumerate()
        .for_each(|(i, layer)| {
            let grad_parts = layer
                .iter()
                .enumerate()
                .map(|(j, neuron)| {
                    if i == 0 {
                        // Output layer
                        // For every neuron in the output layer:
                        // multiply the derivative of its activation function by
                        // the derivative of the loss function
                        math::d_sigmoid(neuron.weighted_input_sum)
                            * node_loss_derivative(neuron.output, train_data.expected_output[j])
                    } else {
                        // Hidden layers
                        // For every neuron in the hidden layer
                        // multiply the derivative of its activation function by
                        // weights of the prevuous layer neurons
                        // multiplied by their respective derivative values
                        math::d_sigmoid(neuron.weighted_input_sum)
                            * prev_layer
                                .iter()
                                .zip(common_gradient_parts.last().unwrap())
                                .map(|(node, der)| node.weights[j] * der)
                                .sum::<f64>()
                    }
                })
                .collect::<Vec<_>>();
            prev_layer = layer.to_vec();
            common_gradient_parts.push(grad_parts);
        });

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

        // log::debug!("gradients\n{:#?}", gradients);
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
    for data in &train_data {
        let data = data.to_owned();
        let g = gradients.clone();
        let n = network.clone();
        let handle = thread::spawn(move || {
            let res = n.compute(data.input);
            update_gradients(res.layer_data, data, g);
        });

        handle.join().expect("Could not join training thread.");
    }

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
            // log::debug!(
            //     "got={:?}\texpected={:?}",
            //     res.output,
            //     data_point.expected_output
            // );
            network_loss(res.output, data_point.expected_output)
        })
        .sum::<f64>()
        / (test_data.len() as f64)
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use crate::network::NetworkBuilder;

    use super::*;

    fn do_log() {
        std::env::set_var("RUST_LOG", "debug");
        env_logger::init();
    }

    fn generate_xor_test_data(
        n: u32,
        c: usize,
    ) -> (Vec<Vec<DataPoint<2, 2>>>, Vec<DataPoint<2, 2>>) {
        let mut rng = rand::thread_rng();
        let mut train_data = vec![];
        for _ in 0..n {
            let x: f64 = rng.gen();
            let y: f64 = rng.gen();

            let expected_output = if (x > 0.5 && y > 0.5) || (x <= 0.5 && y <= 0.5) {
                [1f64, 0f64]
            } else {
                [0f64, 1f64]
            };
            train_data.push(DataPoint {
                input: [x, y],
                expected_output,
            });
        }

        let mut training_data = train_data.chunks(c).collect::<Vec<_>>();
        let test_batch = training_data.pop().unwrap();
        (
            training_data.iter().map(|v| v.to_vec()).collect(),
            test_batch.to_vec(),
        )
    }

    #[test]
    fn test_learning() {
        do_log();
        let mut network = NetworkBuilder::new()
            .input(2)
            .hidden(2)
            .hidden(2)
            .output(2)
            .finalize::<2, 2, 2>();

        let (training_data, test_batch) = generate_xor_test_data(1000, 10);
        // log::debug!(
        //     "data\ntraining: {:?}\ntest: {:?}",
        //     training_data,
        //     test_batch
        // );

        for _ in 0..1000 {
            for batch in &training_data {
                train(&mut network, batch.to_vec(), 0.5);
                // log::debug!("{network:?}");
                let avg_loss = avg_network_loss(&network, &test_batch);
                log::info!("Avg loss: {}", avg_loss);
                // log::debug!("Network\n{:#?}", network.hidden_layers);
            }
        }
        let fin_loss = avg_network_loss(&network, &test_batch);
        log::info!("Final loss: {}", fin_loss);
    }
}
