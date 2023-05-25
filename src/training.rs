use crate::network::Network;

pub trait Trainable<const I: usize, const H: usize, const O: usize> {
    fn loss(&mut self, train_data: TrainData<I, O>) -> f64;
    fn avg_loss(&mut self, train_data: Vec<TrainData<I, O>>) -> f64;
}

impl<const I: usize, const H: usize, const O: usize> Trainable<I, H, O> for Network<I, H, O> {
    fn loss(&mut self, train_data: TrainData<I, O>) -> f64 {
        fn node_loss(got: f64, expected: f64) -> f64 {
            let err = got - expected;
            err * err
        }

        let got_outputs = self.compute(train_data.input);
        got_outputs
            .iter()
            .zip(train_data.expected_output)
            .fold(0f64, |acc, (&got, expected)| acc + node_loss(got, expected))
    }

    fn avg_loss(&mut self, train_data: Vec<TrainData<I, O>>) -> f64 {
        train_data.iter().fold(0f64, |acc, &val| {
            (acc + self.loss(val)) / (train_data.len() as f64)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainData<const I: usize, const O: usize> {
    pub input: [f64; I],
    pub expected_output: [f64; O],
}

fn _train<const I: usize, const H: usize, const O: usize>(_model: Network<I, H, O>) {}
