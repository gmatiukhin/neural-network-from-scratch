use std::{fmt::Debug, sync::Arc};

pub type ActivationFunction = Arc<dyn Activation + Sync + Send>;

pub trait Activation: Debug {
    fn activate(&self, val: f64) -> f64;
    fn derivative(&self, val: f64) -> f64;
}

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, val: f64) -> f64 {
        1f64 / (1f64 + (-val).exp())
    }

    fn derivative(&self, val: f64) -> f64 {
        let s = self.activate(val);
        s * (1f64 - s)
    }
}

#[derive(Debug, Clone)]
pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, val: f64) -> f64 {
        val.max(0f64)
    }

    fn derivative(&self, val: f64) -> f64 {
        if val < 0f64 {
            0f64
        } else {
            1f64
        }
    }
}

#[cfg(test)]
mod test {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    use super::*;

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        (i32::MIN..=i32::MAX).into_par_iter().for_each(|i| {
            let val = sigmoid.activate(i as f64);
            assert!((0f64..=1f64).contains(&val));
        });
    }
}
