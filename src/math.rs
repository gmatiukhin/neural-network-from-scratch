/// Activation function
pub fn relu(val: f64) -> f64 {
    val.max(0f64)
}

/// Slightly modified derivative of ReLU activation function
pub fn d_relu(val: f64) -> f64 {
    if val <= 0f64 {
        0f64
    } else {
        1f64
    }
}

pub fn softmax<const T: usize>(data: [f64; T]) -> [f64; T] {
    log::debug!("data={data:?}");
    let exponents = data.into_iter().map(|el| el.exp()).collect::<Vec<_>>();
    log::debug!("exponents={exponents:?}");
    let sum = exponents.iter().fold(0f64, |acc, el| acc + el);
    log::debug!("sum={sum:?}");
    exponents
        .iter()
        .map(|el| el / sum)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Arrays length do not match")
}

// Log softmax https://medium.com/@AbhiramiVS/softmax-vs-logsoftmax-eb94254445a2
pub fn log_softmax<const T: usize>(data: [f64; T]) -> [f64; T] {
    log::debug!("data={data:?}");
    let exponents = data.iter().map(|el| el.exp()).collect::<Vec<_>>();
    log::debug!("exponents={exponents:?}");
    let max = exponents
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    log::debug!("max={max:?}");
    data.iter()
        .map(|el| el - max)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Arrays length do not match")
}
