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
    let exponents = data.into_iter().map(|el| el.exp());
    let sum = exponents.clone().fold(0f64, |acc, el| acc + el);
    exponents
        .map(|el| el / sum)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Arrays length do not match")
}
