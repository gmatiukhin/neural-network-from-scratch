pub fn sigmoid(val: f64) -> f64 {
    1f64 / (1f64 + (-val).exp())
}

pub fn d_sigmoid(val: f64) -> f64 {
    let s = sigmoid(val);
    s * (1f64 - s)
}
