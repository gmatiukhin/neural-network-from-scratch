pub fn sigmoid(val: f64) -> f64 {
    1f64 / (1f64 + (-val).exp())
}

pub fn d_sigmoid(val: f64) -> f64 {
    let s = sigmoid(val);
    s * (1f64 - s)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sigmoid() {
        for i in i32::MIN..=i32::MAX {
            let val = sigmoid(i as f64);
            assert!((0f64..=1f64).contains(&val));
        }
    }
}
