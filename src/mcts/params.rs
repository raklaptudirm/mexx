use std::f64::consts::SQRT_2;

use derive_new::new;

#[derive(Clone)]
pub struct Params {
    cpuct: Param,
}

impl Params {
    pub fn new() -> Params {
        Params {
            cpuct: Param::new(SQRT_2, 1.0, 10.0),
        }
    }
    pub fn cpuct(&self) -> f64 {
        self.cpuct.val
    }
}

#[derive(Clone, new)]
pub struct Param {
    val: f64,

    #[allow(unused)]
    min: f64,
    #[allow(unused)]
    max: f64,
}

impl Param {
    #[allow(unused)]
    pub fn set(&mut self, val: f64) {
        self.val = val;
    }
}
