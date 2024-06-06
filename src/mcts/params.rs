use std::f64::consts::SQRT_2;

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

#[derive(Clone)]
pub struct Param {
    val: f64,

    #[allow(unused)]
    min: f64,
    #[allow(unused)]
    max: f64,
}

impl Param {
    pub fn new(val: f64, min: f64, max: f64) -> Param {
        Param { val, min, max }
    }

    #[allow(unused)]
    pub fn set(&mut self, val: f64) {
        self.val = val;
    }
}
