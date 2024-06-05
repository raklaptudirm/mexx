use super::{value_feature_map, NETS};

pub type Fn = fn(position: &ataxx::Position) -> f64;

pub fn material_count(position: &ataxx::Position) -> f64 {
    let stm = position.side_to_move;

    let stm_piece_n = position.bitboard(stm).cardinality();
    let xtm_piece_n = position.bitboard(!stm).cardinality();

    let eval = stm_piece_n as f64 - xtm_piece_n as f64;

    1.0 / (1.0 + f64::exp(-eval / 400.0))
}

pub fn monty_network(position: &ataxx::Position) -> f64 {
    1.0 / (1.0 + (-(NETS.0.eval(position) as f64) / 400.0).exp())
}

const SCALE: i32 = 400;
const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

#[repr(C, align(64))]
pub struct ValueNetwork<const INPUT: usize, const HIDDEN: usize> {
    l1_weights: [Accumulator<HIDDEN>; INPUT],
    l1_bias: Accumulator<HIDDEN>,
    l2_weights: Accumulator<HIDDEN>,
    l2_bias: i16,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Accumulator<const HIDDEN: usize> {
    vals: [i16; HIDDEN],
}

#[inline]
fn screlu(x: i16) -> i32 {
    i32::from(x).clamp(0, QA).pow(2)
}

impl<const INPUT: usize, const HIDDEN: usize> ValueNetwork<INPUT, HIDDEN> {
    pub fn eval(&self, board: &ataxx::Position) -> i32 {
        let mut acc = self.l1_bias;

        value_feature_map(board, |feat| {
            for (i, d) in acc.vals.iter_mut().zip(&self.l1_weights[feat].vals) {
                *i += *d;
            }
        });

        let mut eval = 0;

        for (&v, &w) in acc.vals.iter().zip(self.l2_weights.vals.iter()) {
            eval += screlu(v) * i32::from(w);
        }

        (eval / QA + i32::from(self.l2_bias)) * SCALE / QAB
    }
}
