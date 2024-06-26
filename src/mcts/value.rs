pub type Fn = fn(position: &ataxx::Position) -> f64;

pub fn eval_to_wdl(eval: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-eval / 400.0))
}

pub fn wdl_to_eval(wdl: f64) -> f64 {
    -400.0 * f64::ln(1.0 / wdl - 1.0)
}

pub fn material(position: &ataxx::Position) -> f64 {
    const SCALE: f64 = 12.5;
    const TEMPO: f64 = SCALE * 4.0;

    let stm = position.side_to_move;

    let stm_piece_n = position.bitboard(stm).cardinality();
    let xtm_piece_n = position.bitboard(!stm).cardinality();

    stm_piece_n as f64 * SCALE - xtm_piece_n as f64 * SCALE + TEMPO
}
