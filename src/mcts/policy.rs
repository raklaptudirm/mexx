use ataxx::{BitBoard, Move};

pub type Fn = fn(position: &ataxx::Position, mov: Move) -> f64;

pub fn handcrafted(position: &ataxx::Position, mov: Move) -> f64 {
    let mut score = 0.0;

    let stm = position.side_to_move;
    let xtm = !position.side_to_move;

    let friends = position.bitboard(stm);
    let enemies = position.bitboard(xtm);

    let old_neighbours = BitBoard::single(mov.source());
    let new_neighbours = BitBoard::single(mov.target());

    score += (enemies & new_neighbours).cardinality() as f64 * 1.0;
    score += (friends & new_neighbours).cardinality() as f64 * 0.4;

    if mov.is_single() {
        score += 0.7;
    } else {
        score -= (friends & old_neighbours).cardinality() as f64 * 0.4;
    }

    score.max(0.1)
}

pub fn monty(node: &Node, mov: Move) -> f64 {
    NETS.1.get(&mov, &get_features(&node.position)) as f64
}

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SubNet {
    ft: layer::SparseConnected<activation::ReLU, 2916, 8>,
}

impl SubNet {
    pub const fn zeroed() -> Self {
        Self {
            ft: layer::SparseConnected::zeroed(),
        }
    }

    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        let matrix = Matrix::from_fn(|_, _| f());
        let vector = Vector::from_fn(|_| f());

        Self {
            ft: layer::SparseConnected::from_raw(matrix, vector),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub subnets: [SubNet; 99],
}

impl PolicyNetwork {
    pub fn get(&self, mov: &Move, feats: &SparseVector) -> f32 {
        let from_subnet = &self.subnets[(mov.source() as usize).min(49)];
        let from_vec = from_subnet.out(feats);

        let to_subnet = &self.subnets[50 + (mov.target() as usize).min(48)];
        let to_vec = to_subnet.out(feats);

        from_vec.dot(&to_vec)
    }
}
