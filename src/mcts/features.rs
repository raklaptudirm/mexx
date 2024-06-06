use goober::SparseVector;

use super::policy::PolicyNetwork;
use super::value::ValueNetwork;

#[repr(C)]
pub struct Nets(pub ValueNetwork<2916, 256>, pub PolicyNetwork);

pub const NETS: Nets =
    unsafe { std::mem::transmute(*include_bytes!("../../resources/net.network")) };

pub fn value_feature_map<F: FnMut(usize)>(position: &ataxx::Position, mut f: F) {
    const PER_TUPLE: usize = 3usize.pow(4);
    const POWERS: [usize; 4] = [1, 3, 9, 27];
    const MASK: u64 = 0b0001_1000_0011;

    let friends = position.bitboard(position.side_to_move).0;
    let enemies = position.bitboard(!position.side_to_move).0;

    for i in 0..6 {
        for j in 0..6 {
            let tuple = 6 * i + j;
            let mut feat = PER_TUPLE * tuple;

            let offset = 7 * i + j;
            let mut b = (friends >> offset) & MASK;
            let mut o = (enemies >> offset) & MASK;

            while b > 0 {
                let mut sq = b.trailing_zeros() as usize;
                if sq > 6 {
                    sq -= 5;
                }

                feat += POWERS[sq];

                b &= b - 1;
            }

            while o > 0 {
                let mut sq = o.trailing_zeros() as usize;
                if sq > 6 {
                    sq -= 5;
                }

                feat += 2 * POWERS[sq];

                o &= o - 1;
            }

            f(feat);
        }
    }
}

pub fn get_features(position: &ataxx::Position) -> SparseVector {
    let mut feats = SparseVector::with_capacity(36);

    value_feature_map(position, |feat| feats.push(feat));

    feats
}
