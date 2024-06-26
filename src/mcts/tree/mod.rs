use derive_more::{Deref, DerefMut};

use ataxx::MoveStore;

pub use self::node::*;

mod lru;
mod node;

#[derive(Clone, Deref, DerefMut)]
pub struct Tree {
    #[deref]
    #[deref_mut]
    nodes: lru::Cache,
    root_pos: ataxx::Position,
}

impl Tree {
    pub fn new(position: ataxx::Position) -> Tree {
        let mut cache = lru::Cache::new_mib(64);
        cache.push(Default::default());
        Tree {
            root_pos: position,
            nodes: cache,
        }
    }

    pub fn pv(&self, node_ptr: NodePtr) -> (Vec<ataxx::Move>, Score) {
        let node = self.node(node_ptr);

        let mut best_edge = -1;
        let mut best_score = 0.0;
        for (edge_ptr, edge) in node.edges.iter().enumerate() {
            if best_edge == -1 || edge.q() > best_score {
                best_edge = edge_ptr as i32;
                best_score = edge.q();
            }
        }

        // No edges found in the current node.
        if best_edge == -1 {
            return (vec![], 0.0);
        }

        let edge = node.edge(best_edge);
        if edge.ptr == -1 {
            (vec![edge.mov], best_score)
        } else {
            let (mut child_pv, _score) = self.pv(edge.ptr);
            let mut pv = vec![edge.mov];
            pv.append(&mut child_pv);
            (pv, best_score)
        }
    }

    pub fn root_position(&self) -> ataxx::Position {
        self.root_pos
    }

    pub fn best_move(&self) -> ataxx::Move {
        let root = self.node(0);

        let mut best_mov = ataxx::Move::NULL;
        let mut best_scr = 0.0;
        for edge in root.edges.iter() {
            if edge.visits == 0 {
                continue;
            }

            let score = edge.q();

            if best_mov == ataxx::Move::NULL || score > best_scr {
                best_mov = edge.mov;
                best_scr = score;
            }
        }

        best_mov
    }
}

impl Tree {
    pub fn verify(&self) -> Result<(), String> {
        self.verify_node(0, self.root_pos)
    }

    fn verify_node(&self, ptr: NodePtr, position: ataxx::Position) -> Result<(), String> {
        let node = self.node(ptr);

        let mut policy_sum = 0.0;
        for edge in node.edges.iter() {
            if !(edge.scores >= 0.0 && edge.scores <= edge.visits as f64) {
                return Err("edge score out of bounds [0, playouts]".to_string());
            }

            policy_sum += edge.policy;

            if edge.ptr == -1 {
                continue;
            }

            self.verify_node(edge.ptr, position.after_move::<true>(edge.mov))?;
        }

        if node.edges.len() > 0 && (1.0 - policy_sum).abs() > 0.00001 {
            return Err(format!("sum of all the policies is {}, not 1", policy_sum));
        }

        Ok(())
    }
}
