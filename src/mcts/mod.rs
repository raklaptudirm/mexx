pub mod policy;
pub mod value;

mod features;
mod node;
mod params;
mod tree;

use std::mem;
use std::time;

pub use self::features::*;
pub use self::node::*;
pub use self::params::*;
pub use self::tree::*;

#[derive(Clone)]
pub struct Searcher {
    tree: Tree,
    params: Params,
    policy: policy::Fn,
    value: value::Fn,
}

#[derive(Debug)]
pub struct Limits {
    pub maxdepth: Option<usize>,
    pub maxnodes: Option<usize>,
    pub movetime: Option<u128>,
    pub movestogo: Option<usize>,
}

impl Searcher {
    pub fn new(position: ataxx::Position, policy: policy::Fn, value: value::Fn) -> Searcher {
        Searcher {
            tree: Tree::new(position),
            params: Params::new(),
            policy,
            value,
        }
    }

    pub fn update_position(&mut self, position: ataxx::Position) {
        self.tree = Tree::new(position);
    }

    pub fn search(&mut self, limits: Limits, total_nodes: &mut u64) -> ataxx::Move {
        let maxdepth = limits.maxdepth.unwrap_or(usize::MAX);
        let maxnodes = limits.maxnodes.unwrap_or(usize::MAX);
        let movetime = limits.movetime.unwrap_or(u128::MAX);

        let start = time::Instant::now();

        let mut rollouts = 0;

        let mut depth = 0;
        let mut seldepth = 0;
        let mut cumulative_depth = 0;

        loop {
            let mut new_depth = 0;
            let mut position = self.tree.root_position();

            self.do_one_rollout(0, &mut position, &mut new_depth);
            rollouts += 1;

            cumulative_depth += new_depth;
            if new_depth > seldepth {
                seldepth = new_depth;
            }

            let avg_depth = cumulative_depth / rollouts;
            if avg_depth > depth {
                depth = avg_depth;

                // Make a new info report.
                println!(
                    "info depth {} seldepth {} score cp {:.0} nodes {} nps {}",
                    depth,
                    seldepth,
                    0.0,
                    rollouts,
                    self.tree.nodes() * 1000 / start.elapsed().as_millis().max(1) as usize
                );
            }

            if rollouts & 127 == 0 {
                if start.elapsed().as_millis() >= movetime
                    || depth >= maxdepth
                    || self.tree.nodes() >= maxnodes
                {
                    break;
                }

                // Hard memory limit to prevent overuse.
                // TODO: Fix this by removing old nodes and stuff.
                if self.tree.nodes() > 2_000_000_000 / mem::size_of::<Node>() {
                    break;
                }
            }
        }

        *total_nodes += self.tree.nodes() as u64;

        println!(
            "info depth {} seldepth {} score cp {:.0} nodes {} nps {}",
            cumulative_depth / rollouts,
            seldepth,
            100.0,
            rollouts,
            self.tree.nodes() * 1000 / start.elapsed().as_millis().max(1) as usize
        );

        // Verify the self.
        // debug_assert!(self.verify().is_ok());

        self.tree.best_move()
    }
}

impl Searcher {
    fn do_one_rollout(
        &mut self,
        node_ptr: NodePtr,
        position: &mut ataxx::Position,
        depth: &mut usize,
    ) -> Score {
        *depth += 1;

        let node = self.tree.node(node_ptr);
        let parent_node = node.parent_node;
        let parent_edge = node.parent_edge;

        let edge_visits = self.tree.edge(parent_node, parent_edge).visits;

        let node = self.tree.node_mut(node_ptr);

        let score = if position.is_game_over() || edge_visits == 0 {
            self.simulate(position)
        } else {
            if !node.expanded() {
                node.expand(position, self.policy)
            }

            let edge_ptr = self.select_edge(node_ptr);
            let edge = self.tree.edge(node_ptr, edge_ptr);
            *position = position.after_move::<true>(edge.mov);

            let mut child_ptr = edge.ptr;
            if child_ptr == -1 {
                child_ptr = self.tree.push_node(Node::new(node_ptr, edge_ptr));
                self.tree.edge_mut(node_ptr, edge_ptr).ptr = child_ptr;
            }

            self.do_one_rollout(child_ptr, position, depth)
        };

        let score = 1.0 - score;

        let edge = self.tree.edge_mut(parent_node, parent_edge);

        edge.visits += 1;
        edge.scores += score;

        score
    }

    //                    v-----------------------v exploitation
    //  node-q + policy * cpuct * sqrt(node-visits) / (1 + child-visits = 0) // not expanded
    // child-q + policy * cpuct * sqrt(node-visits) / (1 + child-visits)     // expanded
    // ^-----^ score / visits
    fn select_edge(&self, ptr: NodePtr) -> EdgePtr {
        let node = self.tree.node(ptr);
        let parent = self.tree.edge(node.parent_node, node.parent_edge);

        // Node exploitation factor (cpuct * sqrt(parent-playouts))
        let e = self.params.cpuct() * f64::sqrt(parent.visits.max(1) as f64);

        let mut best_ptr: EdgePtr = -1;
        let mut best_uct = 0.0;

        for (ptr, edge) in node.edges.iter().enumerate() {
            // If the edge hasn't been expanded yet, use the parent's q value.
            let q = if edge.ptr == -1 { parent.q() } else { edge.q() };

            let child_uct = q + edge.policy * e / (edge.visits as f64 + 1.0);

            // Check if we have a better UCT score for this edge.
            if child_uct > best_uct {
                best_ptr = ptr as EdgePtr;
                best_uct = child_uct;
            }
        }

        best_ptr
    }

    fn simulate(&mut self, position: &ataxx::Position) -> f64 {
        if position.is_game_over() {
            let winner = position.winner();
            return if winner == ataxx::Piece::None {
                0.5 // Draw
            } else if winner == position.side_to_move {
                1.0 // Win
            } else {
                0.0 // Loss
            };
        };

        (self.value)(position)
    }
}
