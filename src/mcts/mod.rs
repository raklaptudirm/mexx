use std::time;

pub use self::params::*;
pub use self::tree::*;

pub mod policy;
pub mod value;

mod params;
mod tree;

#[derive(Clone)]
pub struct Searcher {
    tree: Tree,
    params: Params,
    policy: policy::Fn,
    value: value::Fn,

    start: time::Instant,

    rollouts: usize,
    avgdepth: usize,
    seldepth: usize,
    cumdepth: usize,
}

#[derive(Debug)]
pub struct Limits {
    pub maxdepth: Option<usize>,
    pub maxnodes: Option<usize>,
    pub movetime: Option<u128>,

    #[allow(unused)]
    pub movestogo: Option<usize>,
}

impl Searcher {
    pub fn new(position: ataxx::Position, policy: policy::Fn, value: value::Fn) -> Searcher {
        Searcher {
            tree: Tree::new(position),
            params: Params::new(),
            policy,
            value,

            start: time::Instant::now(),

            rollouts: 0,
            avgdepth: 0,
            seldepth: 0,
            cumdepth: 0,
        }
    }

    pub fn update_position(&mut self, position: ataxx::Position) {
        self.tree = Tree::new(position);
    }

    pub fn search(&mut self, limits: Limits, total_nodes: &mut u64) -> ataxx::Move {
        let maxdepth = limits.maxdepth.unwrap_or(usize::MAX);
        let maxnodes = limits.maxnodes.unwrap_or(usize::MAX);
        let movetime = limits.movetime.unwrap_or(u128::MAX);

        self.start = time::Instant::now();

        self.rollouts = 0;

        self.avgdepth = 0;
        self.seldepth = 0;
        self.cumdepth = 0;

        loop {
            let mut new_depth = 0;
            let mut position = self.tree.root_position();

            self.do_one_rollout(0, &mut position, &mut new_depth);
            self.rollouts += 1;

            self.cumdepth += new_depth;
            if new_depth > self.seldepth {
                self.seldepth = new_depth;
            }

            let avg_depth = self.cumdepth / self.rollouts;
            if avg_depth > self.avgdepth {
                self.avgdepth = avg_depth;

                // Make a new info report.
                self.uci_report();
            }

            if self.rollouts & 127 == 0
                && (self.start.elapsed().as_millis() >= movetime
                    || self.avgdepth >= maxdepth
                    || self.rollouts >= maxnodes)
            {
                break;
            }
        }

        *total_nodes += self.rollouts as u64;

        self.uci_report();

        // Verify the self.
        debug_assert_eq!(self.tree.verify(), Ok(()));

        self.tree.best_move()
    }

    fn uci_report(&self) {
        let (pv, score) = self.tree.pv(0);

        let pv_str = pv
            .iter()
            .map(|mov| mov.to_string())
            .reduce(|acc, ele| format!("{} {}", acc, ele))
            .unwrap_or("".to_string());

        let score_str = if score >= 1.0 {
            format!("mate {}", (pv.len() + 1) / 2)
        } else if score <= 0.0 {
            format!("mate -{}", pv.len() / 2)
        } else {
            format!("cp {:.0}", value::wdl_to_eval(score))
        };

        println!(
            "info depth {} seldepth {} score {} nodes {} nps {} pv {}",
            self.avgdepth,
            self.seldepth,
            score_str,
            self.rollouts,
            self.rollouts * 1000 / self.start.elapsed().as_millis().max(1) as usize,
            pv_str,
        );
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

        self.tree.promote(node_ptr);

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
                child_ptr = self.tree.push(Node::new(node_ptr, edge_ptr));
                self.tree.edge_mut(node_ptr, edge_ptr).ptr = child_ptr;
            }

            self.do_one_rollout(child_ptr, position, depth)
        };

        let score = 1.0 - score;

        let edge = self.tree.edge_mut(parent_node, parent_edge);

        edge.visits += 1;
        edge.scores += score;

        self.tree.promote(node_ptr);

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

        let fpu = 1.0 - parent.q();

        for (ptr, edge) in node.edges.iter().enumerate() {
            // If the edge hasn't been expanded yet, use the parent's q value.
            let q = if edge.ptr == -1 { fpu } else { edge.q() };

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

        value::eval_to_wdl((self.value)(position))
    }
}
