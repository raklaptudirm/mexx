use super::{policy, simulate, EdgePtr, Node, NodePtr, Params};
use ataxx::MoveStore;

pub struct Tree {
    nodes: Vec<Node>,
    params: Params,
    policy: policy::Fn,
    simulator: simulate::Fn,
}

impl Tree {
    pub fn new(position: ataxx::Position) -> Tree {
        let policy = policy::monty;
        let simulator = simulate::material_count;

        let mut root = Node::new(position, -1);
        root.expand(policy);

        Tree {
            nodes: vec![root],
            params: Params::new(),
            policy,
            simulator,
        }
    }

    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn playouts(&self) -> usize {
        self.node(0).playouts
    }

    pub fn best_move(&self) -> ataxx::Move {
        let root = self.node(0);

        let mut best_mov = ataxx::Move::NULL;
        let mut best_scr = 0.0;
        for edge in root.edges.iter() {
            if edge.ptr == -1 {
                continue;
            }

            let node = self.node(edge.ptr);
            let score = 1.0 - node.q();

            if best_mov == ataxx::Move::NULL || score > best_scr {
                best_mov = edge.mov;
                best_scr = score;
            }
        }

        best_mov
    }
}

impl Tree {
    pub fn playout(&mut self) -> NodePtr {
        let selected = self.select(); // Select Node to be expanded
        let expanded = self.expand(selected); // Expand the selected Node
        let simulate = self.simulate(expanded); // Simulate the Node's result
        self.backpropagate(expanded, simulate); // Backpropagate the simulation

        expanded
    }

    pub fn select(&mut self) -> NodePtr {
        let policy = self.policy;
        let mut node_ptr: NodePtr = 0;

        loop {
            let node = self.node_mut(node_ptr);

            if node_ptr != 0 && node.playouts == 1 {
                // If the selected Node's Edges haven't been expanded, expand.
                node.expand(policy);
            }

            let node = self.node(node_ptr);

            if node.is_terminal() {
                break;
            }

            // Select a new Edge from the current Node, and get the child Node.
            let edge = self.select_edge(node_ptr);
            let node = node.edge(edge).ptr;

            if node == -1 {
                // Selected Edge hasn't been expanded, so end selection for expansion.
                break;
            }

            // Replace the Node pointer with the newly selected Node.
            node_ptr = node;
        }

        node_ptr
    }

    //                    v-----------------------v exploitation
    //  node-q + policy * cpuct * sqrt(node-visits) / (1 + child-visits = 1)
    // child-q + policy * cpuct * sqrt(node-visits) / (1 + child-visits)
    // ^-----^ score / visits
    pub fn select_edge(&self, ptr: NodePtr) -> EdgePtr {
        let node = self.node(ptr);

        // Node exploitation factor (cpuct * sqrt(parent-playouts))
        let e = self.params.cpuct() * f64::sqrt(node.playouts.max(1) as f64);

        let mut best_ptr: EdgePtr = -1;
        let mut best_uct = 0.0;

        // Q value (score / playouts) for parent node.
        let node_q = node.q();

        for (ptr, edge) in node.edges.iter().enumerate() {
            let ptr = ptr as EdgePtr;

            // Fetch the Q value, Policy value, and Playout count.
            let (q, p, c) = if edge.ptr == -1 {
                // Edge hasn't been expanded, so no node information available.
                // Use the parent (current) node's information instead for uct.
                (node_q, edge.policy, 1.0) // No child playouts, so playouts + 1 = 1
            } else {
                let child = self.node(edge.ptr);
                (child.q(), edge.policy, (child.playouts + 1) as f64)
            };

            let child_uct = q + p * e / c;

            // Check if we have a better UCT score for this edge.
            if child_uct > best_uct {
                best_ptr = ptr;
                best_uct = child_uct;
            }
        }

        best_ptr
    }

    pub fn expand(&mut self, ptr: NodePtr) -> NodePtr {
        if self.node(ptr).is_terminal() {
            return ptr;
        }

        // Select an Edge to expand from the current Node.
        let edge_ptr = self.select_edge(ptr);

        let node = self.node(ptr);
        let edge = node.edge(edge_ptr);

        // Expand the Edge into a new Node.
        let new_node = Node::new(node.position.after_move::<true>(edge.mov), ptr);

        // Add the new Node to the Tree.
        self.nodes.push(new_node);

        // Get the NodePtr of the new Node.
        let new_ptr = (self.nodes.len() - 1) as isize;

        let edge = self.node_mut(ptr).edge_mut(edge_ptr);

        // Make the Edge point to the new Node.
        edge.ptr = new_ptr;

        new_ptr
    }

    pub fn simulate(&mut self, ptr: NodePtr) -> f64 {
        let position = &self.node(ptr).position;

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

        (self.simulator)(position)
    }

    pub fn backpropagate(&mut self, ptr: NodePtr, result: f64) {
        let mut node_ptr = ptr;
        let mut result = result;

        loop {
            let node = self.node_mut(node_ptr);

            node.playouts += 1;
            node.total_score += result;

            // Stop backpropagation if root has been reached.
            if node_ptr == 0 {
                break;
            }

            node_ptr = node.parent_node;
            result = 1.0 - result;
        }
    }
}

impl Tree {
    pub fn node(&self, ptr: NodePtr) -> &Node {
        &self.nodes[ptr as usize]
    }

    fn node_mut(&mut self, ptr: NodePtr) -> &mut Node {
        &mut self.nodes[ptr as usize]
    }
}

impl Tree {
    pub fn verify(&self) -> Result<(), String> {
        self.verify_node(0)
    }

    fn verify_node(&self, ptr: NodePtr) -> Result<(), String> {
        let node = self.node(ptr);
        if !(node.total_score >= 0.0 && node.total_score <= node.playouts as f64) {
            return Err("node score out of bounds [0, playouts]".to_string());
        }

        let mut child_playouts = 0;
        let mut policy_sum = 0.0;
        for edge in node.edges.iter() {
            policy_sum += edge.policy;

            if edge.ptr == -1 {
                continue;
            }

            let child = self.node(edge.ptr);

            if child.position.checksum != node.position.after_move::<true>(edge.mov).checksum {
                return Err("position not matching after making node's move".to_string());
            }

            self.verify_node(edge.ptr)?;

            child_playouts += child.playouts;
        }

        if node.edges.len() > 0 && (1.0 - policy_sum).abs() > 0.00001 {
            return Err(format!("total playout probability {} not 1", policy_sum));
        }

        if (ptr == 0 && node.playouts != child_playouts)
            || (ptr != 0 && !node.is_terminal() && node.playouts != child_playouts + 1)
        {
            println!("{}", node.position);
            Err(format!(
                "node playouts {} while child playouts {}",
                node.playouts, child_playouts
            ))
        } else {
            Ok(())
        }
    }
}
