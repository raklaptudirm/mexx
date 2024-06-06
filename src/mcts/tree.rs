use super::{Edge, EdgePtr, Node, NodePtr};

#[derive(Clone)]
pub struct Tree {
    root_pos: ataxx::Position,
    root_edge: Edge,
    nodes: Vec<Node>,
}

impl Tree {
    pub fn new(position: ataxx::Position) -> Tree {
        let root = Node::new(-1, -1);

        Tree {
            root_pos: position,
            root_edge: Edge::new(ataxx::Move::NULL),
            nodes: vec![root],
        }
    }

    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_position(&self) -> ataxx::Position {
        self.root_pos
    }

    pub fn node(&self, ptr: NodePtr) -> &Node {
        &self.nodes[ptr as usize]
    }

    pub fn node_mut(&mut self, ptr: NodePtr) -> &mut Node {
        &mut self.nodes[ptr as usize]
    }

    pub fn edge(&self, parent: NodePtr, edge_ptr: EdgePtr) -> &Edge {
        if parent == -1 {
            &self.root_edge
        } else {
            self.node(parent).edge(edge_ptr)
        }
    }

    pub fn edge_mut(&mut self, parent: NodePtr, edge_ptr: EdgePtr) -> &mut Edge {
        if parent == -1 {
            &mut self.root_edge
        } else {
            self.node_mut(parent).edge_mut(edge_ptr)
        }
    }

    pub fn push_node(&mut self, node: Node) -> NodePtr {
        self.nodes.push(node);
        self.nodes.len() as NodePtr - 1
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

// impl Tree {
//     pub fn verify(&self) -> Result<(), String> {
//         self.verify_node(0, self.root_pos)
//     }

//     fn verify_node(&self, ptr: NodePtr, position: ataxx::Position) -> Result<(), String> {
//         let node = self.node(ptr);
//         if !(node.total_score >= 0.0 && node.total_score <= node.playouts as f64) {
//             return Err("node score out of bounds [0, playouts]".to_string());
//         }

//         let mut child_playouts = 0;
//         let mut policy_sum = 0.0;
//         for edge in node.edges.iter() {
//             policy_sum += edge.policy;

//             if edge.ptr == -1 {
//                 continue;
//             }

//             let child_position = position.after_move::<true>(edge.mov);
//             let child = self.node(edge.ptr);

//             self.verify_node(edge.ptr, child_position)?;

//             child_playouts += child.playouts;
//         }

//         if node.edges.len() > 0 && (1.0 - policy_sum).abs() > 0.00001 {
//             return Err(format!("total playout probability {} not 1", policy_sum));
//         }

//         if (ptr == 0 && node.playouts != child_playouts)
//             || (ptr != 0 && !position.is_game_over() && node.playouts != child_playouts + 1)
//         {
//             println!("{}", position);
//             Err(format!(
//                 "node playouts {} while child playouts {}",
//                 node.playouts, child_playouts
//             ))
//         } else {
//             Ok(())
//         }
//     }
// }
