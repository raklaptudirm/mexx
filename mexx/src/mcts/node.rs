use super::policy;
use core::slice;

pub type NodePtr = isize;
pub type Result = f64;

pub struct Node {
    pub position: ataxx::Position,

    pub edges: Edges,

    pub playouts: usize,
    pub total_score: Result,
    pub parent_node: NodePtr,
}

impl Node {
    pub fn new(position: ataxx::Position, parent_node: NodePtr) -> Node {
        Node {
            position,

            edges: Edges::new(),

            playouts: 0,
            total_score: 0.0,
            parent_node,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.position.is_game_over()
    }

    pub fn expand(&mut self, policy: policy::Fn) {
        self.position.generate_moves_into(&mut self.edges);

        let mut sum = 0.0;
        let mut policies = vec![];
        for edge in self.edges.iter() {
            let policy = policy(self, edge.mov).max(0.0);
            let square = policy;
            policies.push(square);
            sum += square;
        }

        for (i, edge) in self.edges.iter_mut().enumerate() {
            edge.policy = policies[i] / sum;
        }
    }
    pub fn edge(&self, ptr: EdgePtr) -> &Edge {
        &self.edges.edges[ptr as usize]
    }

    pub fn edge_mut(&mut self, ptr: EdgePtr) -> &mut Edge {
        &mut self.edges.edges[ptr as usize]
    }
}

impl Node {
    pub fn q(&self) -> f64 {
        self.total_score / self.playouts.max(1) as f64
    }
}

pub struct Edges {
    edges: Vec<Edge>,
}

impl Edges {
    pub fn new() -> Self {
        Edges { edges: vec![] }
    }

    pub fn iter(&self) -> slice::Iter<'_, Edge> {
        self.edges.iter()
    }
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Edge> {
        self.edges.iter_mut()
    }
}

impl ataxx::MoveStore for Edges {
    fn push(&mut self, m: ataxx::Move) {
        self.edges.push(Edge::new(m));
    }

    fn len(&self) -> usize {
        self.edges.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type EdgePtr = isize;

pub struct Edge {
    pub mov: ataxx::Move,
    pub ptr: NodePtr,
    pub policy: f64,
}

impl Edge {
    pub fn new(m: ataxx::Move) -> Edge {
        Edge {
            mov: m,
            ptr: -1,
            policy: 0.0,
        }
    }
}