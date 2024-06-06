use ataxx::MoveStore;

use super::policy;
use core::slice;

pub type NodePtr = isize;
pub type Score = f64;

#[derive(Clone)]
pub struct Node {
    pub edges: Edges,

    pub parent_node: NodePtr,
    pub parent_edge: EdgePtr,
}

impl Node {
    pub fn new(parent_node: NodePtr, parent_edge: EdgePtr) -> Node {
        Node {
            edges: Edges::new(),
            parent_node,
            parent_edge,
        }
    }

    pub fn expand(&mut self, position: &ataxx::Position, policy: policy::Fn) {
        position.generate_moves_into(&mut self.edges);

        let mut sum = 0.0;
        let mut policies = vec![];
        for edge in self.edges.iter() {
            let policy = policy(position, edge.mov).exp();
            policies.push(policy);
            sum += policy;
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

    pub fn expanded(&self) -> bool {
        self.edges.len() > 0
    }
}
#[derive(Clone)]
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

#[derive(Clone)]
pub struct Edge {
    pub mov: ataxx::Move,
    pub ptr: NodePtr,

    pub visits: usize,
    pub scores: Score,

    pub policy: f64,
}

impl Edge {
    pub fn new(m: ataxx::Move) -> Edge {
        Edge {
            mov: m,
            ptr: -1,

            visits: 0,
            scores: 0.0,

            policy: 0.0,
        }
    }

    pub fn q(&self) -> f64 {
        self.scores / self.visits.max(1) as f64
    }
}
