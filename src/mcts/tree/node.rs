use ataxx::MoveStore;
use derive_new::new;

use super::super::policy;
use std::slice;

pub type NodePtr = i32;
pub type Score = f64;

#[derive(Clone, new)]
pub struct Node {
    #[new(value = "Edges::new()")]
    pub edges: Edges,

    pub parent_node: NodePtr,
    pub parent_edge: EdgePtr,
}

impl Node {
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
#[derive(Clone, new)]
pub struct Edges {
    #[new(value = "vec![]")]
    edges: Vec<Edge>,
}

impl Edges {
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

impl Default for Node {
    fn default() -> Self {
        Node {
            edges: Edges::new(),
            parent_node: -1,
            parent_edge: -1,
        }
    }
}

pub type EdgePtr = i32;

#[derive(Clone, new)]
pub struct Edge {
    pub mov: ataxx::Move,
    #[new(value = "-1")]
    pub ptr: NodePtr,

    #[new(value = "0")]
    pub visits: usize,
    #[new(value = "0.0")]
    pub scores: Score,

    #[new(value = "0.0")]
    pub policy: f64,
}

impl Edge {
    pub fn q(&self) -> f64 {
        self.scores / self.visits.max(1) as f64
    }
}
