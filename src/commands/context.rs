use std::str::FromStr;

use ataxx::Position;

use crate::mcts::{self, Searcher};

pub struct Context {
    pub position: Position,
    pub searcher: mcts::Searcher,
}

impl Default for Context {
    fn default() -> Self {
        let position = Position::from_str("x5o/7/7/7/7/7/o5x x 0 1").unwrap();
        Context {
            position,
            searcher: Searcher::new(position, mcts::policy::monty, mcts::value::monty),
        }
    }
}
