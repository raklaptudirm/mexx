use std::time;

use uxi::{error, Bundle, Command, Flag, RunError};

use super::Context;
use crate::mcts;

pub fn go() -> Command<Context> {
    Command::new(|bundle: Bundle<Context>| {
        let binc = bundle.is_flag_set("binc");
        let winc = bundle.is_flag_set("winc");
        let btime = bundle.is_flag_set("btime");
        let wtime = bundle.is_flag_set("wtime");
        let movestogo = bundle.is_flag_set("movestogo");
        let depth = bundle.is_flag_set("depth");
        let nodes = bundle.is_flag_set("nodes");
        let movetime = bundle.is_flag_set("movetime");
        // let ponder = bundle.is_flag_set("ponder");
        let infinite = bundle.is_flag_set("infinite");

        let std_tc = btime && wtime;

        if !std_tc && (binc | winc | btime | wtime | movestogo) {
            return error!("bad flag set: unexpected standard time control flags");
        }

        let ctx = bundle.lock();
        let position = ctx.position;
        drop(ctx);

        let best_move = if infinite {
            if std_tc | depth | nodes | movetime {
                return error!("bad flag set: no other flags with infinite");
            }

            todo!()
        } else if std_tc {
            if depth | nodes | movetime {
                return error!("bad flag set: unexpected non-standard time control flags");
            }

            macro_rules! get_duration {
                ($name:literal) => {
                    time::Duration::from_millis(bundle.get_single_flag($name).unwrap().parse()?)
                };
            }

            let tc = StandardTC {
                btime: get_duration!("btime"),
                wtime: get_duration!("wtime"),
                binc: get_duration!("binc"),
                winc: get_duration!("winc"),
                movestogo: match bundle.get_single_flag("movestogo") {
                    Some(movestogo) => Some(movestogo.parse()?),
                    None => None,
                },
            };

            search_std(position, tc)
        } else {
            let tc = InternalTC {
                nodes: match bundle.get_single_flag("nodes") {
                    Some(nodes) => Some(nodes.parse()?),
                    None => None,
                },
                depth: match bundle.get_single_flag("depth") {
                    Some(depth) => Some(depth.parse()?),
                    None => None,
                },
                movetime: match bundle.get_single_flag("movetime") {
                    Some(movetime) => Some(time::Duration::from_millis(movetime.parse()?)),
                    None => None,
                },
            };

            search_int(position, tc)
        };

        println!("bestmove {}", best_move);

        Ok(())
    })
    // Flags for reporting the current time situation.
    .flag("binc", Flag::Single)
    .flag("winc", Flag::Single)
    .flag("btime", Flag::Single)
    .flag("wtime", Flag::Single)
    .flag("movestogo", Flag::Single)
    // Flags for imposing other search limits.
    .flag("depth", Flag::Single)
    .flag("nodes", Flag::Single)
    .flag("movetime", Flag::Single)
    // Flags for setting the search type.
    // .flag("ponder", Flag::Single)
    .flag("infinite", Flag::Single)
    // This command should be run in a separate thread so that the Client
    // can still respond to and run other Commands while this one is running.
    .parallelize(true)
}

#[allow(dead_code)]
struct StandardTC {
    btime: time::Duration,
    wtime: time::Duration,
    binc: time::Duration,
    winc: time::Duration,
    movestogo: Option<u8>,
}

fn search_std(position: ataxx::Position, tc: StandardTC) -> ataxx::Move {
    let our_time = match position.side_to_move {
        ataxx::Piece::Black => tc.btime,
        ataxx::Piece::White => tc.wtime,
        _ => unreachable!(),
    };

    let our_inc = match position.side_to_move {
        ataxx::Piece::Black => tc.binc,
        ataxx::Piece::White => tc.winc,
        _ => unreachable!(),
    };

    let movetime = our_time / tc.movestogo.unwrap_or(20) as u32 + our_inc / 2;

    let mut tree = mcts::Tree::new(position);
    let start = time::Instant::now();
    let mut last_info = start;
    let mut seldepth = 0;

    loop {
        let node = tree.playout();
        if tree.playouts() & 4095 == 0 {
            if last_info.elapsed() > time::Duration::from_secs(1) {
                // Update last info report timestamp.
                last_info = time::Instant::now();

                let node = tree.node(node);
                let new_depth = node.position.ply_count - position.ply_count;
                if new_depth > seldepth {
                    seldepth = new_depth;
                }

                // Make a new info report.
                println!(
                    "info depth {} seldepth {} score {:.0} nodes {} nps {}",
                    new_depth,
                    seldepth,
                    node.q() * 100.0,
                    tree.playouts(),
                    tree.nodes() * 1000 / start.elapsed().as_millis().max(1) as usize
                );
            }

            //
            if start.elapsed() > movetime {
                break;
            }
        }
    }

    tree.best_move()
}

struct InternalTC {
    depth: Option<u16>,
    nodes: Option<usize>,
    movetime: Option<time::Duration>,
}

fn search_int(position: ataxx::Position, tc: InternalTC) -> ataxx::Move {
    let mut tree = mcts::Tree::new(position);
    let start = time::Instant::now();
    let mut last_info = start;
    let mut seldepth = 0;

    loop {
        let node = tree.playout();
        if tree.playouts() & 4095 == 0 {
            if last_info.elapsed() > time::Duration::from_secs(1) {
                // Update last info report timestamp.
                last_info = time::Instant::now();

                let node = tree.node(node);
                let new_depth = node.position.ply_count - position.ply_count;
                if new_depth > seldepth {
                    seldepth = new_depth;
                }

                // Make a new info report.
                println!(
                    "info depth {} seldepth {} score {:.0} nodes {} nps {}",
                    new_depth,
                    seldepth,
                    node.q() * 100.0,
                    tree.playouts(),
                    tree.nodes() * 1000 / start.elapsed().as_millis().max(1) as usize
                );
            }

            //
            if let Some(nodes) = tc.nodes {
                if tree.nodes() > nodes {
                    break;
                }
            }

            if let Some(depth) = tc.depth {
                if seldepth > depth {
                    break;
                }
            }

            if let Some(movetime) = tc.movetime {
                if start.elapsed() > movetime {
                    break;
                }
            }
        }
    }

    tree.best_move()
}