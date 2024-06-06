use uxi::{error, Bundle, Command, Flag, RunError};

use super::Context;
use crate::mcts;

pub fn go() -> Command<Context> {
    Command::new(|bundle: Bundle<Context>| {
        let binc = bundle.is_flag_set("binc");
        let winc = bundle.is_flag_set("winc");
        let btime = bundle.is_flag_set("btime");
        let wtime = bundle.is_flag_set("wtime");

        let depth = bundle.is_flag_set("depth");
        let nodes = bundle.is_flag_set("nodes");
        let movetime = bundle.is_flag_set("movetime");

        let infinite = bundle.is_flag_set("infinite");

        let std_tc = btime || wtime || binc || winc;

        if std_tc && !(btime && wtime && binc && winc) {
            return error!("bad flag set: missing standard time control flags");
        }

        if infinite && (std_tc || depth || nodes || movetime) {
            return error!("bad flag set: time control flags set alongside infinite");
        }

        let ctx = bundle.lock();
        let position = ctx.position;
        let mut searcher = ctx.searcher.clone();
        drop(ctx);

        let limits = mcts::Limits {
            maxnodes: match bundle.get_single_flag("nodes") {
                Some(nodes) => Some(nodes.parse()?),
                None => None,
            },
            maxdepth: match bundle.get_single_flag("depth") {
                Some(depth) => Some(depth.parse()?),
                None => None,
            },
            movetime: {
                let mov_time = match bundle.get_single_flag("movetime") {
                    Some(movetime) => movetime.parse()?,
                    None => u128::MAX,
                };

                if std_tc {
                    macro_rules! get_flag {
                        ($name:literal) => {
                            bundle.get_single_flag($name).unwrap().parse()?
                        };
                    }

                    let our_time: u128 = match position.side_to_move {
                        ataxx::Piece::Black => get_flag!("wtime"),
                        ataxx::Piece::White => get_flag!("btime"),
                        _ => unreachable!(),
                    };

                    let our_inc: u128 = match position.side_to_move {
                        ataxx::Piece::Black => get_flag!("winc"),
                        ataxx::Piece::White => get_flag!("binc"),
                        _ => unreachable!(),
                    };

                    let usable_time = (our_time / 20 + our_inc / 2).max(1);

                    Some(usable_time.min(mov_time))
                } else if movetime {
                    Some(mov_time)
                } else {
                    None
                }
            },

            movestogo: match bundle.get_single_flag("movestogo") {
                Some(string) => Some(string.parse()?),
                None => None,
            },
        };

        searcher.update_position(position);

        let mut nodes = 0;
        let bestmove = searcher.search(limits, &mut nodes);

        println!("bestmove {}", bestmove);

        let mut ctx = bundle.lock();
        ctx.searcher = searcher;
        drop(ctx);

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
