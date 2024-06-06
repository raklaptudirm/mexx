use std::{env, str::FromStr, time};
use uxi::Client;

mod commands;
mod mcts;
mod options;

#[rustfmt::skip]
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        if args.len() != 2 || args[1] != "bench" {
            eprintln!("error: unexpected number of command line arguments");
            return;
        }

        const BENCH_FENS: &[&str] = &[
            "x-1-1-o/-1-1-1-/1-1-1-1/-1-1-1-/1-1-1-1/-1-1-1-/o-1-1-x x 0 1",
            // "x-1-1-o/1-1-1-1/1-1-1-1/1-1-1-1/1-1-1-1/1-1-1-1/o-1-1-x x 0 1",
            "x1-1-1o/2-1-2/-------/2-1-2/-------/2-1-2/o1-1-1x x 0 1",
            // "x5o/1-----1/1-3-1/1-1-1-1/1-3-1/1-----1/o5x x 0 1",
            "x-1-1-o/1-1-1-1/-1-1-1-/-1-1-1-/-1-1-1-/1-1-1-1/o-1-1-x x 0 1",
            "x5o/1--1--1/1--1--1/7/1--1--1/1--1--1/o5x x 0 1",
            // "x-3-o/1-1-1-1/1-1-1-1/3-3/1-1-1-1/1-1-1-1/o-3-x x 0 1",
            // "x2-2o/3-3/3-3/-------/3-3/3-3/o2-2x x 0 1",
            // "x2-2o/2-1-2/1-3-1/-2-2-/1-3-1/2-1-2/o2-2x x 0 1",
            "x5o/7/7/7/7/7/o5x x 0 1",
            "x5o/7/2-1-2/7/2-1-2/7/o5x x 0 1",
            "x5o/7/3-3/2-1-2/3-3/7/o5x x 0 1",
            "x2-2o/3-3/2---2/7/2---2/3-3/o2-2x x 0 1",
            "x2-2o/3-3/7/--3--/7/3-3/o2-2x x 0 1",
            "x1-1-1o/2-1-2/2-1-2/7/2-1-2/2-1-2/o1-1-1x x 0 1",
            // "x5o/7/2-1-2/3-3/2-1-2/7/o5x x 0 1",
            // "x5o/7/3-3/2---2/3-3/7/o5x x 0 1",
            "x5o/2-1-2/1-3-1/7/1-3-1/2-1-2/o5x x 0 1",
            "x5o/1-3-1/2-1-2/7/2-1-2/1-3-1/o5x x 0 1",
            "2x3o/7/7/7/o6/5x1/6x o 2 2",
            "5oo/7/x6/x6/7/7/o5x o 0 2",
            "x5o/1x5/7/7/7/2o4/4x2 o 0 2",
            "7/7/2x1o2/1x5/7/7/o5x o 0 2",
            "7/7/1x4o/7/4x2/7/o6 o 3 2",
            "x5o/7/6x/7/1o5/7/7 o 3 2",
            "5oo/7/2x4/7/7/4x2/o6 o 1 2",
            "x5o/7/7/3x3/7/1o5/o6 o 1 2",
            "x5o/7/7/7/7/2x1x2/3x3 o 0 2",
            "7/7/1x4o/7/7/4x2/o6 o 3 2",
            "x5o/7/7/5x1/5x1/1o5/o6 o 0 2",
            "6o/7/4x2/7/7/1o5/o5x o 1 2",
            "x5o/x5o/7/7/7/6x/o5x o 0 2",
            "4x1o/7/7/7/7/o6/o5x o 1 2",
            "6o/7/x6/7/7/2o4/6x o 3 2",
            "x5o/7/7/7/1o4x/7/5x1 o 2 2",
            "x5o/6o/7/7/4x2/7/o6 o 1 2",
            "7/7/1xx1o2/7/7/7/o5x o 0 2",
            "2x3o/2x4/7/7/7/7/2o3x o 0 2",
            "x5o/6o/7/7/4x2/3x3/o6 o 0 2",
            "x5o/7/7/7/o3xx1/7/7 o 0 2",
            "6o/6o/1x5/7/4x2/7/o6 o 1 2",
            "7/7/4x1o/7/7/7/o5x o 3 2",
            "4o2/7/2x4/7/7/7/o4xx o 0 2",
            "2x3o/x6/7/7/7/o6/o5x o 1 2",
            "6o/7/2x4/7/1o5/7/4x2 o 3 2",
            "x6/4o2/7/7/6x/7/o6 o 3 2",
            "x6/7/5o1/7/7/4x2/o6 o 3 2",
            "x5o/1x4o/7/7/7/7/o3x2 o 0 2",
            "xx4o/7/7/7/7/6x/oo4x o 0 2",
            "x6/7/4x2/3x3/7/7/o5x o 2 2",
        ];

        let mut total_nodes = 0;

        let start = time::Instant::now();
        for (i, fen) in BENCH_FENS.iter().enumerate() {
            println!("[#{}] {}", i + 1, fen);
            let position = ataxx::Position::from_str(fen).unwrap();
            let mut searcher = mcts::Searcher::new(position, mcts::policy::handcrafted, mcts::value::material);
            let limits = mcts::Limits {
                maxnodes: Some(50000),
                maxdepth: Some(10),
                movetime: None,
                movestogo: None,
            };

            searcher.search(limits, &mut total_nodes);
        }
        let elapsed = start.elapsed().as_millis();

        // Assert that the node-count hasn't changed unexpectedly.
        debug_assert!(total_nodes == 261787);

        println!("nodes {} nps {}", total_nodes, total_nodes as u128 * 1000 / elapsed);

        return
    }

    Client::new()
        .protocol("uai")
        .engine("Mexx v0.0.0")
        .author("Rak Laptudirm")
        // Register engine options.
        .option("Hash",    options::hash   ())
        .option("Threads", options::threads())
        // Register the custom commands.
        .command(         "d", commands::d())
        .command(        "go", commands::go())
        .command(  "protocol", commands::protocol())
        .command(  "position", commands::position())
        .command("uainewgame", commands::uainewgame())
        // Start the Client so it can start running Commands.
        .start(Default::default());
}
