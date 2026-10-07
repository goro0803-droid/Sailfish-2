//! Audit independent teacher rows without modifying the network or searching.
use sailfish2::{nn::{Network, Acc}, eval::evaluate, position::Position,
    training::{eligible_sample, TrainingFilter}};
use std::{fs, io::{Write, BufWriter}};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 4, "nn_audit network.nn positions.tsv output.tsv");
    let net = Network::from_bytes(&fs::read(&a[1]).unwrap()).unwrap();
    let mut out = BufWriter::new(fs::File::create(&a[3]).unwrap());
    writeln!(out, "teacher\tresult\tnn\thandcrafted\tside\tply").unwrap();
    let mut acc = Acc::default();
    let filter = TrainingFilter { quiet_drops: true, skip_opening: false };
    let mut count = 0;
    for line in fs::read_to_string(&a[2]).unwrap().lines() {
        let c: Vec<_> = line.split('\t').collect();
        assert_eq!(c.len(), 6); assert_eq!(c[0], "P");
        let mut pos = Position::from_sfen(c[1]).unwrap();
        let score: f32 = c[2].parse().unwrap();
        let result: f32 = c[3].parse().unwrap();
        let m = pos.parse_usi_move(c[4]).unwrap();
        assert!(pos.do_move_legal(m)); pos.undo_move();
        if !eligible_sample(&mut pos, m, score, 0, 0, filter) { continue; }
        net.refresh(&pos, &mut acc);
        writeln!(out, "{}\t{}\t{}\t{}\t{}\t{}", score, result,
            net.forward(&acc, &pos), evaluate(&pos), pos.side, c[5]).unwrap();
        count += 1;
    }
    println!("audited {} legal, eligible positions", count);
}
