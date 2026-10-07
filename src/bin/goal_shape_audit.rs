//! Diagnose exact three-ply mating shapes in recorded positions.
use sailfish2::position::Position;
use sailfish2::types::*;
use std::{fs,io::{Write,BufWriter}};
fn main(){let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),3);let mut out=BufWriter::new(fs::File::create(&a[2]).unwrap());writeln!(out,"row\tmove\treplies\tquiet\tsee\tmate3\tactual").unwrap();let mut positions=0;
for (id,line) in fs::read_to_string(&a[1]).unwrap().lines().enumerate(){let c:Vec<_>=line.split('\t').collect();let mut p=Position::from_sfen(c[0]).unwrap();let actual=c[1];let moves=p.legal_moves();let before=(p.board,p.hand,p.side,p.key);for m in moves{if !p.gives_check(m){continue;}let quiet=mv_is_drop(m)||p.board[mv_to(m)]==0;let see=p.see(m);assert!(p.do_move_legal(m));let replies=p.legal_moves();let count=replies.len();let mut mate3=count==0;
if count>0&&count<=2{mate3=true;for reply in replies{assert!(p.do_move_legal(reply));let mate=p.mate_in_one().is_some();p.undo_move();if !mate{mate3=false;break;}}}p.undo_move();assert!(before==(p.board,p.hand,p.side,p.key));if count<=2{writeln!(out,"{}\t{}\t{}\t{}\t{}\t{}\t{}",id,move_to_usi(m),count,quiet as u8,see,mate3 as u8,actual).unwrap();}}
positions+=1;}println!("audited {} positions",positions);}
