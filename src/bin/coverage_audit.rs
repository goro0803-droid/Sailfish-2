//! Descriptive coverage audit. No search, learning, or network replacement.
use sailfish2::{position::Position,types::*,training::{eligible_sample,TrainingFilter},nn::{Network,Acc},eval::PIECE_VALUE};
use std::{fs,io::{Write,BufWriter}};
fn ring(pos:&Position,us:usize)->usize{
 let k=pos.king[us];if k>=81{return 0;}let mut n=0;
 for sq in 0..81{if sq!=k&&(sq/9).abs_diff(k/9)<=1&&(sq%9).abs_diff(k%9)<=1&&pos.attacked(sq,us^1){n+=1;}}n
}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),4,"coverage_audit network.nn input.tsv output.tsv");
 let net=Network::from_bytes(&fs::read(&a[1]).unwrap()).unwrap();let mut acc=Acc::default();
 let mut out=BufWriter::new(fs::File::create(&a[3]).unwrap());
 writeln!(out,"id\tscore\tresult\tply\tside\tin_check\traw_eligible\ttrain_eligible\tcapture_move\tpromo_move\tdrop_move\tgives_check\tlegal\tcaptures\tchecks\tpromos\tdrops\town_hand\topp_hand\town_major_hand\topp_major_hand\tpromoted\town_ring\topp_ring\tloose_major\tloose_minor\tmaterial\tnn").unwrap();
 let mut count=0;
 for (id,line) in fs::read_to_string(&a[2]).unwrap().lines().enumerate(){
  let c:Vec<_>=line.split('\t').collect();assert_eq!(c.len(),6);assert_eq!(c[0],"P");
  let mut p=Position::from_sfen(c[1]).unwrap();let score:f32=c[2].parse().unwrap();let result:f32=c[3].parse().unwrap();let ply:usize=c[5].parse().unwrap();let m=p.parse_usi_move(c[4]).unwrap();
  let us=p.side;let chk=p.in_check();let cap=!mv_is_drop(m)&&p.board[mv_to(m)]!=0;assert!(p.do_move_legal(m),"illegal row {}",id);let gives=p.in_check();p.undo_move();
  let raw=ply>=16&&score.abs()<=2000.0&&!chk;
  let train=raw&&eligible_sample(&mut p,m,score,ply,0,TrainingFilter{quiet_drops:true,skip_opening:false});
  let legal=p.legal_moves();let mut captures=0;let mut checks=0;let mut promos=0;let mut drops=0;
  for &q in &legal{if !mv_is_drop(q)&&p.board[mv_to(q)]!=0{captures+=1;}if mv_is_promo(q){promos+=1;}if mv_is_drop(q){drops+=1;}p.do_move(q);if p.in_check(){checks+=1;}p.undo_move();}
  let own_hand:u32=p.hand[us].iter().map(|&v|v as u32).sum();let opp_hand:u32=p.hand[us^1].iter().map(|&v|v as u32).sum();
  let own_major=p.hand[us][BISHOP as usize]+p.hand[us][ROOK as usize];let opp_major=p.hand[us^1][BISHOP as usize]+p.hand[us^1][ROOK as usize];
  let mut promoted=0;let mut loose_major=0;let mut loose_minor=0;let mut material=0;
  for sq in 0..81{let v=p.board[sq];if v==0{continue;}let pt=type_of(v);let col=color_of(v);if pt>KING{promoted+=1;}if pt!=KING{material+=if col==us{PIECE_VALUE[pt as usize]}else{-PIECE_VALUE[pt as usize]};}
   if col==us&&pt!=KING&&p.attacked(sq,us^1)&&!p.attacked(sq,us){if matches!(unpromote(pt),BISHOP|ROOK){loose_major+=1;}else if PIECE_VALUE[pt as usize]>=PIECE_VALUE[KNIGHT as usize]{loose_minor+=1;}}
  }
  for col in 0..2{for pt in 1..8{material+=(if col==us{1}else{-1})*p.hand[col][pt] as i32*PIECE_VALUE[pt];}}
  net.refresh(&p,&mut acc);let nn=net.forward(&acc,&p);
  let vals=vec![id as i64,score as i64,(result*2.0) as i64,ply as i64,us as i64,chk as i64,raw as i64,train as i64,cap as i64,mv_is_promo(m) as i64,mv_is_drop(m) as i64,gives as i64,legal.len() as i64,captures,checks,promos,drops,own_hand as i64,opp_hand as i64,own_major as i64,opp_major as i64,promoted,ring(&p,us) as i64,ring(&p,us^1) as i64,loose_major,loose_minor,material as i64,nn as i64];
  writeln!(out,"{}",vals.iter().map(|v|v.to_string()).collect::<Vec<_>>().join("\t")).unwrap();count+=1;
 }
 println!("coverage audited {} legal positions",count);
}
