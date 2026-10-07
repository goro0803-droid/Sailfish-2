//! floodgate_import <wdoor.7z> <out.txt> [games=24000]
//! Approved external game positions; external evaluation comments are ignored.
use sailfish2::position::Position;
use sailfish2::types::*;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
fn piece(s: &str) -> Option<u8> {
    Some(match s {
        "FU" => PAWN,
        "KY" => LANCE,
        "KE" => KNIGHT,
        "GI" => SILVER,
        "KI" => GOLD,
        "KA" => BISHOP,
        "HI" => ROOK,
        "OU" => KING,
        "TO" => PRO_PAWN,
        "NY" => PRO_LANCE,
        "NK" => PRO_KNIGHT,
        "NG" => PRO_SILVER,
        "UM" => HORSE,
        "RY" => DRAGON,
        _ => return None,
    })
}
fn csa_move(p: &Position, s: &str) -> Result<String, &'static str> {
    let b = s.as_bytes();
    if b.len() != 7 || !b.is_ascii() {
        return Err("bad_move");
    }
    if b[0] != if p.side == BLACK { b'+' } else { b'-' } {
        return Err("wrong_side");
    }
    let pt = piece(&s[5..]).ok_or("bad_piece")?;
    if !(b'1'..=b'9').contains(&b[3]) || !(b'1'..=b'9').contains(&b[4]) {
        return Err("bad_square");
    }
    let to = format!("{}{}", b[3] as char, (b'a' + b[4] - b'1') as char);
    if &s[1..3] == "00" {
        let c = match pt {
            PAWN => 'P',
            LANCE => 'L',
            KNIGHT => 'N',
            SILVER => 'S',
            GOLD => 'G',
            BISHOP => 'B',
            ROOK => 'R',
            _ => return Err("bad_drop"),
        };
        return Ok(format!("{}*{}", c, to));
    }
    if !(b'1'..=b'9').contains(&b[1]) || !(b'1'..=b'9').contains(&b[2]) {
        return Err("bad_square");
    }
    let pc = p.board[((b[1] - b'1') * 9 + (b[2] - b'1')) as usize];
    if pc == 0 || color_of(pc) != p.side {
        return Err("bad_source");
    }
    let ty = type_of(pc);
    let promo = if pt == ty {
        ""
    } else if can_promote(ty) && pt == ty + PROMOTE {
        "+"
    } else {
        return Err("bad_promotion");
    };
    Ok(format!(
        "{}{}{}{}",
        b[1] as char,
        (b'a' + b[2] - b'1') as char,
        to,
        promo
    ))
}
fn parse_game(lines: &[String]) -> Result<(String, Vec<String>, f32), &'static str> {
    let mut p = Position::startpos();
    let initial = p.board;
    let mut rows = [false; 9];
    let mut pi = false;
    let mut started = false;
    let mut event = String::new();
    let mut moves = Vec::new();
    let mut end = None;
    let codes = [
        "", "FU", "KY", "KE", "GI", "KA", "HI", "KI", "OU", "TO", "NY", "NK", "NG", "UM", "RY",
    ];
    for raw in lines {
        let l = raw.trim();
        if let Some(s) = l.strip_prefix("$EVENT:") {
            event = s.to_owned();
        }
        if l.starts_with('\'')
            || l.starts_with('$')
            || l.starts_with('N')
            || l.starts_with('T')
            || l.starts_with('V')
            || l.is_empty()
        {
            continue;
        }
        if l == "PI" && !started {
            pi = true;
            continue;
        }
        if l.starts_with("PI") {
            return Err("nonstandard_start");
        }
        if l.len() >= 2 && l.starts_with('P') && l.as_bytes()[1].is_ascii_digit() {
            if started {
                return Err("nonstandard_start");
            }
            let r = l.as_bytes()[1].wrapping_sub(b'1') as usize;
            if r >= 9 || rows[r] {
                return Err("bad_rows");
            }
            let compact = l[2..]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>();
            let mut expected = String::new();
            for f in (0..9).rev() {
                let pc = initial[f * 9 + r];
                if pc == 0 {
                    expected.push('*');
                } else {
                    expected.push(if color_of(pc) == BLACK { '+' } else { '-' });
                    expected.push_str(codes[type_of(pc) as usize]);
                }
            }
            if compact != expected {
                return Err("nonstandard_start");
            }
            rows[r] = true;
            continue;
        }
        if l.starts_with("P+") || l.starts_with("P-") {
            if l.len() != 2 {
                return Err("nonstandard_start");
            }
            continue;
        }
        if l == "+" || l == "-" {
            if started || l != "+" || !(pi || rows.iter().all(|&v| v)) {
                return Err("nonstandard_start");
            }
            started = true;
            continue;
        }
        if l.starts_with('+') || l.starts_with('-') {
            if !started || end.is_some() {
                return Err("bad_sequence");
            }
            let usi = csa_move(&p, l)?;
            let m = p.parse_usi_move(&usi).ok_or("illegal_move")?;
            p.do_move(m);
            moves.push(usi);
            continue;
        }
        if l.starts_with('%') {
            if end.is_some() {
                return Err("multiple_terminal");
            }
            end = Some(match l {
                "%TORYO" => {
                    if p.side == BLACK {
                        0.0
                    } else {
                        1.0
                    }
                }
                "%SENNICHITE" | "%JISHOGI" => 0.5,
                _ => return Err("excluded_terminal"),
            });
            continue;
        }
        return Err("unknown_record");
    }
    if !started || event.is_empty() {
        return Err("missing_header");
    }
    if !event.starts_with("wdoor+floodgate-") {
        return Err("not_floodgate");
    }
    Ok((event, moves, end.ok_or("unfinished")?))
}
fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn run() -> Result<(), String> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        return Err("floodgate_import <wdoor.7z> <out.txt> [games=24000]".into());
    }
    let limit: usize = a
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "games")?
        .unwrap_or(24000);
    if limit == 0 || limit > 100000 {
        return Err("invalid sample limit".into());
    }
    let manifest = format!("{}.manifest.tsv", a[1]);
    if std::path::Path::new(&a[1]).exists() || std::path::Path::new(&manifest).exists() {
        return Err("output exists".into());
    }
    let mut cmd = Command::new("tar");
    cmd.args(["-xOf", &a[0]])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let mut buf = BufReader::new(child.stdout.take().ok_or("stdout")?);
    let mut bytes = Vec::new();
    let mut game = Vec::new();
    let mut seen = HashSet::new();
    let mut reasons = BTreeMap::new();
    let mut reservoir = Vec::new();
    let mut seed = 915172331u64;
    let mut total = 0usize;
    let mut eligible = 0usize;
    let started = std::time::Instant::now();
    let mut finish = |game: &[String]| {
        if game.is_empty() {
            return;
        }
        total += 1;
        let result = parse_game(game).and_then(|(id, moves, res)| {
            if moves.len() < 40 || moves.len() > 512 {
                return Err("length");
            }
            if !seen.insert(moves.join(" ")) {
                return Err("duplicate");
            }
            Ok((id, moves, res))
        });
        match result {
            Ok((id, moves, res)) => {
                eligible += 1;
                // Placeholder zeros must be replaced by Sailfish teacher searches before training.
                let line = format!(
                    "R {} 20 {}",
                    res,
                    moves
                        .iter()
                        .take(180)
                        .map(|m| format!("{}:0", m))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let row = (id, line);
                if reservoir.len() < limit {
                    reservoir.push(row);
                } else {
                    let j = (rng(&mut seed) % eligible as u64) as usize;
                    if j < limit {
                        reservoir[j] = row;
                    }
                }
            }
            Err(r) => {
                *reasons.entry(r).or_insert(0usize) += 1;
            }
        }
        if total % 10000 == 0 {
            println!(
                "read {} eligible {} {:.0}s",
                total,
                eligible,
                started.elapsed().as_secs_f64()
            );
        }
    };
    loop {
        bytes.clear();
        let n = buf
            .read_until(b'\n', &mut bytes)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        let l = String::from_utf8_lossy(&bytes).trim_end().to_owned();
        if l.starts_with("V2") {
            finish(&game);
            game.clear();
        }
        if !l.is_empty() {
            game.push(l);
        }
    }
    finish(&game);
    if !child.wait().map_err(|e| e.to_string())?.success() {
        return Err("archive read failed".into());
    }
    if reservoir.is_empty() {
        return Err(format!("no valid games: {:?}", reasons));
    }
    for i in (1..reservoir.len()).rev() {
        let j = (rng(&mut seed) % (i as u64 + 1)) as usize;
        reservoir.swap(i, j);
    }
    fs::write(
        &a[1],
        reservoir
            .iter()
            .map(|r| r.1.as_str())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        &manifest,
        "row\tevent\n".to_owned()
            + &reservoir
                .iter()
                .enumerate()
                .map(|(i, r)| format!("{}\t{}", i, r.0))
                .collect::<Vec<_>>()
                .join("\n")
            + "\n",
    )
    .map_err(|e| e.to_string())?;
    println!(
        "Complete total={} eligible={} selected={} {:.0}s excluded={:?}",
        total,
        eligible,
        reservoir.len(),
        started.elapsed().as_secs_f64(),
        reasons
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conversion_checks_piece_side_and_square() {
        let p = Position::startpos();
        assert_eq!(csa_move(&p, "+7776FU"), Ok("7g7f".into()));
        assert!(csa_move(&p, "-7776FU").is_err());
        assert!(csa_move(&p, "+7776HI").is_err());
        assert_eq!(csa_move(&p, "+0055KA"), Ok("B*5e".into()));
        assert!(csa_move(&p, "+0055UM").is_err());
        assert!(csa_move(&p, "+7770FU").is_err());
    }
    #[test]
    fn outcome_legality_and_external_score_exclusion() {
        let lines = [
            "V2",
            "$EVENT:wdoor+floodgate-test",
            "PI",
            "+",
            "+7776FU",
            "'** 99999",
            "-3334FU",
            "%TORYO",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let (_, m, r) = parse_game(&lines).unwrap();
        assert_eq!(m, vec!["7g7f", "3c3d"]);
        assert_eq!(r, 0.0);
        let mut foreign = lines.clone();
        foreign[1] = "$EVENT:wdoor+other-test".into();
        assert_eq!(parse_game(&foreign), Err("not_floodgate"));
        let mut white = lines.clone();
        white.remove(6);
        assert_eq!(parse_game(&white).unwrap().2, 1.0);
        let mut bad = lines.clone();
        *bad.last_mut().unwrap() = "%TIME_UP".into();
        assert_eq!(parse_game(&bad), Err("excluded_terminal"));
        let mut bad = lines.clone();
        bad[4] = "+7776TO".into();
        assert_eq!(parse_game(&bad), Err("illegal_move"));
        let mut bad = lines.clone();
        bad.pop();
        assert_eq!(parse_game(&bad), Err("unfinished"));
    }
}
