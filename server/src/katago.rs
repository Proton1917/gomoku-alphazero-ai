//! KataGo GTP 子进程管理：按会话惰性启动，落子前重放完整手顺。

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::config::config;
use crate::rules::{BOARD_SIZE, PASS_MOVE, RESIGN_MOVE};
use crate::session::MoveRecord;

const GTP_COLUMNS: &str = "ABCDEFGHJKLMNOPQRST";

pub struct KataGoProcess {
    model_path: PathBuf,
    max_visits: u32,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    output_rx: Option<Receiver<String>>,
    /// 上一手的真实搜索访问数（来自 kata-genmove_analyze 的 rootInfo）
    pub last_search_visits: u32,
    /// 上一手搜索耗时（毫秒），用于计算 visits/s 算力
    pub last_search_millis: u64,
    /// KataGo 是否支持 kata-genmove_analyze（老版本回退 genmove）
    analyze_supported: bool,
}

impl KataGoProcess {
    pub fn new(model_path: &str, max_visits: i64) -> Self {
        Self {
            model_path: PathBuf::from(model_path),
            max_visits: max_visits.clamp(1, 2000) as u32,
            child: None,
            stdin: None,
            output_rx: None,
            last_search_visits: 0,
            last_search_millis: 0,
            analyze_supported: true,
        }
    }

    pub fn choose_move(
        &mut self,
        current_player: i32,
        move_history: &[MoveRecord],
    ) -> Result<(i32, i32), String> {
        self.ensure_running()?;
        self.load_position(move_history)?;
        let color = if current_player == 1 { "B" } else { "W" };

        let started = Instant::now();
        let (mv, visits) = if self.analyze_supported {
            // kata-genmove_analyze 流式输出分析行，最后一行 play <move>；
            // rootInfo 携带根节点真实 visits。interval 100 = 每 1s 一帧，控制流量。
            let command = format!("kata-genmove_analyze {color} interval 100 rootInfo true");
            match self.command(&command, Duration::from_secs(120)) {
                Ok(response) => parse_analyze_response(&response)?,
                Err(e) if e.contains("拒绝命令") => {
                    // 老版本 KataGo 不认识该命令：本会话内永久回退到 genmove
                    self.analyze_supported = false;
                    let response =
                        self.command(&format!("genmove {color}"), Duration::from_secs(120))?;
                    (parse_gtp_move(response.trim())?, self.max_visits)
                }
                Err(e) => return Err(e),
            }
        } else {
            let response = self.command(&format!("genmove {color}"), Duration::from_secs(120))?;
            (parse_gtp_move(response.trim())?, self.max_visits)
        };

        self.last_search_millis = started.elapsed().as_millis() as u64;
        self.last_search_visits = visits;
        Ok(mv)
    }

    fn ensure_running(&mut self) -> Result<(), String> {
        if let Some(child) = self.child.as_mut() {
            if child.try_wait().map_err(|e| e.to_string())?.is_none() {
                return Ok(());
            }
        }

        let cfg = config();
        if !cfg.katago_executable.exists() {
            return Err(format!(
                "未找到 KataGo 可执行文件: {}",
                cfg.katago_executable.display()
            ));
        }
        if !cfg.katago_config_path.exists() {
            return Err(format!(
                "未找到 KataGo GTP 配置: {}",
                cfg.katago_config_path.display()
            ));
        }
        if !self.model_path.exists() {
            return Err(format!("未找到 KataGo 模型: {}", self.model_path.display()));
        }

        let mut child = Command::new(&cfg.katago_executable)
            .args([
                "gtp",
                "-model",
                &self.model_path.to_string_lossy(),
                "-config",
                &cfg.katago_config_path.to_string_lossy(),
                "-override-config",
                &format!(
                    "maxVisits={},logAllGTPCommunication=false",
                    self.max_visits
                ),
            ])
            .current_dir(cfg.katago_executable.parent().unwrap_or(&cfg.repo_root))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("启动 KataGo 失败: {e}"))?;

        let stdout = child.stdout.take().ok_or("无法获取 KataGo stdout")?;
        let stderr = child.stderr.take().ok_or("无法获取 KataGo stderr")?;
        let stdin = child.stdin.take().ok_or("无法获取 KataGo stdin")?;

        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let tx_err = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        // stderr 与 stdout 合流，与 Python 版 stderr=STDOUT 行为一致
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx_err.send(line).is_err() {
                    break;
                }
            }
        });

        self.child = Some(child);
        self.stdin = Some(stdin);
        self.output_rx = Some(rx);

        self.command("boardsize 19", Duration::from_secs(60))?;
        self.command("komi 6.5", Duration::from_secs(10))?;
        Ok(())
    }

    fn load_position(&mut self, move_history: &[MoveRecord]) -> Result<(), String> {
        self.command("clear_board", Duration::from_secs(10))?;
        for record in move_history {
            let color = if record.player == 1 { "B" } else { "W" };
            let coord = if record.mv == PASS_MOVE {
                "pass".to_string()
            } else if record.mv == RESIGN_MOVE {
                break;
            } else {
                board_to_gtp(record.mv[0], record.mv[1])?
            };
            self.command(&format!("play {color} {coord}"), Duration::from_secs(10))?;
        }
        Ok(())
    }

    fn command(&mut self, command: &str, timeout: Duration) -> Result<String, String> {
        let stdin = self.stdin.as_mut().ok_or("KataGo 进程未启动")?;
        let rx = self.output_rx.as_ref().ok_or("KataGo 进程未启动")?;

        stdin
            .write_all(format!("{command}\n").as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("向 KataGo 写入命令失败: {e}"))?;

        let deadline = Instant::now() + timeout;
        let mut started = false;
        let mut status = ' ';
        let mut payload: Vec<String> = Vec::new();
        let mut noise_tail: VecDeque<String> = VecDeque::new();

        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = match rx.recv_timeout(remaining.min(Duration::from_millis(200))) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    if let Some(child) = self.child.as_mut() {
                        if child.try_wait().ok().flatten().is_some() {
                            return Err(format!("KataGo 已退出，命令失败: {command}"));
                        }
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(format!("KataGo 输出流已关闭，命令失败: {command}"));
                }
            };
            let clean = line.trim().to_string();

            if clean.starts_with('=') || clean.starts_with('?') {
                started = true;
                status = clean.chars().next().unwrap();
                let first_payload = clean[1..].trim().to_string();
                if !first_payload.is_empty() {
                    payload.push(first_payload);
                }
                continue;
            }

            if started {
                if clean.is_empty() {
                    if status == '?' {
                        return Err(format!("KataGo 拒绝命令 {command}: {}", payload.join(" ")));
                    }
                    return Ok(payload.join("\n"));
                }
                payload.push(clean);
                continue;
            }

            if !clean.is_empty() {
                noise_tail.push_back(clean);
                while noise_tail.len() > 12 {
                    noise_tail.pop_front();
                }
            }
        }

        self.close();
        Err(format!(
            "KataGo 响应超时: {command}; tail={:?}",
            noise_tail.iter().collect::<Vec<_>>()
        ))
    }

    pub fn close(&mut self) {
        if let Some(mut stdin) = self.stdin.take() {
            let _ = stdin.write_all(b"quit\n");
            let _ = stdin.flush();
        }
        if let Some(mut child) = self.child.take() {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    _ => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                }
            }
        }
        self.output_rx = None;
    }
}

impl Drop for KataGoProcess {
    fn drop(&mut self) {
        self.close();
    }
}

pub fn board_to_gtp(row: i32, col: i32) -> Result<String, String> {
    if !crate::rules::is_on_board(row, col) {
        return Err(format!("坐标超出 19 路棋盘: {row}, {col}"));
    }
    let column = GTP_COLUMNS.as_bytes()[col as usize] as char;
    Ok(format!("{column}{}", BOARD_SIZE as i32 - row))
}

/// 解析 kata-genmove_analyze 的完整响应：
/// 取最后一帧分析行提取真实 visits，取 `play <move>` 行提取着手。
fn parse_analyze_response(raw: &str) -> Result<((i32, i32), u32), String> {
    let mut mv: Option<(i32, i32)> = None;
    let mut last_analysis: Option<&str> = None;

    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(vertex) = trimmed.strip_prefix("play ") {
            mv = Some(parse_gtp_move(vertex.trim())?);
        } else if trimmed.contains("info move") || trimmed.contains("rootInfo") {
            last_analysis = Some(trimmed);
        }
    }

    let mv = mv.ok_or("kata-genmove_analyze 未返回 play 行")?;
    let visits = last_analysis.map(extract_root_visits).unwrap_or(0);
    Ok((mv, visits))
}

/// 从一帧分析行中提取根节点 visits：优先 rootInfo，否则累加各候选着法。
fn extract_root_visits(line: &str) -> u32 {
    if let Some(idx) = line.find("rootInfo") {
        if let Some(v) = number_after_token(&line[idx..], "visits") {
            return v;
        }
    }
    line.split("info move")
        .skip(1)
        .filter_map(|segment| {
            // 候选段可能包含 rootInfo 尾巴，截断避免重复计数
            let segment = segment.split("rootInfo").next().unwrap_or(segment);
            number_after_token(segment, "visits")
        })
        .sum()
}

fn number_after_token(text: &str, token: &str) -> Option<u32> {
    let mut words = text.split_whitespace();
    while let Some(word) = words.next() {
        if word == token {
            return words.next()?.parse().ok();
        }
    }
    None
}

pub fn parse_gtp_move(raw: &str) -> Result<(i32, i32), String> {
    let normalized = raw.trim().to_lowercase();
    if normalized == "pass" {
        return Ok((PASS_MOVE[0], PASS_MOVE[1]));
    }
    if normalized == "resign" {
        return Ok((RESIGN_MOVE[0], RESIGN_MOVE[1]));
    }
    gtp_to_board(raw)
}

pub fn gtp_to_board(mv: &str) -> Result<(i32, i32), String> {
    let normalized = mv.trim().to_uppercase();
    if normalized.len() < 2 {
        return Err(format!("KataGo 返回了无法解析的坐标: {mv}"));
    }
    let col = GTP_COLUMNS
        .find(normalized.chars().next().unwrap())
        .ok_or_else(|| format!("KataGo 返回了无法解析的列坐标: {mv}"))? as i32;
    let gtp_row: i32 = normalized[1..]
        .parse()
        .map_err(|_| format!("KataGo 返回了无法解析的行坐标: {mv}"))?;
    let row = BOARD_SIZE as i32 - gtp_row;
    if !(0..BOARD_SIZE as i32).contains(&row) {
        return Err(format!("KataGo 返回了越界坐标: {mv}"));
    }
    Ok((row, col))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gtp_coordinate_roundtrip() {
        assert_eq!(board_to_gtp(0, 0).unwrap(), "A19");
        assert_eq!(board_to_gtp(18, 18).unwrap(), "T1");
        // GTP 列跳过 I
        assert_eq!(board_to_gtp(0, 8).unwrap(), "J19");
        assert_eq!(gtp_to_board("A19").unwrap(), (0, 0));
        assert_eq!(gtp_to_board("T1").unwrap(), (18, 18));
        assert_eq!(gtp_to_board("j19").unwrap(), (0, 8));
    }

    #[test]
    fn parse_pass_and_resign() {
        assert_eq!(parse_gtp_move("pass").unwrap(), (-1, -1));
        assert_eq!(parse_gtp_move("RESIGN").unwrap(), (-2, -2));
    }

    #[test]
    fn analyze_response_with_root_info() {
        let raw = "info move Q16 visits 37 winrate 0.51 order 0 pv Q16 D4\n\
                   info move D4 visits 20 winrate 0.49 order 1 pv D4 rootInfo visits 96 winrate 0.505 scoreMean 0.3\n\
                   play Q16";
        let ((row, col), visits) = parse_analyze_response(raw).unwrap();
        assert_eq!((row, col), (3, 15)); // Q16
        assert_eq!(visits, 96);
    }

    #[test]
    fn analyze_response_sums_candidates_without_root_info() {
        let raw = "info move C3 visits 30 winrate 0.5 info move E5 visits 12 winrate 0.4\nplay pass";
        let (mv, visits) = parse_analyze_response(raw).unwrap();
        assert_eq!(mv, (-1, -1));
        assert_eq!(visits, 42);
    }

    #[test]
    fn analyze_response_takes_last_frame_and_ignores_noise() {
        let raw = "info move C3 visits 5 winrate 0.5 rootInfo visits 8 winrate 0.5\n\
                   NN eval cache hits: 123\n\
                   info move C3 visits 60 winrate 0.5 rootInfo visits 64 winrate 0.5\n\
                   play C3";
        let (_, visits) = parse_analyze_response(raw).unwrap();
        assert_eq!(visits, 64);
    }

    #[test]
    fn analyze_response_requires_play_line() {
        assert!(parse_analyze_response("info move C3 visits 5").is_err());
    }
}
