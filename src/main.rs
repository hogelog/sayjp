use std::env;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

mod engine;
use engine::tts::{SynthesizeOptions, TTSModelHolder};

/// 再生時の頭切れ対策。出力は 16bit PCM/mono なので Int 経路が主。
fn prepend_silence(wav: &[u8], secs: f32) -> Result<Vec<u8>> {
    let spec = hound::WavReader::new(Cursor::new(wav))
        .context("wav 解析失敗")?
        .spec();
    let lead = (spec.sample_rate as f32 * secs) as usize * spec.channels as usize;

    let mut out = Vec::new();
    {
        let mut w =
            hound::WavWriter::new(Cursor::new(&mut out), spec).context("wav 書き込み準備失敗")?;
        match spec.sample_format {
            hound::SampleFormat::Float => {
                for _ in 0..lead {
                    w.write_sample(0.0f32)?;
                }
                for s in hound::WavReader::new(Cursor::new(wav))?.into_samples::<f32>() {
                    w.write_sample(s?)?;
                }
            }
            hound::SampleFormat::Int => {
                for _ in 0..lead {
                    w.write_sample(0i32)?;
                }
                for s in hound::WavReader::new(Cursor::new(wav))?.into_samples::<i32>() {
                    w.write_sample(s?)?;
                }
            }
        }
        w.finalize()?;
    }
    Ok(out)
}

const HELP: &str = r#"sayjp — オフライン日本語 TTS

USAGE:
    sayjp [OPTIONS] "読み上げるテキスト"

OPTIONS:
    -o FILE          出力 wav (既定: out.wav)
    -s ID            スタイル ID (既定: 0)
    --speed N        話速 (1.0=標準, 大きいほど速い。既定: 1.0)
    --sdp N          抑揚(リズム)の強さ 0.0〜1.0 (大きいほど豊か。既定: 0.3)
    --style-weight N スタイルの強さ (小さいほど無感情・落ち着く。既定: 1.0)
    --model-dir DIR  モデルディレクトリ (既定: 実行ファイル隣の models/)
    --no-play        再生せず wav 出力のみ (既定は生成後に再生)
    --serve          常駐モード。モデルを 1 回ロードし、stdin から 1 行 = 1 リクエストの
                     JSON {"text":"<テキスト>"} を受けるたびに合成し、結果 JSON
                     {"ok":true,"wav_base64":"<base64 の wav>"} か
                     {"ok":false,"error":...} を stdout に 1 行返す (EOF で終了)。
                     ファイルを介さずパイプで完結。プロセスを使い回すことで 2 回目
                     以降をウォーム実行する用途。
    -h, --help       このヘルプ
"#;

struct Args {
    text: String,
    out: PathBuf,
    out_explicit: bool,
    style_id: i32,
    speed: f32,
    sdp: f32,
    style_weight: f32,
    play: bool,
    model_dir: Option<PathBuf>,
    serve: bool,
}

fn parse_args() -> Result<Option<Args>> {
    let mut out = PathBuf::from("out.wav");
    let mut out_explicit = false;
    let mut style_id = 0i32;
    let mut speed = 1.0f32;
    let mut sdp = 0.3f32;
    let mut style_weight = 1.0f32;
    let mut play = true;
    let mut model_dir: Option<PathBuf> = None;
    let mut serve = false;
    let mut text: Option<String> = None;

    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "-o" => {
                out = PathBuf::from(it.next().ok_or_else(|| anyhow!("-o は出力ファイル名が必要です"))?);
                out_explicit = true;
            }
            "-s" => {
                style_id = it
                    .next()
                    .ok_or_else(|| anyhow!("-s はスタイル ID が必要です"))?
                    .parse()
                    .context("-s のスタイル ID が不正です")?
            }
            "--speed" => {
                speed = it
                    .next()
                    .ok_or_else(|| anyhow!("--speed は数値が必要です"))?
                    .parse()
                    .context("--speed の値が不正です")?
            }
            "--sdp" => {
                sdp = it
                    .next()
                    .ok_or_else(|| anyhow!("--sdp は数値が必要です"))?
                    .parse()
                    .context("--sdp の値が不正です")?
            }
            "--style-weight" => {
                style_weight = it
                    .next()
                    .ok_or_else(|| anyhow!("--style-weight は数値が必要です"))?
                    .parse()
                    .context("--style-weight の値が不正です")?
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow!("--model-dir はディレクトリが必要です"))?,
                ))
            }
            "--no-play" => play = false,
            "--serve" => serve = true,
            s if s.starts_with('-') && s.len() > 1 => return Err(anyhow!("不明なオプション: {s}")),
            s => text = Some(s.to_string()),
        }
    }

    // serve モードは stdin から1リクエストずつ読むので、起動引数のテキストは不要。
    let text = if serve {
        text.unwrap_or_default()
    } else {
        let t = text.ok_or_else(|| anyhow!("読み上げるテキストを指定してください (-h でヘルプ)"))?;
        if t.trim().is_empty() {
            return Err(anyhow!("テキストが空です"));
        }
        t
    };
    if speed <= 0.0 {
        return Err(anyhow!("--speed は正の数を指定してください"));
    }
    if !(0.0..=1.0).contains(&sdp) {
        return Err(anyhow!("--sdp は 0.0〜1.0 で指定してください"));
    }
    Ok(Some(Args {
        text,
        out,
        out_explicit,
        style_id,
        speed,
        sdp,
        style_weight,
        play,
        model_dir,
        serve,
    }))
}

fn model_dir(arg: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(d) = arg {
        return Ok(d);
    }
    if let Ok(d) = env::var("SAYJP_MODEL_DIR") {
        return Ok(PathBuf::from(d));
    }
    let exe = env::current_exe().context("実行ファイルパスの取得に失敗")?;
    let dir = exe.parent().unwrap_or_else(|| Path::new(".")).join("models");
    Ok(dir)
}

fn play(path: &Path) {
    let p = path.to_string_lossy().to_string();
    let r = if cfg!(target_os = "macos") {
        Command::new("afplay").arg(&p).status()
    } else if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", "start", "", &p]).status()
    } else {
        Command::new("aplay").arg(&p).status()
    };
    if r.is_err() {
        eprintln!("警告: 再生コマンドを起動できませんでした ({})", p);
    }
}

fn main() {
    // 解析不能なテキストでは移植元 g2p の assert! が発火しうる。catch_unwind で日本語メッセージに
    // 変換するため、ここで既定のバックトレース出力を黙らせる。
    std::panic::set_hook(Box::new(|_| {}));
    if let Err(e) = run() {
        eprintln!("エラー: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = match parse_args()? {
        None => {
            print!("{HELP}");
            return Ok(());
        }
        Some(a) => a,
    };

    // デバッグ: g2p (音素列/アクセント tone 列) だけを JSON で出力して終了 (モデル不要)。
    if env::var("SAYJP_DUMP_G2P").is_ok() {
        let jtalk = engine::jtalk::JTalk::new().map_err(|e| anyhow!("jtalk 初期化失敗: {e}"))?;
        let text = jtalk
            .num2word(&args.text)
            .map_err(|e| anyhow!("num2word 失敗: {e}"))?;
        let normalized = engine::norm::normalize_text(&text);
        let process = jtalk
            .process_text(&normalized)
            .map_err(|e| anyhow!("process_text 失敗: {e}"))?;
        let (phones, tones, word2ph) = process.g2p().map_err(|e| anyhow!("g2p 失敗: {e}"))?;
        println!(
            "{}",
            serde_json::json!({
                "text": args.text,
                "normalized": normalized,
                "phones": phones,
                "tones": tones,
                "word2ph": word2ph,
            })
        );
        return Ok(());
    }

    let dir = model_dir(args.model_dir.clone())?;
    let mut holder = load_models(&dir)?;

    if args.serve {
        return serve(&mut holder, &args);
    }

    let audio = synth_one(&mut holder, &args.text, &args)?;

    let keep = args.out_explicit || !args.play;
    let target = if keep {
        args.out.clone()
    } else {
        std::env::temp_dir().join(format!("sayjp-play-{}.wav", std::process::id()))
    };

    fs::write(&target, audio).with_context(|| format!("書き込み失敗: {}", target.display()))?;
    if keep {
        eprintln!("生成: {}", target.display());
    }

    if args.play {
        play(&target);
        if !keep {
            let _ = fs::remove_file(&target);
        }
    }
    Ok(())
}

const IDENT: &str = "voice";

/// モデルディレクトリから BERT + 音声モデルを読み込む (重い処理。serve では 1 回だけ)。
fn load_models(dir: &Path) -> Result<TTSModelHolder> {
    if !dir.is_dir() {
        return Err(anyhow!(
            "モデルディレクトリが見つかりません: {}\n--model-dir か SAYJP_MODEL_DIR で指定するか、models/ を実行ファイルと同階層に配置してください。",
            dir.display()
        ));
    }
    let bert = dir.join("deberta.onnx");
    let tokenizer = dir.join("tokenizer.json");
    let voice = dir.join("voice.onnx");
    let style = dir.join("style_vectors.json");

    let mut holder = TTSModelHolder::new(
        &fs::read(&bert).with_context(|| format!("読み込み失敗: {}", bert.display()))?,
        &fs::read(&tokenizer).with_context(|| format!("読み込み失敗: {}", tokenizer.display()))?,
    )
    .map_err(|e| anyhow!("モデルの初期化に失敗しました: {e}"))?;
    holder
        .load(
            IDENT,
            fs::read(&voice).with_context(|| format!("読み込み失敗: {}", voice.display()))?,
            fs::read(&style).with_context(|| format!("読み込み失敗: {}", style.display()))?,
        )
        .map_err(|e| anyhow!("音声モデルの読み込みに失敗 ({}): {e}", voice.display()))?;
    Ok(holder)
}

/// 1 文を合成して wav バイト列を返す (頭の無音付き)。run/serve 共通。
fn synth_one(holder: &mut TTSModelHolder, text: &str, args: &Args) -> Result<Vec<u8>> {
    let opts = SynthesizeOptions {
        length_scale: 1.0 / args.speed,
        sdp_ratio: args.sdp,
        style_weight: args.style_weight,
        ..Default::default()
    };
    let audio = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        holder.easy_synthesize(IDENT, text, args.style_id, 0, opts)
    }))
    .map_err(|_| anyhow!("テキストを解析できませんでした (記号や特殊な文字だけの入力などが原因の可能性があります)"))?
    .map_err(|e| anyhow!("音声合成に失敗しました (テキストやスタイル ID を確認してください): {e}"))?;
    Ok(prepend_silence(&audio, 0.5).unwrap_or(audio))
}

#[derive(serde::Deserialize)]
struct ServeRequest {
    /// 読み上げるテキスト (JSON 文字列なので改行を含んでも 1 行で渡せる)。
    text: String,
}

/// 常駐モード。モデルを保持したまま stdin から 1 行 = 1 リクエストの JSON を処理し続ける。
/// 入出力ともファイルを介さず JSON で完結する。
/// 入力: {"text":"<テキスト>"}
/// 出力: {"ok":true,"wav_base64":"<base64 の wav>"} または {"ok":false,"error":"<理由>"}
/// テキストの改行は JSON エスケープで 1 行に収まる。1 件の失敗ではループを抜けない (EOF で終了)。
fn serve(holder: &mut TTSModelHolder, args: &Args) -> Result<()> {
    use base64::Engine;
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.context("stdin の読み取りに失敗")?;
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<ServeRequest>(&line) {
            Ok(req) => match synth_one(holder, &req.text, args) {
                Ok(audio) => {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&audio);
                    serde_json::json!({ "ok": true, "wav_base64": b64 })
                }
                Err(e) => serde_json::json!({ "ok": false, "error": format!("{e:#}") }),
            },
            Err(e) => serde_json::json!({ "ok": false, "error": format!("不正なリクエスト JSON: {e}") }),
        };
        writeln!(out, "{resp}")?;
        out.flush()?;
    }
    Ok(())
}
