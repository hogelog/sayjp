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
    --serve          常駐モード。起動時にするのはモデルロードだけ (--model-dir /
                     SAYJP_MODEL_DIR のみ参照)。あとは stdin から 1 行 = 1 リクエストの
                     JSON を受けるたびに、元の CLI 1 回分と同じ処理 (wav を out へ書き、
                     必要なら再生) をウォームで実行し、結果 JSON を stdout に 1 行返す
                     (EOF で終了)。合成パラメータはすべてリクエストで渡す。
                     リクエスト: {"text":必須, "out"?, "play"?, "style_id"?, "speed"?,
                       "sdp"?, "style_weight"?}  (text 以外は省略時 one-shot と同じ既定値)
                     応答: {"status":"ok","path":"<書いた wav>"} / {"status":"error","error":...}
    -h, --help       このヘルプ
"#;

// 合成パラメータの既定値。one-shot の引数省略時と serve のリクエスト省略時で共有する。
const DEFAULT_STYLE_ID: i32 = 0;
const DEFAULT_SPEED: f32 = 1.0;
const DEFAULT_SDP: f32 = 0.3;
const DEFAULT_STYLE_WEIGHT: f32 = 1.0;
const DEFAULT_OUT: &str = "out.wav";

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
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut out_explicit = false;
    let mut style_id = DEFAULT_STYLE_ID;
    let mut speed = DEFAULT_SPEED;
    let mut sdp = DEFAULT_SDP;
    let mut style_weight = DEFAULT_STYLE_WEIGHT;
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
        return serve(&mut holder);
    }

    let audio = synth_one(
        &mut holder,
        &args.text,
        args.style_id,
        args.speed,
        args.sdp,
        args.style_weight,
    )?;

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
fn synth_one(
    holder: &mut TTSModelHolder,
    text: &str,
    style_id: i32,
    speed: f32,
    sdp: f32,
    style_weight: f32,
) -> Result<Vec<u8>> {
    let opts = SynthesizeOptions {
        length_scale: 1.0 / speed,
        sdp_ratio: sdp,
        style_weight,
        ..Default::default()
    };
    let audio = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        holder.easy_synthesize(IDENT, text, style_id, 0, opts)
    }))
    .map_err(|_| anyhow!("テキストを解析できませんでした (記号や特殊な文字だけの入力などが原因の可能性があります)"))?
    .map_err(|e| anyhow!("音声合成に失敗しました (テキストやスタイル ID を確認してください): {e}"))?;
    Ok(prepend_silence(&audio, 0.5).unwrap_or(audio))
}

/// serve の 1 リクエスト。起動時は --model-dir でモデルを読むだけで、合成パラメータは
/// すべてここで渡す (起動フラグからは取らない)。text 以外は省略時に CLI と同じ既定値。
#[derive(serde::Deserialize)]
struct ServeRequest {
    /// 読み上げるテキスト (JSON 文字列なので改行を含んでも 1 行で渡せる)。
    text: String,
    /// 出力 wav のパス (-o 相当)。省略時は out.wav。
    #[serde(default)]
    out: Option<String>,
    /// 生成後に再生するか。省略時は再生する。
    #[serde(default)]
    play: Option<bool>,
    #[serde(default)]
    style_id: Option<i32>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    sdp: Option<f32>,
    #[serde(default)]
    style_weight: Option<f32>,
}

/// 常駐モード。起動時にしたのはモデルロードだけ。あとは stdin から 1 行 = 1 リクエストの
/// JSON を受け、合成に必要なパラメータはすべてリクエストから取る (起動フラグは見ない)。
/// 入力: {"text":..., "out"?:..., "play"?:..., "style_id"?:..., "speed"?:..., "sdp"?:..., "style_weight"?:...}
/// 出力: {"status":"ok","path":"<書いた wav>"} または {"status":"error","error":"<理由>"}
/// 1 件の失敗ではループを抜けない (EOF で終了)。
fn serve(holder: &mut TTSModelHolder) -> Result<()> {
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
            Ok(req) => serve_one(holder, &req),
            Err(e) => serde_json::json!({ "status": "error", "error": format!("不正なリクエスト JSON: {e}") }),
        };
        writeln!(out, "{resp}")?;
        out.flush()?;
    }
    Ok(())
}

/// 1 リクエストを処理: 合成 → out のパスへ書き出し → 必要なら再生。結果 JSON を返す。
/// 省略パラメータは CLI と同じ既定値を使う。
fn serve_one(holder: &mut TTSModelHolder, req: &ServeRequest) -> serde_json::Value {
    let audio = match synth_one(
        holder,
        &req.text,
        req.style_id.unwrap_or(DEFAULT_STYLE_ID),
        req.speed.unwrap_or(DEFAULT_SPEED),
        req.sdp.unwrap_or(DEFAULT_SDP),
        req.style_weight.unwrap_or(DEFAULT_STYLE_WEIGHT),
    ) {
        Ok(a) => a,
        Err(e) => return serde_json::json!({ "status": "error", "error": format!("{e:#}") }),
    };
    let path = req.out.clone().unwrap_or_else(|| DEFAULT_OUT.to_string());
    if let Err(e) = fs::write(&path, &audio) {
        return serde_json::json!({ "status": "error", "error": format!("書き込み失敗 {path}: {e}") });
    }
    if req.play.unwrap_or(true) {
        play(Path::new(&path));
    }
    serde_json::json!({ "status": "ok", "path": path })
}
