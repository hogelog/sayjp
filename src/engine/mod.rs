//! Style-Bert-VITS2 (JP-Extra) のオフライン推論エンジン。
//! コードは neodyland/sbv2-api の sbv2_core (MIT, © 2024 tuna2134 / © 2025- neodyland) を
//! 移植・改変したもの。sbv2_core 自体が litagin02/Style-Bert-VITS2 (AGPL-3.0, g2p は LGPL-3.0)
//! の Rust 移植で、各処理が対応する SBV2 原典は個別ファイル先頭に記す。

pub mod bert;
pub mod error;
pub mod jtalk;
pub mod model;
pub mod mora;
pub mod nlp;
pub mod norm;
pub mod style;
pub mod tokenizer;
pub mod tts;
pub mod tts_util;
pub mod utils;
