use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("JPreprocess error: {0}")]
    JPreprocess(#[from] jpreprocess::error::JPreprocessError),
    #[error("ONNX error: {0}")]
    Ort(#[from] ort::Error),
    #[error("NDArray error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("Value error: {0}")]
    Value(String),
    #[error("Serde_json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("hound error: {0}")]
    Hound(#[from] hound::Error),
    #[error("model not found error: {0}")]
    ModelNotFound(String),
    #[error("Style error: {0}")]
    Style(String),
}

pub type Result<T> = std::result::Result<T, Error>;
