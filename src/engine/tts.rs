use crate::engine::error::{Error, Result};
use crate::engine::{jtalk, model, style, tokenizer, tts_util};
use ndarray::{concatenate, Array1, Array2, Array3, Axis};
use ort::session::Session;
use tokenizers::Tokenizer;

#[derive(PartialEq, Eq, Clone)]
pub struct TTSIdent(String);

impl std::fmt::Display for TTSIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<S: AsRef<str>> From<S> for TTSIdent {
    fn from(value: S) -> Self {
        TTSIdent(value.as_ref().to_string())
    }
}

struct TTSModel {
    vits2: Session,
    style_vectors: Array2<f32>,
    ident: TTSIdent,
}

pub struct TTSModelHolder {
    tokenizer: Tokenizer,
    bert: Session,
    models: Vec<TTSModel>,
    jtalk: jtalk::JTalk,
}

impl TTSModelHolder {
    pub fn new<P: AsRef<[u8]>>(bert_model_bytes: P, tokenizer_bytes: P) -> Result<Self> {
        let bert = model::load_model(bert_model_bytes)?;
        let jtalk = jtalk::JTalk::new()?;
        let tokenizer = tokenizer::get_tokenizer(tokenizer_bytes)?;
        Ok(TTSModelHolder {
            bert,
            models: vec![],
            jtalk,
            tokenizer,
        })
    }

    pub fn load<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        vits2_bytes: P,
        style_json_bytes: P,
    ) -> Result<()> {
        let ident = ident.into();
        if self.find_model(ident.clone()).is_err() {
            self.models.push(TTSModel {
                vits2: model::load_model(&vits2_bytes)?,
                style_vectors: style::load_style(style_json_bytes)?,
                ident,
            });
        }
        Ok(())
    }

    fn find_model<I: Into<TTSIdent>>(&mut self, ident: I) -> Result<&mut TTSModel> {
        let ident = ident.into();
        self.models
            .iter_mut()
            .find(|m| m.ident == ident)
            .ok_or(Error::ModelNotFound(ident.to_string()))
    }

    pub fn get_style_vector<I: Into<TTSIdent>>(
        &mut self,
        ident: I,
        style_id: i32,
        weight: f32,
    ) -> Result<Array1<f32>> {
        style::get_style_vector(&self.find_model(ident)?.style_vectors, style_id, weight)
    }

    pub fn easy_synthesize<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        style_id: i32,
        speaker_id: i64,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        let style_vector = self.get_style_vector(ident, style_id, options.style_weight)?;
        let audio_array = if options.split_sentences {
            let texts: Vec<&str> = text.split('\n').collect();
            let mut audios = vec![];
            for (i, t) in texts.iter().enumerate() {
                if t.is_empty() {
                    continue;
                }
                let (bert_ori, phones, tones, lang_ids) =
                    tts_util::parse_text(t, &self.jtalk, &self.tokenizer, &mut self.bert)?;
                let vits2 = &mut self.find_model(ident)?.vits2;
                let audio = model::synthesize(
                    vits2,
                    bert_ori,
                    phones,
                    Array1::from_vec(vec![speaker_id]),
                    tones,
                    lang_ids,
                    style_vector.clone(),
                    options.sdp_ratio,
                    options.length_scale,
                    0.677,
                    0.8,
                )?;
                audios.push(audio);
                if i != texts.len() - 1 {
                    audios.push(Array3::zeros((1, 1, 22050)));
                }
            }
            concatenate(
                Axis(2),
                &audios.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )?
        } else {
            let (bert_ori, phones, tones, lang_ids) =
                tts_util::parse_text(text, &self.jtalk, &self.tokenizer, &mut self.bert)?;
            let vits2 = &mut self.find_model(ident)?.vits2;
            model::synthesize(
                vits2,
                bert_ori,
                phones,
                Array1::from_vec(vec![speaker_id]),
                tones,
                lang_ids,
                style_vector,
                options.sdp_ratio,
                options.length_scale,
                0.677,
                0.8,
            )?
        };
        tts_util::array_to_vec(audio_array)
    }
}

pub struct SynthesizeOptions {
    pub sdp_ratio: f32,
    pub length_scale: f32,
    pub style_weight: f32,
    pub split_sentences: bool,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        SynthesizeOptions {
            sdp_ratio: 0.0,
            length_scale: 1.0,
            style_weight: 1.0,
            split_sentences: true,
        }
    }
}
