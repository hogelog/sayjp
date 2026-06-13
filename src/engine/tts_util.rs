use std::io::Cursor;

use crate::engine::error::Result;
use crate::engine::{bert, jtalk, nlp, norm, tokenizer, utils};
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{concatenate, s, Array, Array1, Array2, Array3, Axis};
use ort::session::Session;
use tokenizers::Tokenizer;

/// BERT 特徴を word2ph に従って各文字→音素へ複製整列する (Style-Bert-VITS2 の前処理)。
#[allow(clippy::type_complexity)]
pub fn parse_text(
    text: &str,
    jtalk: &jtalk::JTalk,
    tokenizer: &Tokenizer,
    bert: &mut Session,
) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>)> {
    let text = jtalk.num2word(text)?;
    let normalized_text = norm::normalize_text(&text);

    let process = jtalk.process_text(&normalized_text)?;
    let (phones, tones, mut word2ph) = process.g2p()?;
    let (phones, tones, lang_ids) = nlp::cleaned_text_to_sequence(phones, tones);

    let phones = utils::intersperse(&phones, 0);
    let tones = utils::intersperse(&tones, 0);
    let lang_ids = utils::intersperse(&lang_ids, 0);
    for item in &mut word2ph {
        *item *= 2;
    }
    word2ph[0] += 1;

    let text = {
        let (seq_text, _) = process.text_to_seq_kata()?;
        seq_text.join("")
    };
    let (token_ids, attention_masks) = tokenizer::tokenize(&text, tokenizer)?;

    let bert_content = bert::predict(bert, token_ids, attention_masks)?;

    assert!(
        word2ph.len() == text.chars().count() + 2,
        "{} {}",
        word2ph.len(),
        normalized_text.chars().count()
    );

    let mut phone_level_feature = vec![];
    for (i, reps) in word2ph.iter().enumerate() {
        let repeat_feature = {
            let arr_len = bert_content.slice(s![i, ..]).len();
            let mut results: Array2<f32> = Array::zeros((*reps as usize, arr_len));
            for j in 0..*reps {
                let mut view = results.slice_mut(s![j, ..]);
                view.assign(&bert_content.slice(s![i, ..]));
            }
            results
        };
        phone_level_feature.push(repeat_feature);
    }
    let phone_level_feature = concatenate(
        Axis(0),
        &phone_level_feature
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>(),
    )?;
    let bert_ori = phone_level_feature.t();
    Ok((
        bert_ori.to_owned(),
        phones.into(),
        tones.into(),
        lang_ids.into(),
    ))
}

pub fn array_to_vec(audio_array: Array3<f32>) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec)?;
    for i in 0..audio_array.shape()[0] {
        for sample in audio_array.slice(s![i, 0, ..]).to_vec() {
            writer.write_sample(sample)?;
        }
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}
