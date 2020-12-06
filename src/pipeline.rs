use crate::binner::{BinConfig, Binner};
use crate::channeled::Channeled;
use crate::exponential_smoothing::ExponentialSmoothing;
use crate::fft::FramedFft;
use crate::framed::{Framed, Sampled, Samples};
use crate::savitzky_golay::SavitzkyGolayConfig;
use crate::sliding::SlidingFrame;
use crate::timer::FramedTimed;
use crate::window::{BlackmanNuttall, WindowingFunction};
use anyhow::{anyhow, Result};
use num_rational::Rational64;
use serde::Deserialize;
use std::fs::File;
use std::include_str;
use std::io::ErrorKind;
use std::time::Duration;
use crate::util::VizFloat;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VizPipelineConfig {
    pub fps: u64,
    pub data_window_ms: u64,
    pub alpha0: VizFloat,
    pub alpha1: VizFloat,
    pub smoothing0: SavitzkyGolayConfig,
    pub smoothing1: SavitzkyGolayConfig,
    pub min_db: VizFloat,
    pub max_db: VizFloat,
    pub binning: VizBinningConfig,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VizBinningConfig {
    pub bins: usize,
    pub fmax: VizFloat,
    pub fmin: VizFloat,
    pub gamma: VizFloat,
    pub discrete_levels: u32,
}

impl VizPipelineConfig {
    pub fn data_window(&self) -> Duration {
        Duration::from_millis(self.data_window_ms)
    }
}

const SEEK_BACK_LIMIT: usize = 1;

pub fn create_viz_pipeline<E, I, S>(source: S, config: VizPipelineConfig) -> Result<impl Framed<VizFloat, I>>
where
    S: Samples<Channeled<E>, I>,
    E: Into<VizFloat>,
{
    Ok(source
        // change RawSample to VizFloat
        .map(move |v| v.map(move |c| c.into()))
        // sliding frames of data
        .compose(move |wav| {
            let frame_size = wav.samples_from_dur(config.data_window());
            let sample_rate: Rational64 = (wav.sample_rate() as i64).into();
            let frame_rate = Rational64::new_raw(1, config.fps as i64);
            let frame_stride = frame_rate * sample_rate;
            let frame_stride = *frame_stride.round().numer() as usize;
            println!(
                "sliding window: stride={}, size={}",
                frame_stride, frame_size
            );
            SlidingFrame::new(wav, frame_size, frame_stride)
        })
        // blackman nuttall window
        .lift(move |size| BlackmanNuttall::mapper(size))
        // FFT
        .try_lift(move |size| FramedFft::new(size))?
        // time smoothing
        .lift(move |_| ExponentialSmoothing::new(SEEK_BACK_LIMIT, config.alpha0))
        // nearby bars smoothing Savitzky Golay
        .lift(move |size| config.smoothing0.into_mapper(size))
        // bin the FFT output into a smaller number of bars
        .compose(move |source| {
            let config = BinConfig {
                bins: config.binning.bins,
                fmin: config.binning.fmin,
                fmax: config.binning.fmax,
                gamma: config.binning.gamma,
                input_size: source.full_frame_size(),
                sample_rate: source.sample_rate(),
            };
            source.apply_mapper(Binner::new(config))
        })
        // dB conversion
        .map_mut(channeled_map_mut(to_db))
        // clamp between min/max dB -> (0, 1)
        .map_mut(channeled_map_mut(move |v| {
            normalize_between(v, config.min_db, config.max_db)
        }))
        // normalize infinities and NaNs
        .map_mut(channeled_map_mut(normalize_infs))
        // more savitzky golay smoothing after binning
        .lift(move |size| config.smoothing1.into_mapper(size))
        // keep smooth data inside (0, 1)
        .map_mut(channeled_map_mut(constrain_normalized))
        // time smoothing again
        .lift(move |_| ExponentialSmoothing::new(SEEK_BACK_LIMIT, config.alpha1))
        // Channeled data to single value per bar
        .map(flatten_channels)
        // 48 distinct "levels" each bar can take on
        .map_mut(discrete_levels(config.binning.discrete_levels))
        // time the frames and log it
        .compose(move |frames| FramedTimed::new(frames, 1024)))
}

fn to_db(v: &mut VizFloat) {
    *v = 20.0 * v.log10();
}

fn normalize_between(v: &mut VizFloat, min: VizFloat, max: VizFloat) {
    let vv = *v;
    if vv < min {
        *v = 0.0;
    } else if vv > max {
        *v = 1.0;
    } else {
        *v = (vv - min) / (max - min);
    }
}

fn normalize_infs(v: &mut VizFloat) {
    let vv = *v;
    if v.is_nan() || vv == VizFloat::NEG_INFINITY {
        *v = 0.0;
    } else if vv == VizFloat::INFINITY {
        *v = 1.0;
    }
}

fn constrain_normalized(v: &mut VizFloat) {
    let vv = *v;
    if vv > 1.0 {
        *v = 1.0;
    } else if vv < 0.0 {
        *v = 0.0;
    }
}

fn flatten_channels(input: &Channeled<VizFloat>) -> VizFloat {
    use Channeled::*;
    match *input {
        Stereo(a, b) => (a + b) / (2.0 as VizFloat),
        Mono(v) => v,
    }
}

fn discrete_levels(levels: u32) -> impl FnMut(&mut VizFloat) {
    let levels = levels as VizFloat;
    move |v| *v = (*v * levels).floor() / levels
}

fn channeled_map_mut<F, T>(mut f: F) -> impl FnMut(&mut Channeled<T>)
where
    F: FnMut(&mut T),
{
    move |input| {
        input.as_mut_ref().for_each(&mut f);
    }
}

pub fn open_config_or_default() -> Result<VizPipelineConfig> {
    match open_config() {
        Ok(Some(config)) => Ok(config),
        Ok(None) => Ok(default_config()),
        Err(err) => Err(err),
    }
}

macro_rules! try_load_config_from {
    ($e: literal) => {
        match open_config_file($e) {
            Ok(Some(v)) => {
                eprintln!("[config] loaded config from {}", $e);
                return Ok(Some(v));
            }
            Ok(None) => {
                eprintln!(
                    "[config] skipping load from {}, no config at this location",
                    $e
                );
            }
            Err(err) => return Err(err),
        }
    };
}

pub fn open_config() -> Result<Option<VizPipelineConfig>> {
    try_load_config_from!("config.yaml");
    try_load_config_from!("config.yml");
    try_load_config_from!("config");
    Ok(None)
}

pub fn open_config_file(file: &str) -> Result<Option<VizPipelineConfig>> {
    Ok(Some(validate_config(serde_yaml::from_reader(
        match File::open(file) {
            Ok(f) => f,
            Err(err) => {
                return match err.kind() {
                    ErrorKind::NotFound => Ok(None),
                    other => Err(anyhow!("error opening file {} :: {:?}", file, other)),
                }
            }
        },
    )?)?))
}

fn validate_config(cfg: VizPipelineConfig) -> Result<VizPipelineConfig> {
    if cfg.fps <= 1 {
        return Err(anyhow!("fps must be > 1, got {}", cfg.fps));
    }

    if cfg.data_window_ms <= 1 {
        return Err(anyhow!(
            "data window ms must be > 1ms, got {}ms",
            cfg.data_window_ms
        ));
    }

    if cfg.alpha0 <= 0.0 || cfg.alpha0 > 1.0 || !cfg.alpha0.is_normal() {
        return Err(anyhow!(
            "smoothing constant alpha0 out of range, got {} need (0.0, 1.0]",
            cfg.alpha0
        ));
    }

    if cfg.alpha1 <= 0.0 || cfg.alpha1 > 1.0 || !cfg.alpha1.is_normal() {
        return Err(anyhow!(
            "smoothing constant alpha1 out of range, got {} need (0.0, 1.0]",
            cfg.alpha1
        ));
    }

    validate_smoothing_config(&cfg.smoothing0)?;
    validate_smoothing_config(&cfg.smoothing1)?;

    if !cfg.min_db.is_normal() {
        return Err(anyhow!("invalid min_db, non-normal number {}", cfg.min_db));
    }

    if !cfg.max_db.is_normal() {
        return Err(anyhow!("invalid max_db, non-normal number {}", cfg.min_db));
    }

    if cfg.min_db >= cfg.max_db {
        return Err(anyhow!(
            "min_db must be strictly less than max_db, got min={}, max={}",
            cfg.min_db,
            cfg.max_db
        ));
    }

    let binning = &cfg.binning;
    if binning.bins <= 1 {
        return Err(anyhow!("must specify > 1 bin, got {}", binning.bins));
    }

    if !binning.fmin.is_normal() {
        return Err(anyhow!(
            "invalid fmin, must be a normal number, got {}",
            binning.fmin
        ));
    }

    if !binning.fmax.is_normal() {
        return Err(anyhow!(
            "invalid fmax, must be a normal number, got {}",
            binning.fmax
        ));
    }

    if binning.fmin >= binning.fmax {
        return Err(anyhow!(
            "fmin must be strictly less than fmax, got min={}, max={}",
            binning.fmin,
            binning.fmax
        ));
    }

    if !binning.gamma.is_normal() || binning.gamma <= 0.0 {
        return Err(anyhow!(
            "gamma must be a normal positive number, got {}",
            binning.gamma
        ));
    }

    if binning.discrete_levels <= 2 {
        return Err(anyhow!(
            "discrete_levels must be a number greater than 2, got {}",
            binning.discrete_levels
        ));
    }

    Ok(cfg)
}

fn validate_smoothing_config(cfg: &SavitzkyGolayConfig) -> Result<()> {
    if cfg.degree == 0 {
        return Err(anyhow!(
            "invalid smoothing degree, need > 0, got {}",
            cfg.degree
        ));
    }

    if cfg.window_size < 2 || cfg.window_size % 2 != 1 {
        return Err(anyhow!("need odd window_size > 2, got {}", cfg.window_size));
    }

    if cfg.order >= 1 {
        eprintln!("[warn] smoothing order > 0 is not recommended")
    }

    Ok(())
}

fn default_config() -> VizPipelineConfig {
    let out = serde_yaml::from_str(include_str!("default-config.yml")).expect("should be valid");
    eprintln!("[config] using default config...");
    out
}
