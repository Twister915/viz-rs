use crate::binner::{BinConfig, BinLayout, Binner};
use crate::channeled::Channeled;
use crate::exponential_smoothing::ExponentialSmoothing;
use crate::fft::FramedFft;
use crate::framed::{Framed, FramedMapper, Sampled, Samples, SplitChanneledFramedMapper};
use crate::savitzky_golay::SavitzkyGolayConfig;
use crate::sliding::SlidingFrame;
use crate::timer::FramedTimed;
use crate::util::VizFloat;
use crate::window::{BlackmanNuttall, WindowingFunction};
use anyhow::{Result, anyhow};
use num_rational::Rational64;
use serde::{Deserialize, Deserializer, de::Error as DeError};
use std::fs::File;
use std::include_str;
use std::io::ErrorKind;
use std::time::Duration;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VizPipelineConfig {
    pub fps: u64,
    pub data_window_ms: u64,
    #[serde(default = "default_background_color")]
    pub background_color: VizColor,
    #[serde(default = "default_bar_color")]
    pub bar_color: VizColor,
    #[serde(default = "default_bar_difference_color")]
    pub bar_difference_color: VizColor,
    #[serde(default = "default_debug_fft_overlay_color")]
    pub debug_fft_overlay_color: VizColor,
    #[serde(default)]
    pub high_water_line: VizHighWaterLineConfig,
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
    #[serde(default)]
    pub compensation: VizBinCompensationConfig,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VizBinCompensationConfig {
    #[serde(default)]
    pub bandwidth_weight: VizFloat,
    #[serde(default)]
    pub spectral_tilt_db_per_octave: VizFloat,
    #[serde(default = "default_tilt_reference_hz")]
    pub tilt_reference_hz: VizFloat,
    #[serde(default = "default_max_compensation_adjustment_db")]
    pub max_adjustment_db: VizFloat,
}

impl Default for VizBinCompensationConfig {
    fn default() -> Self {
        Self {
            bandwidth_weight: 0.0,
            spectral_tilt_db_per_octave: 0.0,
            tilt_reference_hz: default_tilt_reference_hz(),
            max_adjustment_db: default_max_compensation_adjustment_db(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VizHighWaterLineConfig {
    #[serde(default = "default_high_water_line_color")]
    pub color: VizColor,
    #[serde(default = "default_high_water_line_fall_acceleration")]
    pub fall_acceleration: VizFloat,
}

impl Default for VizHighWaterLineConfig {
    fn default() -> Self {
        Self {
            color: default_high_water_line_color(),
            fall_acceleration: default_high_water_line_fall_acceleration(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VizColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl VizColor {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

impl<'de> Deserialize<'de> for VizColor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        parse_hex_color(&value).map_err(D::Error::custom)
    }
}

fn default_bar_color() -> VizColor {
    VizColor::rgb(79, 159, 209)
}

fn default_background_color() -> VizColor {
    VizColor::rgb(214, 236, 252)
}

fn default_bar_difference_color() -> VizColor {
    VizColor::rgb(185, 231, 255)
}

fn default_debug_fft_overlay_color() -> VizColor {
    VizColor::rgb(95, 135, 165)
}

fn default_high_water_line_color() -> VizColor {
    VizColor::rgb(248, 252, 255)
}

fn default_high_water_line_fall_acceleration() -> VizFloat {
    2.0
}

fn default_tilt_reference_hz() -> VizFloat {
    1000.0
}

fn default_max_compensation_adjustment_db() -> VizFloat {
    18.0
}

fn parse_hex_color(value: &str) -> Result<VizColor, String> {
    let hex = value.strip_prefix('#').unwrap_or(value);
    let bytes = hex.as_bytes();
    if bytes.len() != 6 {
        return Err(format!(
            "color must be a 6-digit hex color like #4f9fd1, got {value:?}"
        ));
    }

    let r = parse_hex_byte(bytes[0], bytes[1], value)?;
    let g = parse_hex_byte(bytes[2], bytes[3], value)?;
    let b = parse_hex_byte(bytes[4], bytes[5], value)?;
    Ok(VizColor::rgb(r, g, b))
}

fn parse_hex_byte(hi: u8, lo: u8, original: &str) -> Result<u8, String> {
    let hi = parse_hex_digit(hi, original)?;
    let lo = parse_hex_digit(lo, original)?;
    Ok((hi << 4) | lo)
}

fn parse_hex_digit(byte: u8, original: &str) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!(
            "color must contain only hex digits, got {original:?}"
        )),
    }
}

impl VizPipelineConfig {
    pub fn data_window(&self) -> Duration {
        Duration::from_millis(self.data_window_ms)
    }
}

const SEEK_BACK_LIMIT: usize = 1;

pub fn create_viz_pipeline<E, S>(
    source: S,
    config: VizPipelineConfig,
) -> Result<impl Framed<Item = Channeled<VizFloat>>>
where
    S: Samples<Item = Channeled<E>>,
    E: Into<VizFloat>,
{
    Ok(
        create_binned_fft_pipeline(source, config, config.binning.compensation)?
            // dBFS-ish conversion and clamp between min/max dB -> (0, 1)
            .map_mut(move |v| normalize_amplitude_db_channels(v, config.min_db, config.max_db))
            // nearby bars smoothing Savitzky Golay, now in display space
            .lift(move |size| config.smoothing0.into_mapper(size))
            // keep smooth data inside (0, 1)
            .map_mut(constrain_normalized_channels)
            // time smoothing
            .lift(move |_| ExponentialSmoothing::new(SEEK_BACK_LIMIT, config.alpha0))
            // one more gentle spatial smoothing pass after the first temporal pass
            .lift(move |size| config.smoothing1.into_mapper(size))
            // keep smooth data inside (0, 1)
            .map_mut(constrain_normalized_channels)
            // time smoothing again
            .lift(move |_| ExponentialSmoothing::new(SEEK_BACK_LIMIT, config.alpha1))
            // 48 distinct "levels" each bar can take on
            .map_mut(discrete_channel_levels(config.binning.discrete_levels))
            // time the frames and log it
            .compose(move |frames| FramedTimed::new(frames, 1024)),
    )
}

pub fn create_debug_fft_pipeline<E, S>(
    source: S,
    config: VizPipelineConfig,
) -> Result<impl Framed<Item = VizFloat>>
where
    S: Samples<Item = Channeled<E>>,
    E: Into<VizFloat>,
{
    Ok(
        create_binned_fft_pipeline(source, config, VizBinCompensationConfig::default())?
            .map(flatten_channels)
            // Smooth raw binned FFT amplitudes without applying the display dB scale.
            .lift(move |size| config.smoothing0.into_mapper(size).split_channeled(size))
            .lift(move |size| {
                ExponentialSmoothing::new(SEEK_BACK_LIMIT, config.alpha0).split_channeled(size)
            }),
    )
}

fn create_binned_fft_pipeline<E, S>(
    source: S,
    config: VizPipelineConfig,
    compensation: VizBinCompensationConfig,
) -> Result<impl Framed<Item = Channeled<VizFloat>>>
where
    S: Samples<Item = Channeled<E>>,
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
        .lift(BlackmanNuttall::mapper)
        // FFT
        .try_lift(FramedFft::new)?
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
            let layout = BinLayout::new(config);
            let compensation = BinDisplayCompensation::new(&layout, compensation);
            source
                .apply_mapper(Binner::from_layout(layout))
                .apply_mapper(compensation)
        }))
}

struct BinDisplayCompensation {
    amplitude_gains: Vec<VizFloat>,
}

impl BinDisplayCompensation {
    fn new(layout: &BinLayout, config: VizBinCompensationConfig) -> Self {
        let amplitude_gains = (0..layout.len())
            .map(move |idx| {
                let gain_db =
                    compensation_gain_db(layout.bin_size(idx), layout.center_hz(idx), config);
                VizFloat::powf(10.0, gain_db / 20.0)
            })
            .collect();

        Self { amplitude_gains }
    }
}

impl FramedMapper for BinDisplayCompensation {
    type Input = Channeled<VizFloat>;
    type Output = Channeled<VizFloat>;

    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<VizFloat>],
    ) -> Result<Option<&'a mut [Channeled<VizFloat>]>> {
        if input.len() != self.amplitude_gains.len() {
            return Ok(None);
        }

        for (elem, gain) in input.iter_mut().zip(self.amplitude_gains.iter().copied()) {
            if gain != 1.0 {
                *elem = (*elem).map(move |v| v * gain);
            }
        }

        Ok(Some(input))
    }
}

fn compensation_gain_db(
    bin_size: usize,
    center_hz: VizFloat,
    config: VizBinCompensationConfig,
) -> VizFloat {
    let bandwidth_db = if bin_size <= 1 {
        0.0
    } else {
        config.bandwidth_weight * 10.0 * (bin_size as VizFloat).log10()
    };

    let tilt_db = if center_hz > 0.0 && config.tilt_reference_hz > 0.0 {
        config.spectral_tilt_db_per_octave * (center_hz / config.tilt_reference_hz).log2()
    } else {
        0.0
    };

    (bandwidth_db + tilt_db).clamp(-config.max_adjustment_db, config.max_adjustment_db)
}

fn normalize_amplitude_db(v: &mut VizFloat, min: VizFloat, max: VizFloat) {
    let floor = VizFloat::powf(10.0, min / 20.0);
    let amp = if v.is_finite() && *v > floor {
        *v
    } else {
        floor
    };
    let mut db = 20.0 * amp.log10();
    normalize_between(&mut db, min, max);
    *v = db;
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

fn constrain_normalized(v: &mut VizFloat) {
    if v.is_nan() {
        *v = 0.0;
    } else {
        *v = v.clamp(0.0, 1.0);
    }
}

fn normalize_amplitude_db_channels(input: &mut Channeled<VizFloat>, min: VizFloat, max: VizFloat) {
    for_each_channel(input, move |v| normalize_amplitude_db(v, min, max));
}

fn constrain_normalized_channels(input: &mut Channeled<VizFloat>) {
    for_each_channel(input, constrain_normalized);
}

fn for_each_channel<F>(input: &mut Channeled<VizFloat>, f: F)
where
    F: FnMut(&mut VizFloat),
{
    input.as_mut_ref().for_each(f);
}

fn flatten_channels(input: &Channeled<VizFloat>) -> VizFloat {
    use Channeled::*;
    match *input {
        Stereo(a, b) => (((a * a) + (b * b)) / 2.0).sqrt(),
        Mono(v) => v,
    }
}

fn discrete_channel_levels(levels: u32) -> impl FnMut(&mut Channeled<VizFloat>) {
    let levels = levels as VizFloat;
    move |input| for_each_channel(input, move |v| *v = (*v * levels).floor() / levels)
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
                };
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
    validate_high_water_line_config(&cfg.high_water_line)?;

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

    validate_compensation_config(&binning.compensation)?;

    Ok(cfg)
}

fn validate_compensation_config(cfg: &VizBinCompensationConfig) -> Result<()> {
    if !cfg.bandwidth_weight.is_finite() || cfg.bandwidth_weight < 0.0 || cfg.bandwidth_weight > 1.0
    {
        return Err(anyhow!(
            "bandwidth_weight must be finite and within [0.0, 1.0], got {}",
            cfg.bandwidth_weight
        ));
    }

    if !cfg.spectral_tilt_db_per_octave.is_finite() {
        return Err(anyhow!(
            "spectral_tilt_db_per_octave must be finite, got {}",
            cfg.spectral_tilt_db_per_octave
        ));
    }

    if !cfg.tilt_reference_hz.is_finite() || cfg.tilt_reference_hz <= 0.0 {
        return Err(anyhow!(
            "tilt_reference_hz must be finite and > 0, got {}",
            cfg.tilt_reference_hz
        ));
    }

    if !cfg.max_adjustment_db.is_finite() || cfg.max_adjustment_db < 0.0 {
        return Err(anyhow!(
            "max_adjustment_db must be finite and >= 0, got {}",
            cfg.max_adjustment_db
        ));
    }

    Ok(())
}

fn validate_high_water_line_config(cfg: &VizHighWaterLineConfig) -> Result<()> {
    if !cfg.fall_acceleration.is_finite() || cfg.fall_acceleration < 0.0 {
        return Err(anyhow!(
            "high_water_line.fall_acceleration must be finite and >= 0, got {}",
            cfg.fall_acceleration
        ));
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensation_blends_bandwidth_energy_and_spectral_tilt() {
        let config = VizBinCompensationConfig {
            bandwidth_weight: 0.5,
            spectral_tilt_db_per_octave: 3.0,
            tilt_reference_hz: 1000.0,
            max_adjustment_db: 18.0,
        };

        let gain = compensation_gain_db(32, 8000.0, config);

        assert!((gain - 16.52574989159953).abs() < 1e-12);
    }

    #[test]
    fn compensation_clamps_large_adjustments() {
        let config = VizBinCompensationConfig {
            bandwidth_weight: 1.0,
            spectral_tilt_db_per_octave: 12.0,
            tilt_reference_hz: 1000.0,
            max_adjustment_db: 6.0,
        };

        assert_eq!(compensation_gain_db(64, 16000.0, config), 6.0);
        assert_eq!(compensation_gain_db(1, 125.0, config), -6.0);
    }

    #[test]
    fn tracked_configs_parse() {
        let default_config: VizPipelineConfig =
            serde_yaml::from_str(include_str!("default-config.yml")).unwrap();
        validate_config(default_config).unwrap();

        let workspace_config: VizPipelineConfig =
            serde_yaml::from_str(include_str!("../config.yml")).unwrap();
        validate_config(workspace_config).unwrap();
    }

    #[test]
    fn parses_bar_color_hex_codes() {
        assert_eq!(
            parse_hex_color("#1a2B3c").unwrap(),
            VizColor::rgb(0x1a, 0x2b, 0x3c)
        );
        assert_eq!(
            parse_hex_color("ff0044").unwrap(),
            VizColor::rgb(0xff, 0x00, 0x44)
        );
    }

    #[test]
    fn rejects_invalid_bar_color_hex_codes() {
        assert!(parse_hex_color("#123").is_err());
        assert!(parse_hex_color("#12345g").is_err());
        assert!(parse_hex_color("#a\u{e9}\u{e9}b").is_err());
    }
}
