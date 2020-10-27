use crate::binner::{BinConfig, Binner};
use crate::channeled::Channeled;
use crate::db::DbMapper;
use crate::exponential_smoothing::ExponentialSmoothing;
use crate::fft::FramedFft;
use crate::fraction::Fraction;
use crate::framed::{Framed, MapperToChanneled, Sampled, Samples};
use crate::savitzky_golay::{SavitzkyGolayConfig, SavitzkyGolayMapper};
use crate::sliding::SlidingFrame;
use crate::wav::WavFile;
use crate::window::{BlackmanNuttall, WindowingFunction};
use anyhow::Result;
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::WindowCanvas;
use std::ops::Add;
use std::time::{Duration, Instant};

const FPS: u64 = 142;
const DATA_WINDOW_MS: u64 = 70;

pub fn visualize(file: &str) -> Result<()> {
    let sdl_context = sdl2::init().map_err(map_sdl_err)?;
    let video_subsystem = sdl_context.video().map_err(map_sdl_err)?;
    let window = video_subsystem
        .window("vis-rs", 1280, 720)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().accelerated().build()?;
    canvas.clear();
    canvas.present();

    let (mut frames, wav_src) = create_data_src(file)?;

    let audio_player = sdl_context
        .audio()
        .map_err(map_sdl_err)?
        .open_playback(
            None,
            &AudioSpecDesired {
                freq: Some(wav_src.sample_rate as i32),
                samples: None,
                channels: Some(wav_src.num_channels as u8),
            },
            move |_| WavPlayer { source: wav_src },
        )
        .map_err(map_sdl_err)?;

    let mut event_pump = sdl_context.event_pump().map_err(map_sdl_err)?;

    audio_player.resume();
    let mut last_frame_for_ts: Option<Instant> = None;
    let frame_delta = Duration::new(0, (1_000_000_000u64 / FPS) as u32);
    let frame_for_offset = Duration::from_millis(DATA_WINDOW_MS / 2);
    // frame_for_offset += frame_delta.mul_f64(alpha_frame_offset());
    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return Ok(()),
                _ => {}
            }
        }
        let now = Instant::now();
        if let Some(last_frame_for) = &last_frame_for_ts {
            let cur_frame_for = last_frame_for.add(frame_delta);
            let cur_audio_at = now;
            // three cases: we're behind by more than one frame, we're ahead by more than one frame, or we're in line

            let status = if cur_frame_for > cur_audio_at {
                let t_delta = cur_frame_for - cur_audio_at;
                if t_delta > frame_delta {
                    // we're ahead by more than one frame
                    t_delta.div_duration_f64(frame_delta) as i32
                } else {
                    0
                }
            } else if cur_frame_for < cur_audio_at {
                let t_delta = cur_audio_at - cur_frame_for;
                if t_delta > frame_delta {
                    // we're behind by more than one frame
                    -(t_delta.div_duration_f64(frame_delta) as i32)
                } else {
                    0
                }
            } else {
                0
            };

            if status.abs() > 1 {
                println!("status = {}", status);
            }
            if status > 0 {
                std::thread::sleep(frame_delta);
            } else {
                if let Some(frame) = frames.next_frame()? {
                    last_frame_for_ts = Some(cur_frame_for);
                    if status == 0 {
                        draw_frame(&mut canvas, frame)?;
                    }
                } else {
                    audio_player.pause();
                    return Ok(());
                }
            }
        } else {
            last_frame_for_ts = Some(now.add(frame_for_offset));
        }
    }
}

const ALPHA0: f64 = 0.85;
const ALPHA1: f64 = 0.55;

fn create_data_src(file: &str) -> Result<(impl Framed<f64>, WavFile)> {
    const SEEK_BACK_LIMIT: usize = 1;
    const BUF_SIZE: usize = 32768;

    let frame_src = WavFile::open(file, BUF_SIZE)?
        .map(move |v| v.map(move |c| (*c).into()))
        .compose(move |wav| {
            let frame_size = wav.samples_from_dur(Duration::from_millis(DATA_WINDOW_MS));
            let sample_rate = Fraction::new(wav.sample_rate() as i64, 1).unwrap();
            let frame_rate = Fraction::new(1, FPS as i64).unwrap();
            let frame_stride = frame_rate * sample_rate;
            let frame_stride = frame_stride.rounded() as usize;
            SlidingFrame::new(wav, frame_size, frame_stride)
        })
        .lift(move |size| BlackmanNuttall::mapper(size).into_channeled())
        .try_lift(move |size| FramedFft::new(size))?
        .compose(move |frames| ExponentialSmoothing::new(frames, SEEK_BACK_LIMIT, ALPHA0))
        .lift(move |size| {
            SavitzkyGolayMapper::new(
                size,
                SavitzkyGolayConfig {
                    window_size: 45,
                    polynomial_order: 4,
                },
            )
            .into_channeled()
        })
        .compose(move |source| {
            let config = BinConfig {
                bins: 48,
                fmin: 42.0,
                fmax: 14000.0,
                gamma: 2.3,
                input_size: source.full_frame_size(),
                sample_rate: source.sample_rate(),
            };
            source.apply_mapper(Binner::new(config).into_channeled())
        })
        .lift(move |size| DbMapper::new(size).into_channeled())
        .map(move |d| d.map(move |v| normalize_between(*v, 29.0, 56.0)))
        .map(move |d| d.map(move |v| normalize_infs(*v)))
        .lift(move |size| {
            SavitzkyGolayMapper::new(
                size,
                SavitzkyGolayConfig {
                    window_size: 5,
                    polynomial_order: 2,
                },
            )
            .into_channeled()
        })
        .map(move |v| v.map(move |e| constrain_normalized(&e)))
        .compose(move |frames| ExponentialSmoothing::new(frames, SEEK_BACK_LIMIT, ALPHA1))
        .map(channel_avg);

    Ok((frame_src, WavFile::open(file, BUF_SIZE)?))
}

fn channel_avg(v: &Channeled<f64>) -> f64 {
    use Channeled::*;
    match v {
        Stereo(a, b) => (*a + *b) / 2.0,
        Mono(v) => *v,
    }
}

fn normalize_between(v: f64, min: f64, max: f64) -> f64 {
    if v < min {
        0.0
    } else if v > max {
        1.0
    } else {
        (v - min) / (max - min)
    }
}

fn normalize_infs(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        1.0
    } else if v == f64::NEG_INFINITY {
        0.0
    } else {
        v
    }
}

fn constrain_normalized(v: &f64) -> f64 {
    let v = *v;
    if v > 1.0 {
        1.0
    } else if v < 0.0 {
        0.0
    } else {
        v
    }
}

fn draw_frame(canvas: &mut WindowCanvas, frame: &[f64]) -> Result<()> {
    const BIN_MARGIN: u32 = 3;

    canvas.set_draw_color(Color::BLACK);
    canvas.clear();
    let (width, height) = canvas.output_size().map_err(map_sdl_err)?;
    canvas.set_draw_color(Color::GREEN);

    let avail_height = height - (BIN_MARGIN * 2);
    let n_bins = frame.len() as u32;
    let total_margin_used = (n_bins + 1) * BIN_MARGIN;
    let width_per_bin = (width - total_margin_used) / n_bins;
    let mut cur_x = BIN_MARGIN;
    for i in 0..n_bins {
        let lx = cur_x;
        let rx = lx + width_per_bin;
        cur_x = rx + BIN_MARGIN;

        let v = frame[i as usize];
        let mut ty = ((1.0 - v) * (avail_height as f64)) as u32;
        const MIN_HEIGHT: u32 = 4;
        if ty < MIN_HEIGHT {
            ty = MIN_HEIGHT
        }

        let by = avail_height;

        let x = lx as i32;
        let y = ty as i32;
        let width = rx - lx;
        if by < ty {
            panic!(
                "subtract with overflow: {} - {} ... v={}, avail_height={}",
                by, ty, v, avail_height
            )
        }
        let height = by - ty;

        let rect = Rect::new(x, y, width, height);
        canvas.fill_rect(rect).map_err(map_sdl_err)?;
    }

    canvas.present();
    Ok(())
}

fn map_sdl_err(err: String) -> anyhow::Error {
    anyhow::anyhow!("sdl2: {}", err)
}

struct WavPlayer {
    source: WavFile,
}

impl AudioCallback for WavPlayer {
    type Channel = f32;

    fn callback(&mut self, data: &mut [f32]) {
        let mut idx = 0;
        while let Some(sample) = self.source.next_sample().expect("no err") {
            match sample {
                Channeled::Mono(v) => {
                    let v: f64 = v.into();
                    let v = v as f32;
                    data[idx] = v;
                }
                Channeled::Stereo(l, r) => {
                    let l: f64 = l.into();
                    let r: f64 = r.into();
                    let l = l as f32;
                    let r = r as f32;
                    data[idx] = l;
                    idx += 1;
                    data[idx] = r;
                }
            }

            idx += 1;
            if idx == data.len() {
                return;
            }
        }
    }
}
