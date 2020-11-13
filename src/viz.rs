use crate::framed::Framed;
use crate::pipeline::{create_viz_pipeline, open_config_or_default, VizPipelineConfig};
use crate::player::WavPlayer;
use crate::util::log_timed;
use crate::wav::WavFile;
use anyhow::Result;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::WindowCanvas;
use std::ops::{Add, Sub};
use std::time::{Duration, Instant};

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

    let (mut frames, config, wav_src) = log_timed(
        format!("setup visualizer math pipeline for {}", file),
        || create_data_src(file),
    )?;
    let mut wav_player = WavPlayer::new(sdl_context.audio().map_err(map_sdl_err)?, wav_src);

    let mut event_pump = sdl_context.event_pump().map_err(map_sdl_err)?;

    wav_player.play()?;
    let mut paused = false;
    let mut last_frame_for_ts: Option<Instant> = None;
    let frame_delta = Duration::new(0, (1_000_000_000u64 / config.fps) as u32);
    let frame_for_offset = config.data_window() / 2;
    // frame_for_offset += frame_delta.mul_f64(alpha_frame_offset());
    loop {
        let now = Instant::now();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return Ok(()),
                Event::KeyDown {
                    keycode: Some(Keycode::Right),
                    ..
                } => {
                    let mut amount_seek = Duration::from_secs(10);
                    let frames_seek = amount_seek.div_duration_f64(frame_delta).floor() as u32;
                    amount_seek = frame_delta * frames_seek;

                    wav_player.seek(amount_seek)?;
                    frames.seek_frame(frames_seek as isize)?;
                    last_frame_for_ts = Some(now.sub(frame_delta));
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    ..
                } => {
                    if paused {
                        wav_player.play()?;
                    } else {
                        wav_player.stop()?;
                    }

                    paused = !paused;
                }
                _ => {}
            }
        }
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
                last_frame_for_ts = Some(cur_frame_for);
                if !paused {
                    if let Some(frame) = frames.next_frame()? {
                        if status == 0 {
                            draw_frame(&mut canvas, frame)?;
                        }
                    } else {
                        wav_player.stop()?;
                        return Ok(());
                    }
                }
            }
        } else {
            last_frame_for_ts = Some(now.add(frame_for_offset));
        }
    }
}

fn create_data_src(file: &str) -> Result<(impl Framed<f64, WavFile>, VizPipelineConfig, WavFile)> {
    const BUF_SIZE: usize = 32768;

    let config = open_config_or_default()?;
    let frame_src = create_viz_pipeline(WavFile::open(file, BUF_SIZE)?, config)?;
    Ok((frame_src, config, WavFile::open(file, BUF_SIZE)?))
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
        let height = by - ty + 1;

        let rect = Rect::new(x, y, width, height);
        canvas.fill_rect(rect).map_err(map_sdl_err)?;
    }

    canvas.present();
    Ok(())
}

fn map_sdl_err(err: String) -> anyhow::Error {
    anyhow::anyhow!("sdl2: {}", err)
}
