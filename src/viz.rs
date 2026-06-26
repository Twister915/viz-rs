use crate::framed::Framed;
use crate::pipeline::{
    create_debug_fft_pipeline, create_viz_pipeline, open_config_or_default, VizColor,
    VizPipelineConfig,
};
use crate::player::WavPlayer;
use crate::util::{log_timed, VizFloat};
use crate::wav::WavFile;
use anyhow::Result;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::{Point, Rect};
use sdl2::render::WindowCanvas;
use sdl2::video::FullscreenType;
use std::ops::{Add, Sub};
use std::time::{Duration, Instant};

pub fn visualize(file: &str) -> Result<()> {
    let sdl_context = sdl2::init().map_err(map_sdl_err)?;
    let video_subsystem = sdl_context.video().map_err(map_sdl_err)?;
    let (window_width, window_height) = initial_window_size(&video_subsystem);
    let window = video_subsystem
        .window("vis-rs", window_width, window_height)
        .position_centered()
        .resizable()
        .build()?;

    let mut canvas = window.into_canvas().accelerated().build()?;
    clear_canvas(&mut canvas);

    let (mut frames, mut debug_frames, config, wav_src) = log_timed(
        format!("setup visualizer math pipeline for {}", file),
        || create_data_src(file),
    )?;
    let mut wav_player = WavPlayer::new(sdl_context.audio().map_err(map_sdl_err)?, wav_src);

    let mut event_pump = sdl_context.event_pump().map_err(map_sdl_err)?;

    wav_player.play()?;
    let mut paused = false;
    let mut debug_fft_overlay = false;
    let mut last_frame_for_ts: Option<Instant> = None;
    let frame_delta = Duration::new(0, (1_000_000_000u64 / config.fps) as u32);
    let frame_for_offset = config.data_window() / 2;
    let bar_color = config.bar_color;
    let mut last_frame = Vec::new();
    let mut last_debug_frame = Vec::new();
    let mut redraw_cached_frame = false;
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
                    debug_frames.seek_frame(frames_seek as isize)?;
                    last_frame_for_ts = Some(now.sub(frame_delta));
                }
                Event::KeyDown {
                    keycode: Some(Keycode::D | Keycode::R),
                    repeat: false,
                    ..
                } => {
                    debug_fft_overlay = !debug_fft_overlay;
                    redraw_cached_frame = true;
                    println!(
                        "debug FFT overlay {}",
                        if debug_fft_overlay { "on" } else { "off" }
                    );
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    ..
                } => {
                    if paused {
                        wav_player.play()?;
                        last_frame_for_ts = Some(Instant::now().sub(frame_delta));
                    } else {
                        wav_player.stop()?;
                    }

                    paused = !paused;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::F),
                    repeat: false,
                    ..
                } => {
                    toggle_desktop_fullscreen(&mut canvas)?;
                    redraw_cached_frame = true;
                }
                Event::Window {
                    win_event:
                        WindowEvent::Exposed
                        | WindowEvent::Resized(_, _)
                        | WindowEvent::SizeChanged(_, _)
                        | WindowEvent::Maximized
                        | WindowEvent::Restored,
                    ..
                } => {
                    redraw_cached_frame = true;
                }
                _ => {}
            }
        }

        if paused {
            redraw_cached_frame_if_needed(
                &mut redraw_cached_frame,
                &mut canvas,
                &last_frame,
                &last_debug_frame,
                debug_fft_overlay,
                bar_color,
            )?;
            std::thread::sleep(frame_delta);
            continue;
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
                redraw_cached_frame_if_needed(
                    &mut redraw_cached_frame,
                    &mut canvas,
                    &last_frame,
                    &last_debug_frame,
                    debug_fft_overlay,
                    bar_color,
                )?;
                std::thread::sleep(frame_delta);
            } else {
                last_frame_for_ts = Some(cur_frame_for);
                if !paused {
                    if status < 0 {
                        redraw_cached_frame_if_needed(
                            &mut redraw_cached_frame,
                            &mut canvas,
                            &last_frame,
                            &last_debug_frame,
                            debug_fft_overlay,
                            bar_color,
                        )?;
                    }
                    match (frames.next_frame()?, debug_frames.next_frame()?) {
                        (Some(frame), Some(debug_frame)) => {
                            if status == 0 {
                                redraw_cached_frame = false;
                                last_frame.clear();
                                last_frame.extend_from_slice(frame);
                                last_debug_frame.clear();
                                last_debug_frame.extend_from_slice(debug_frame);
                                let debug_frame = if debug_fft_overlay {
                                    Some(&last_debug_frame[..])
                                } else {
                                    None
                                };
                                draw_frame(&mut canvas, &last_frame, debug_frame, bar_color)?;
                            }
                        }
                        _ => {
                            wav_player.stop()?;
                            return Ok(());
                        }
                    }
                }
            }
        } else {
            redraw_cached_frame_if_needed(
                &mut redraw_cached_frame,
                &mut canvas,
                &last_frame,
                &last_debug_frame,
                debug_fft_overlay,
                bar_color,
            )?;
            last_frame_for_ts = Some(now.add(frame_for_offset));
        }
    }
}

fn toggle_desktop_fullscreen(canvas: &mut WindowCanvas) -> Result<()> {
    let fullscreen_target = match canvas.window().fullscreen_state() {
        FullscreenType::Off => FullscreenType::Desktop,
        FullscreenType::Desktop | FullscreenType::True => FullscreenType::Off,
    };

    canvas
        .window_mut()
        .set_fullscreen(fullscreen_target)
        .map_err(map_sdl_err)
}

fn initial_window_size(video_subsystem: &sdl2::VideoSubsystem) -> (u32, u32) {
    const DEFAULT_WINDOW_SIZE: (u32, u32) = (1280, 720);

    let Ok(display_bounds) = video_subsystem
        .display_usable_bounds(0)
        .or_else(|_| video_subsystem.display_bounds(0))
    else {
        return DEFAULT_WINDOW_SIZE;
    };

    (
        scale_initial_window_dimension(display_bounds.width()),
        scale_initial_window_dimension(display_bounds.height()),
    )
}

fn scale_initial_window_dimension(display_dimension: u32) -> u32 {
    const SCREEN_NUMERATOR: u64 = 3;
    const SCREEN_DENOMINATOR: u64 = 5;

    (((display_dimension as u64) * SCREEN_NUMERATOR) / SCREEN_DENOMINATOR)
        .max(1)
        .min(u32::MAX as u64) as u32
}

fn create_data_src(
    file: &str,
) -> Result<(
    impl Framed<Item = VizFloat>,
    impl Framed<Item = VizFloat>,
    VizPipelineConfig,
    WavFile,
)> {
    const BUF_SIZE: usize = 32768;

    let config = open_config_or_default()?;
    let frame_src = create_viz_pipeline(WavFile::open(file, BUF_SIZE)?, config)?;
    let debug_frame_src = create_debug_fft_pipeline(WavFile::open(file, BUF_SIZE)?, config)?;
    Ok((
        frame_src,
        debug_frame_src,
        config,
        WavFile::open(file, BUF_SIZE)?,
    ))
}

fn clear_canvas(canvas: &mut WindowCanvas) {
    canvas.set_draw_color(Color::BLACK);
    canvas.clear();
    canvas.present();
}

fn draw_cached_frame(
    canvas: &mut WindowCanvas,
    frame: &[VizFloat],
    debug_frame: &[VizFloat],
    debug_fft_overlay: bool,
    bar_color: VizColor,
) -> Result<()> {
    if frame.is_empty() {
        clear_canvas(canvas);
        return Ok(());
    }

    let debug_frame = if debug_fft_overlay && !debug_frame.is_empty() {
        Some(debug_frame)
    } else {
        None
    };
    draw_frame(canvas, frame, debug_frame, bar_color)
}

fn redraw_cached_frame_if_needed(
    redraw_cached_frame: &mut bool,
    canvas: &mut WindowCanvas,
    frame: &[VizFloat],
    debug_frame: &[VizFloat],
    debug_fft_overlay: bool,
    bar_color: VizColor,
) -> Result<()> {
    if *redraw_cached_frame {
        *redraw_cached_frame = false;
        draw_cached_frame(canvas, frame, debug_frame, debug_fft_overlay, bar_color)?;
    }
    Ok(())
}

fn draw_frame(
    canvas: &mut WindowCanvas,
    frame: &[VizFloat],
    debug_fft_overlay: Option<&[VizFloat]>,
    bar_color: VizColor,
) -> Result<()> {
    const BIN_MARGIN: u32 = 3;
    const MIN_HEIGHT: u32 = 4;

    canvas.set_draw_color(Color::BLACK);
    canvas.clear();
    let (width, height) = canvas.output_size().map_err(map_sdl_err)?;
    canvas.set_draw_color(to_sdl_color(bar_color));

    let avail_height = height.saturating_sub(BIN_MARGIN * 2);
    let n_bins = frame.len() as u32;
    if n_bins == 0 {
        canvas.present();
        return Ok(());
    }

    let total_margin_used = (n_bins + 1).saturating_mul(BIN_MARGIN);
    let bar_area_width = width.saturating_sub(total_margin_used);
    for i in 0..n_bins {
        let Some((lx, rx)) = bar_bounds(i, n_bins, BIN_MARGIN, bar_area_width) else {
            continue;
        };

        let v = frame[i as usize];
        let v = if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut ty = ((1.0 - v) * (avail_height as VizFloat)) as u32;
        if ty < MIN_HEIGHT {
            ty = MIN_HEIGHT
        }

        let by = avail_height;

        let x = lx as i32;
        let y = ty as i32;
        let width = rx.saturating_sub(lx);
        let height = by.saturating_sub(ty) + 1;

        let rect = Rect::new(x, y, width, height);
        canvas.fill_rect(rect).map_err(map_sdl_err)?;
    }

    if let Some(debug_fft_overlay) = debug_fft_overlay {
        draw_debug_fft_overlay(
            canvas,
            debug_fft_overlay,
            n_bins,
            BIN_MARGIN,
            bar_area_width,
            avail_height,
        )?;
    }

    canvas.present();
    Ok(())
}

fn bar_bounds(i: u32, n_bins: u32, bin_margin: u32, bar_area_width: u32) -> Option<(u32, u32)> {
    let lx = (i + 1)
        .saturating_mul(bin_margin)
        .saturating_add((i.saturating_mul(bar_area_width)) / n_bins);
    let rx = (i + 1)
        .saturating_mul(bin_margin)
        .saturating_add(((i + 1).saturating_mul(bar_area_width)) / n_bins);
    (rx > lx).then_some((lx, rx))
}

fn to_sdl_color(color: VizColor) -> Color {
    Color::RGB(color.r, color.g, color.b)
}

fn draw_debug_fft_overlay(
    canvas: &mut WindowCanvas,
    frame: &[VizFloat],
    n_bins: u32,
    bin_margin: u32,
    bar_area_width: u32,
    avail_height: u32,
) -> Result<()> {
    let Some((min, max)) = finite_range(frame) else {
        return Ok(());
    };

    canvas.set_draw_color(Color::RGB(255, 48, 48));
    let point_count = frame.len().min(n_bins as usize);
    let mut prev = None;
    for (idx, v) in frame.iter().copied().take(point_count).enumerate() {
        if !v.is_finite() {
            prev = None;
            continue;
        }

        let Some((lx, rx)) = bar_bounds(idx as u32, n_bins, bin_margin, bar_area_width) else {
            prev = None;
            continue;
        };
        let x = lx + ((rx - lx) / 2);
        let y = scale_debug_y(v, min, max, avail_height);
        let point = Point::new(x as i32, y);

        if let Some(prev) = prev {
            canvas.draw_line(prev, point).map_err(map_sdl_err)?;
        } else {
            canvas.draw_point(point).map_err(map_sdl_err)?;
        }
        prev = Some(point);
    }

    Ok(())
}

fn finite_range(values: &[VizFloat]) -> Option<(VizFloat, VizFloat)> {
    values.iter().copied().filter(|v| v.is_finite()).fold(
        None,
        |range: Option<(VizFloat, VizFloat)>, v| match range {
            Some((min, max)) => Some((min.min(v), max.max(v))),
            None => Some((v, v)),
        },
    )
}

fn scale_debug_y(v: VizFloat, min: VizFloat, max: VizFloat, avail_height: u32) -> i32 {
    let range = max - min;
    let normalized = if range <= VizFloat::EPSILON {
        0.5
    } else {
        ((v - min) / range).clamp(0.0, 1.0)
    };
    ((1.0 - normalized) * (avail_height as VizFloat)).round() as i32
}

fn map_sdl_err(err: String) -> anyhow::Error {
    anyhow::anyhow!("sdl2: {}", err)
}
