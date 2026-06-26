use crate::channeled::Channeled;
use crate::framed::Framed;
use crate::pipeline::{
    VizColor, VizPipelineConfig, create_debug_fft_pipeline, create_viz_pipeline,
    open_config_or_default,
};
use crate::player::WavPlayer;
use crate::util::{VizFloat, log_timed};
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

#[derive(Clone, Copy)]
struct BarColors {
    shared: VizColor,
    difference: VizColor,
}

#[derive(Clone, Copy)]
struct RenderStyle {
    background_color: VizColor,
    bars: BarColors,
    debug_fft_overlay_color: VizColor,
    high_water_color: VizColor,
}

#[derive(Debug, Default)]
struct HighWaterLines {
    lines: Vec<HighWaterLine>,
    values: Vec<VizFloat>,
}

impl HighWaterLines {
    fn values(&self) -> &[VizFloat] {
        &self.values
    }

    fn reset(&mut self) {
        for line in &mut self.lines {
            line.reset();
        }
        self.values.iter_mut().for_each(|value| *value = 0.0);
    }

    fn update(
        &mut self,
        frame: &[Channeled<VizFloat>],
        elapsed: Duration,
        fall_acceleration: VizFloat,
    ) -> &[VizFloat] {
        self.lines.resize_with(frame.len(), HighWaterLine::default);
        self.values.resize(frame.len(), 0.0);

        for ((line, value), bar) in self
            .lines
            .iter_mut()
            .zip(self.values.iter_mut())
            .zip(frame.iter().copied())
        {
            *value = line.update(bar_heights(bar).peak, elapsed, fall_acceleration);
        }

        &self.values
    }
}

#[derive(Debug, Default)]
struct HighWaterLine {
    value: VizFloat,
    fall_velocity: VizFloat,
}

impl HighWaterLine {
    fn reset(&mut self) {
        self.value = 0.0;
        self.fall_velocity = 0.0;
    }

    fn update(
        &mut self,
        pushed_to: VizFloat,
        elapsed: Duration,
        fall_acceleration: VizFloat,
    ) -> VizFloat {
        let pushed_to = normalized_bar_value(pushed_to);
        if pushed_to >= self.value {
            self.value = pushed_to;
            self.fall_velocity = 0.0;
            return self.value;
        }

        let elapsed = elapsed.as_secs_f64();
        let fall_distance =
            (self.fall_velocity * elapsed) + (0.5 * fall_acceleration * elapsed * elapsed);
        self.fall_velocity += fall_acceleration * elapsed;

        let next_value = self.value - fall_distance;
        if next_value <= pushed_to {
            self.value = pushed_to;
            self.fall_velocity = 0.0;
        } else {
            self.value = normalized_bar_value(next_value);
        }

        self.value
    }
}

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

    let (mut frames, mut debug_frames, config, wav_src) = log_timed(
        format!("setup visualizer math pipeline for {}", file),
        || create_data_src(file),
    )?;
    let mut wav_player = WavPlayer::new(sdl_context.audio().map_err(map_sdl_err)?, wav_src);

    let mut event_pump = sdl_context.event_pump().map_err(map_sdl_err)?;
    let mouse_util = sdl_context.mouse();

    wav_player.play()?;
    let mut paused = false;
    let mut debug_fft_overlay = false;
    let mut last_frame_for_ts: Option<Instant> = None;
    let frame_delta = Duration::new(0, (1_000_000_000u64 / config.fps) as u32);
    let frame_for_offset = config.data_window() / 2;
    let render_style = RenderStyle {
        background_color: config.background_color,
        bars: BarColors {
            shared: config.bar_color,
            difference: config.bar_difference_color,
        },
        debug_fft_overlay_color: config.debug_fft_overlay_color,
        high_water_color: config.high_water_line.color,
    };
    clear_canvas(&mut canvas, render_style.background_color);
    let high_water_fall_acceleration = config.high_water_line.fall_acceleration;
    let mut high_water_lines = HighWaterLines::default();
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
                } => {
                    set_window_mouse_capture(&mut canvas, &mouse_util, false);
                    return Ok(());
                }
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
                    high_water_lines.reset();
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
                    toggle_desktop_fullscreen(&mut canvas, &mouse_util)?;
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
                render_style,
                high_water_lines.values(),
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
                    render_style,
                    high_water_lines.values(),
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
                            render_style,
                            high_water_lines.values(),
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
                                let high_water_values = high_water_lines.update(
                                    &last_frame,
                                    frame_delta,
                                    high_water_fall_acceleration,
                                );
                                draw_frame(
                                    &mut canvas,
                                    &last_frame,
                                    debug_frame,
                                    render_style,
                                    high_water_values,
                                )?;
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
                render_style,
                high_water_lines.values(),
            )?;
            last_frame_for_ts = Some(now.add(frame_for_offset));
        }
    }
}

fn toggle_desktop_fullscreen(
    canvas: &mut WindowCanvas,
    mouse_util: &sdl2::mouse::MouseUtil,
) -> Result<()> {
    let (fullscreen_target, grab_mouse) = match canvas.window().fullscreen_state() {
        FullscreenType::Off => (FullscreenType::Desktop, true),
        FullscreenType::Desktop | FullscreenType::True => (FullscreenType::Off, false),
    };

    canvas
        .window_mut()
        .set_fullscreen(fullscreen_target)
        .map_err(map_sdl_err)?;
    set_window_mouse_capture(canvas, mouse_util, grab_mouse);

    Ok(())
}

fn set_window_mouse_capture(
    canvas: &mut WindowCanvas,
    mouse_util: &sdl2::mouse::MouseUtil,
    enabled: bool,
) {
    canvas.window_mut().set_mouse_grab(enabled);
    mouse_util.show_cursor(!enabled);
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
    impl Framed<Item = Channeled<VizFloat>>,
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

fn clear_canvas(canvas: &mut WindowCanvas, color: VizColor) {
    canvas.set_draw_color(to_sdl_color(color));
    canvas.clear();
    canvas.present();
}

fn draw_cached_frame(
    canvas: &mut WindowCanvas,
    frame: &[Channeled<VizFloat>],
    debug_frame: &[VizFloat],
    debug_fft_overlay: bool,
    render_style: RenderStyle,
    high_water_values: &[VizFloat],
) -> Result<()> {
    if frame.is_empty() {
        clear_canvas(canvas, render_style.background_color);
        return Ok(());
    }

    let debug_frame = if debug_fft_overlay && !debug_frame.is_empty() {
        Some(debug_frame)
    } else {
        None
    };
    draw_frame(canvas, frame, debug_frame, render_style, high_water_values)
}

fn redraw_cached_frame_if_needed(
    redraw_cached_frame: &mut bool,
    canvas: &mut WindowCanvas,
    frame: &[Channeled<VizFloat>],
    debug_frame: &[VizFloat],
    debug_fft_overlay: bool,
    render_style: RenderStyle,
    high_water_values: &[VizFloat],
) -> Result<()> {
    if *redraw_cached_frame {
        *redraw_cached_frame = false;
        draw_cached_frame(
            canvas,
            frame,
            debug_frame,
            debug_fft_overlay,
            render_style,
            high_water_values,
        )?;
    }
    Ok(())
}

fn draw_frame(
    canvas: &mut WindowCanvas,
    frame: &[Channeled<VizFloat>],
    debug_fft_overlay: Option<&[VizFloat]>,
    render_style: RenderStyle,
    high_water_values: &[VizFloat],
) -> Result<()> {
    const BIN_MARGIN: u32 = 3;
    const MIN_HEIGHT: u32 = 4;
    const HIGH_WATER_LINE_HEIGHT: u32 = 1;

    canvas.set_draw_color(to_sdl_color(render_style.background_color));
    canvas.clear();
    let (width, height) = canvas.output_size().map_err(map_sdl_err)?;

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

        let width = rx.saturating_sub(lx);
        draw_bar(
            canvas,
            frame[i as usize],
            lx,
            width,
            avail_height,
            MIN_HEIGHT,
            render_style.bars,
        )?;

        let high_water_value = high_water_values
            .get(i as usize)
            .copied()
            .unwrap_or_else(|| bar_heights(frame[i as usize]).peak);
        draw_high_water_line(
            canvas,
            high_water_value,
            lx,
            width,
            avail_height,
            MIN_HEIGHT,
            HIGH_WATER_LINE_HEIGHT,
            render_style.high_water_color,
        )?;
    }

    if let Some(debug_fft_overlay) = debug_fft_overlay {
        draw_debug_fft_overlay(
            canvas,
            debug_fft_overlay,
            n_bins,
            BIN_MARGIN,
            bar_area_width,
            avail_height,
            render_style.debug_fft_overlay_color,
        )?;
    }

    canvas.present();
    Ok(())
}

fn draw_bar(
    canvas: &mut WindowCanvas,
    value: Channeled<VizFloat>,
    x: u32,
    width: u32,
    avail_height: u32,
    min_top_y: u32,
    colors: BarColors,
) -> Result<()> {
    let heights = bar_heights(value);
    let shared_top = bar_top_y(heights.shared, avail_height, min_top_y);
    let peak_top = bar_top_y(heights.peak, avail_height, min_top_y);
    let bottom = avail_height.saturating_add(1);

    canvas.set_draw_color(to_sdl_color(colors.shared));
    fill_bar_segment(canvas, x, width, shared_top, bottom)?;

    if heights.peak > heights.shared {
        canvas.set_draw_color(to_sdl_color(colors.difference));
        fill_bar_segment(canvas, x, width, peak_top, shared_top)?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BarHeights {
    shared: VizFloat,
    peak: VizFloat,
}

fn bar_heights(value: Channeled<VizFloat>) -> BarHeights {
    use Channeled::*;
    match value {
        Mono(v) => {
            let v = normalized_bar_value(v);
            BarHeights { shared: v, peak: v }
        }
        Stereo(left, right) => {
            let left = normalized_bar_value(left);
            let right = normalized_bar_value(right);
            BarHeights {
                shared: left.min(right),
                peak: left.max(right),
            }
        }
    }
}

fn normalized_bar_value(v: VizFloat) -> VizFloat {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn bar_top_y(value: VizFloat, avail_height: u32, min_top_y: u32) -> u32 {
    let top = ((1.0 - value) * (avail_height as VizFloat)) as u32;
    top.max(min_top_y)
}

fn fill_bar_segment(
    canvas: &mut WindowCanvas,
    x: u32,
    width: u32,
    top: u32,
    bottom: u32,
) -> Result<()> {
    if width == 0 || bottom <= top {
        return Ok(());
    }

    let rect = Rect::new(x as i32, top as i32, width, bottom - top);
    canvas.fill_rect(rect).map_err(map_sdl_err)
}

fn draw_high_water_line(
    canvas: &mut WindowCanvas,
    value: VizFloat,
    x: u32,
    width: u32,
    avail_height: u32,
    min_top_y: u32,
    line_height: u32,
    color: VizColor,
) -> Result<()> {
    if width == 0 || line_height == 0 {
        return Ok(());
    }

    let top = bar_top_y(normalized_bar_value(value), avail_height, min_top_y);
    let bottom = top
        .saturating_add(line_height)
        .min(avail_height.saturating_add(1));

    canvas.set_draw_color(to_sdl_color(color));
    fill_bar_segment(canvas, x, width, top, bottom)
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
    color: VizColor,
) -> Result<()> {
    let Some((min, max)) = finite_range(frame) else {
        return Ok(());
    };

    canvas.set_draw_color(to_sdl_color(color));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_bar_uses_one_height_for_shared_and_peak() {
        assert_eq!(
            bar_heights(Channeled::Mono(0.5)),
            BarHeights {
                shared: 0.5,
                peak: 0.5,
            }
        );
    }

    #[test]
    fn stereo_bar_splits_shared_and_difference_heights() {
        assert_eq!(
            bar_heights(Channeled::Stereo(0.25, 0.75)),
            BarHeights {
                shared: 0.25,
                peak: 0.75,
            }
        );
    }

    #[test]
    fn bar_heights_clamp_non_display_values() {
        assert_eq!(
            bar_heights(Channeled::Stereo(VizFloat::NAN, 1.5)),
            BarHeights {
                shared: 0.0,
                peak: 1.0,
            }
        );
    }

    #[test]
    fn high_water_line_is_pushed_up_immediately() {
        let mut line = HighWaterLine::default();

        let value = line.update(0.8, Duration::from_secs_f64(0.25), 2.0);

        assert_eq!(value, 0.8);
        assert_eq!(line.fall_velocity, 0.0);
    }

    #[test]
    fn high_water_line_falls_with_acceleration() {
        let mut line = HighWaterLine::default();
        line.update(0.8, Duration::ZERO, 2.0);

        let value = line.update(0.0, Duration::from_secs_f64(0.5), 2.0);

        assert!((value - 0.55).abs() < 1e-12);
        assert!((line.fall_velocity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn high_water_line_lands_on_current_peak() {
        let mut line = HighWaterLine::default();
        line.update(0.8, Duration::ZERO, 2.0);

        let value = line.update(0.75, Duration::from_secs_f64(1.0), 2.0);

        assert_eq!(value, 0.75);
        assert_eq!(line.fall_velocity, 0.0);
    }

    #[test]
    fn high_water_lines_track_each_bar_independently() {
        let mut lines = HighWaterLines::default();
        let frame = [Channeled::Mono(0.25), Channeled::Stereo(0.8, 0.2)];

        let values = lines.update(&frame, Duration::ZERO, 2.0);

        assert_eq!(values, &[0.25, 0.8]);

        let frame = [Channeled::Mono(0.0), Channeled::Stereo(0.7, 0.4)];
        let values = lines.update(&frame, Duration::from_secs_f64(0.5), 2.0);

        assert_eq!(values, &[0.0, 0.7]);
    }
}
