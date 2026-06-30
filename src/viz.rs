use crate::channeled::Channeled;
use crate::command::{Command, MAX_VISIBLE_SUGGESTIONS, Palette, ParamDesc, Theme};
use crate::engine::VizEngine;
use crate::font;
use crate::pipeline::{VizColor, VizPipelineConfig, open_config_or_default, validate_config};
use crate::player::WavPlayer;
use crate::util::VizFloat;
use crate::wav::WavFile;
use anyhow::Result;
use sdl2::AudioSubsystem;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::{Point, Rect};
use sdl2::render::{BlendMode, WindowCanvas};
use sdl2::video::FullscreenType;
use std::ops::{Add, Sub};
use std::time::{Duration, Instant};

const BUF_SIZE: usize = 32768;

// --- Overlay (debug UI / palette) styling --------------------------------------------------
const MARGIN: i32 = 12;
const TEXT_BRIGHT: VizColor = VizColor::rgb(236, 238, 248);
const TEXT_DIM: VizColor = VizColor::rgb(150, 154, 176);
const TEXT_ACCENT: VizColor = VizColor::rgb(120, 200, 255);
const TEXT_ERR: VizColor = VizColor::rgb(255, 120, 128);
const TEXT_OK: VizColor = VizColor::rgb(150, 230, 170);
const TOAST_DURATION: Duration = Duration::from_millis(2800);

fn panel_bg() -> Color {
    Color::RGBA(10, 10, 16, 210)
}
fn panel_bg_strong() -> Color {
    Color::RGBA(6, 6, 12, 234)
}
fn highlight_bg() -> Color {
    Color::RGBA(80, 120, 200, 70)
}

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

impl RenderStyle {
    fn from_config(cfg: &VizPipelineConfig) -> Self {
        RenderStyle {
            background_color: cfg.background_color,
            bars: BarColors {
                shared: cfg.bar_color,
                difference: cfg.bar_difference_color,
            },
            debug_fft_overlay_color: cfg.debug_fft_overlay_color,
            high_water_color: cfg.high_water_line.color,
        }
    }
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

/// What the bars are doing right now, for the debug panel label.
fn play_state(engine: &VizEngine, paused: bool) -> &'static str {
    if !engine.is_loaded() {
        "IDLE"
    } else if engine.is_ended() {
        "ENDED"
    } else if paused {
        "PAUSED"
    } else {
        "PLAYING"
    }
}

/// True only while the bars should track live audio (loaded, not paused, not finished).
fn is_playing(engine: &VizEngine, paused: bool) -> bool {
    engine.is_loaded() && !paused && !engine.is_ended()
}

pub fn visualize(file: Option<&str>) -> Result<()> {
    let sdl_context = sdl2::init().map_err(map_sdl_err)?;
    let video_subsystem = sdl_context.video().map_err(map_sdl_err)?;
    let audio = sdl_context.audio().map_err(map_sdl_err)?;
    let (window_width, window_height) = initial_window_size(&video_subsystem);
    let window = video_subsystem
        .window("vis-rs", window_width, window_height)
        .position_centered()
        .resizable()
        .build()?;

    let mut canvas = window.into_canvas().accelerated().build()?;
    canvas.set_blend_mode(BlendMode::Blend);

    let config = open_config_or_default()?;
    let mut engine = match file {
        Some(f) => VizEngine::with_song(config, f)?,
        None => VizEngine::empty(config),
    };
    let mut player: Option<WavPlayer> = match file {
        Some(f) => {
            let mut p = WavPlayer::new(audio.clone(), WavFile::open(f, BUF_SIZE)?);
            p.play()?;
            Some(p)
        }
        None => None,
    };

    let mut event_pump = sdl_context.event_pump().map_err(map_sdl_err)?;
    let mouse_util = sdl_context.mouse();
    let text_input = video_subsystem.text_input();
    text_input.stop();

    let mut paused = false;
    let mut debug_fft_overlay = false;
    let mut debug_ui = true;
    let mut palette = Palette::default();
    let mut toast: Option<(String, Instant)> = None;
    let mut last_frame_for_ts: Option<Instant> = None;
    let mut high_water_lines = HighWaterLines::default();
    let started_at = Instant::now();

    clear_canvas(&mut canvas, engine.config.background_color);

    loop {
        let now = Instant::now();
        let frame_delta = Duration::new(0, (1_000_000_000u64 / engine.config.fps.max(2)) as u32);
        let frame_for_offset = engine.config.data_window() / 2;
        let fall_acceleration = engine.config.high_water_line.fall_acceleration;
        let pause_decay = engine.config.pause_decay_secs;

        let mut quit = false;
        let mut suppress_slash = false;
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => quit = true,
                Event::TextInput { text, .. } if palette.open => {
                    if suppress_slash && text == "/" {
                        suppress_slash = false;
                    } else {
                        palette.insert(&text);
                    }
                }
                Event::KeyDown {
                    keycode: Some(key),
                    repeat,
                    ..
                } => {
                    if palette.open {
                        match key {
                            Keycode::Escape => {
                                palette.close();
                                text_input.stop();
                            }
                            Keycode::Return | Keycode::Return2 | Keycode::KpEnter => {
                                match palette.parse() {
                                    Ok(cmd) => {
                                        match apply_command(
                                            cmd,
                                            &mut engine,
                                            &mut player,
                                            &audio,
                                            &mut paused,
                                            &mut high_water_lines,
                                            &mut last_frame_for_ts,
                                        ) {
                                            Ok(outcome) => {
                                                if outcome.quit {
                                                    quit = true;
                                                }
                                                toast = Some((outcome.message, Instant::now()));
                                                palette.status = None;
                                                palette.close();
                                                text_input.stop();
                                            }
                                            Err(message) => palette.status = Some(message),
                                        }
                                    }
                                    Err(message) => palette.status = Some(message),
                                }
                            }
                            Keycode::Backspace => palette.backspace(),
                            Keycode::Tab => palette.accept(&engine.config),
                            Keycode::Up => {
                                let n = palette.suggestions(&engine.config).len();
                                palette.move_selection(-1, n);
                            }
                            Keycode::Down => {
                                let n = palette.suggestions(&engine.config).len();
                                palette.move_selection(1, n);
                            }
                            _ => {}
                        }
                    } else {
                        match key {
                            Keycode::Escape => quit = true,
                            Keycode::Slash => {
                                palette.open();
                                text_input.start();
                                suppress_slash = true;
                            }
                            Keycode::Tab => debug_ui = !debug_ui,
                            Keycode::Right => {
                                do_seek(&mut engine, &mut player, 10.0)?;
                                high_water_lines.reset();
                                last_frame_for_ts = Some(now.sub(frame_delta));
                            }
                            Keycode::Left => {
                                do_seek(&mut engine, &mut player, -10.0)?;
                                high_water_lines.reset();
                                last_frame_for_ts = Some(now.sub(frame_delta));
                            }
                            Keycode::D | Keycode::R if !repeat => {
                                debug_fft_overlay = !debug_fft_overlay;
                            }
                            Keycode::Space => {
                                if engine.is_loaded() && !engine.is_ended() {
                                    if paused {
                                        if let Some(p) = &mut player {
                                            p.play()?;
                                        }
                                        last_frame_for_ts = Some(Instant::now().sub(frame_delta));
                                        paused = false;
                                    } else {
                                        if let Some(p) = &mut player {
                                            p.stop()?;
                                        }
                                        paused = true;
                                    }
                                }
                            }
                            Keycode::F if !repeat => {
                                toggle_desktop_fullscreen(&mut canvas, &mouse_util)?;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        if quit {
            set_window_mouse_capture(&mut canvas, &mouse_util, false);
            return Ok(());
        }

        let render_style = RenderStyle::from_config(&engine.config);
        let blink_on = (now.duration_since(started_at).as_millis() / 500) % 2 == 0;
        let toast_text = toast
            .as_ref()
            .filter(|(_, at)| now.duration_since(*at) < TOAST_DURATION)
            .map(|(text, _)| text.clone());

        if !is_playing(&engine, paused) {
            // Paused / unloaded / finished: the party stopped, so the bars fall to the floor and
            // the high-water peaks keep falling under gravity until everything settles.
            decay_frame(&mut engine.last_frame, frame_delta, pause_decay);
            let high_water = high_water_lines.update(&engine.last_frame, frame_delta, fall_acceleration);
            render_scene(
                &mut canvas,
                &engine,
                render_style,
                high_water,
                debug_fft_overlay,
                debug_ui,
                &palette,
                paused,
                toast_text.as_deref(),
                blink_on,
            )?;
            last_frame_for_ts = None;
            std::thread::sleep(frame_delta);
            continue;
        }

        if let Some(last_frame_for) = last_frame_for_ts {
            let cur_frame_for = last_frame_for.add(frame_delta);
            let status = frame_status(cur_frame_for, now, frame_delta);

            if status > 0 {
                // Ahead of the audio: just redraw the current frame and wait.
                render_scene(
                    &mut canvas,
                    &engine,
                    render_style,
                    high_water_lines.values(),
                    debug_fft_overlay,
                    debug_ui,
                    &palette,
                    paused,
                    toast_text.as_deref(),
                    blink_on,
                )?;
                std::thread::sleep(frame_delta);
            } else {
                last_frame_for_ts = Some(cur_frame_for);
                if status < 0 {
                    // Behind: advance without drawing to catch back up to the audio.
                    engine.advance()?;
                } else if engine.advance()? {
                    let high_water = high_water_lines.update(
                        &engine.last_frame,
                        frame_delta,
                        fall_acceleration,
                    );
                    render_scene(
                        &mut canvas,
                        &engine,
                        render_style,
                        high_water,
                        debug_fft_overlay,
                        debug_ui,
                        &palette,
                        paused,
                        toast_text.as_deref(),
                        blink_on,
                    )?;
                }
            }
        } else {
            render_scene(
                &mut canvas,
                &engine,
                render_style,
                high_water_lines.values(),
                debug_fft_overlay,
                debug_ui,
                &palette,
                paused,
                toast_text.as_deref(),
                blink_on,
            )?;
            last_frame_for_ts = Some(now.add(frame_for_offset));
        }
    }
}

/// Are we ahead of (>0), behind (<0), or in line with (0) the audio clock?
fn frame_status(cur_frame_for: Instant, cur_audio_at: Instant, frame_delta: Duration) -> i32 {
    if cur_frame_for > cur_audio_at {
        let t_delta = cur_frame_for - cur_audio_at;
        if t_delta > frame_delta {
            t_delta.div_duration_f64(frame_delta) as i32
        } else {
            0
        }
    } else if cur_frame_for < cur_audio_at {
        let t_delta = cur_audio_at - cur_frame_for;
        if t_delta > frame_delta {
            -(t_delta.div_duration_f64(frame_delta) as i32)
        } else {
            0
        }
    } else {
        0
    }
}

/// Exponentially shrink every bar toward the floor, framerate-independently.
fn decay_frame(frame: &mut [Channeled<VizFloat>], dt: Duration, half_life_secs: VizFloat) {
    if frame.is_empty() {
        return;
    }
    let half_life = half_life_secs.max(1e-3);
    let factor = (0.5_f64).powf(dt.as_secs_f64() / half_life);
    for c in frame.iter_mut() {
        *c = c.map(|v| v * factor);
    }
}

fn do_seek(engine: &mut VizEngine, player: &mut Option<WavPlayer>, secs: f64) -> Result<()> {
    let frames = (secs * engine.config.fps as f64).round() as isize;
    engine.seek_frames(frames)?;
    if let Some(p) = player {
        p.seek_secs(secs)?;
    }
    Ok(())
}

/// The result of running a palette command.
struct Outcome {
    quit: bool,
    message: String,
}

fn ok_msg(message: impl Into<String>) -> Result<Outcome, String> {
    Ok(Outcome {
        quit: false,
        message: message.into(),
    })
}

/// Apply a parsed command to the live engine / player. Errors are returned as user-facing
/// strings for the palette to show; the visualizer loop never crashes on a bad command.
fn apply_command(
    cmd: Command,
    engine: &mut VizEngine,
    player: &mut Option<WavPlayer>,
    audio: &AudioSubsystem,
    paused: &mut bool,
    high_water: &mut HighWaterLines,
    last_frame_for_ts: &mut Option<Instant>,
) -> Result<Outcome, String> {
    match cmd {
        Command::Set { key, value } => {
            let desc =
                ParamDesc::find(&key).ok_or_else(|| format!("unknown setting {:?}", key))?;
            let mut new_cfg = engine.config;
            let old_fps = new_cfg.fps;
            desc.apply(&mut new_cfg, &value)?;
            engine.config = validate_config(new_cfg).map_err(|e| e.to_string())?;
            if desc.rebuild {
                if key.eq_ignore_ascii_case("fps") {
                    engine.rescale_position(old_fps, engine.config.fps);
                    *last_frame_for_ts = None;
                }
                engine.reload_and_refresh().map_err(|e| e.to_string())?;
            }
            ok_msg(format!("{} = {}", key, value))
        }
        Command::Theme(name) => {
            let theme = Theme::find(&name)
                .ok_or_else(|| format!("unknown theme {:?} (Tab to list)", name))?;
            let mut new_cfg = engine.config;
            theme.apply(&mut new_cfg);
            engine.config = validate_config(new_cfg).map_err(|e| e.to_string())?;
            ok_msg(format!("theme {}", theme.key))
        }
        Command::Load(path) => {
            engine
                .load(&path)
                .map_err(|e| format!("load failed: {}", e))?;
            let wav = WavFile::open(&path, BUF_SIZE).map_err(|e| e.to_string())?;
            let mut new_player = WavPlayer::new(audio.clone(), wav);
            new_player.play().map_err(|e| e.to_string())?;
            *player = Some(new_player);
            *paused = false;
            high_water.reset();
            *last_frame_for_ts = None;
            ok_msg(format!("loaded {}", short_name(&path)))
        }
        Command::Unload => {
            if let Some(p) = player {
                p.stop().ok();
            }
            *player = None;
            engine.unload();
            *paused = false;
            *last_frame_for_ts = None;
            ok_msg("unloaded")
        }
        Command::Play => {
            if !engine.is_loaded() {
                return Err("no song loaded".into());
            }
            if engine.is_ended() {
                return Err("song ended - seek back or load".into());
            }
            if *paused {
                if let Some(p) = player {
                    p.play().map_err(|e| e.to_string())?;
                }
                *paused = false;
                *last_frame_for_ts = None;
            }
            ok_msg("playing")
        }
        Command::Pause => {
            if !engine.is_loaded() {
                return Err("no song loaded".into());
            }
            if !*paused {
                if let Some(p) = player {
                    p.stop().map_err(|e| e.to_string())?;
                }
                *paused = true;
            }
            ok_msg("paused")
        }
        Command::Toggle => {
            if !engine.is_loaded() {
                return Err("no song loaded".into());
            }
            if *paused {
                if let Some(p) = player {
                    p.play().map_err(|e| e.to_string())?;
                }
                *paused = false;
                *last_frame_for_ts = None;
                ok_msg("playing")
            } else {
                if let Some(p) = player {
                    p.stop().map_err(|e| e.to_string())?;
                }
                *paused = true;
                ok_msg("paused")
            }
        }
        Command::Seek(secs) => {
            if !engine.is_loaded() {
                return Err("no song loaded".into());
            }
            do_seek(engine, player, secs).map_err(|e| e.to_string())?;
            high_water.reset();
            *last_frame_for_ts = None;
            ok_msg(format!("seek {:+}s", secs))
        }
        Command::Reload => {
            let cfg = open_config_or_default().map_err(|e| e.to_string())?;
            engine.config = cfg;
            engine.reload_and_refresh().map_err(|e| e.to_string())?;
            high_water.reset();
            *last_frame_for_ts = None;
            ok_msg("reloaded config")
        }
        Command::Help => {
            ok_msg("set theme load unload play pause toggle seek reload quit")
        }
        Command::Quit => Ok(Outcome {
            quit: true,
            message: "bye".into(),
        }),
    }
}

// --- Scene rendering -----------------------------------------------------------------------

fn render_scene(
    canvas: &mut WindowCanvas,
    engine: &VizEngine,
    render_style: RenderStyle,
    high_water_values: &[VizFloat],
    debug_fft_overlay: bool,
    debug_ui: bool,
    palette: &Palette,
    paused: bool,
    toast: Option<&str>,
    blink_on: bool,
) -> Result<()> {
    canvas.set_draw_color(to_sdl_color(render_style.background_color));
    canvas.clear();

    if !engine.last_frame.is_empty() {
        let debug = if debug_fft_overlay && !engine.last_debug_frame.is_empty() {
            Some(engine.last_debug_frame.as_slice())
        } else {
            None
        };
        draw_bars_scene(canvas, &engine.last_frame, debug, render_style, high_water_values)?;
    }

    if debug_ui {
        draw_debug_ui(canvas, engine, paused)?;
    }
    if palette.open {
        draw_palette(canvas, engine, palette, blink_on)?;
    } else if let Some(text) = toast {
        draw_toast(canvas, text)?;
    }

    canvas.present();
    Ok(())
}

fn draw_panel(canvas: &mut WindowCanvas, x: i32, y: i32, w: i32, h: i32, color: Color) -> Result<()> {
    if w <= 0 || h <= 0 {
        return Ok(());
    }
    canvas.set_draw_color(color);
    canvas
        .fill_rect(Rect::new(x, y, w as u32, h as u32))
        .map_err(map_sdl_err)
}

fn draw_debug_ui(canvas: &mut WindowCanvas, engine: &VizEngine, paused: bool) -> Result<()> {
    let cfg = &engine.config;
    let file = engine
        .loaded_file()
        .map(short_name)
        .unwrap_or_else(|| "(no song)".to_string());
    let lines = vec![
        format!("vis-rs   {}   [Tab hides]", play_state(engine, paused)),
        format!("song   {}", file),
        format!(
            "pos    {}   frame {}   fps {}",
            format_time(engine.position_secs()),
            engine.frame_index(),
            cfg.fps
        ),
        format!(
            "bins   {}   gamma {}   window {}ms",
            cfg.binning.bins, cfg.binning.gamma, cfg.data_window_ms
        ),
        format!("freq   {:.0}..{:.0} Hz", cfg.binning.fmin, cfg.binning.fmax),
        format!(
            "alpha  {} / {}   dB {}..{}",
            cfg.alpha0, cfg.alpha1, cfg.min_db, cfg.max_db
        ),
        format!(
            "smooth {}/{} + {}/{}   levels {}",
            cfg.smoothing0.window_size,
            cfg.smoothing0.degree,
            cfg.smoothing1.window_size,
            cfg.smoothing1.degree,
            cfg.binning.discrete_levels
        ),
        format!(
            "decay  {}s   fall {}",
            cfg.pause_decay_secs, cfg.high_water_line.fall_acceleration
        ),
        "press  /  for the command palette".to_string(),
    ];

    let scale = 2;
    let pad = 6;
    let w = lines.iter().map(|l| font::text_width(scale, l)).max().unwrap_or(0) + pad * 2;
    let h = lines.len() as i32 * font::line_height(scale) + pad * 2;
    draw_panel(canvas, MARGIN, MARGIN, w, h, panel_bg())?;

    let mut y = MARGIN + pad;
    let last = lines.len() - 1;
    for (i, line) in lines.iter().enumerate() {
        let color = if i == 0 {
            TEXT_ACCENT
        } else if i == last {
            TEXT_DIM
        } else {
            TEXT_BRIGHT
        };
        font::draw_text(canvas, MARGIN + pad, y, scale, color, line)?;
        y += font::line_height(scale);
    }
    Ok(())
}

fn draw_palette(
    canvas: &mut WindowCanvas,
    engine: &VizEngine,
    palette: &Palette,
    blink_on: bool,
) -> Result<()> {
    let (win_w, win_h) = canvas.output_size().map_err(map_sdl_err)?;
    let cfg = &engine.config;

    let input_scale = 3;
    let list_scale = 2;
    let pad = 10;

    let suggestions = palette.suggestions(cfg);
    let total = suggestions.len();
    let visible = MAX_VISIBLE_SUGGESTIONS.min(total);
    let has_counter = total > visible;
    let ghost = palette.ghost(cfg);
    let hint = palette.hint(cfg);

    let mut height = pad + font::line_height(input_scale);
    if hint.is_some() {
        height += font::line_height(list_scale);
    }
    if palette.status.is_some() {
        height += font::line_height(list_scale);
    }
    if total > 0 {
        height += 6 + visible as i32 * font::line_height(list_scale);
        if has_counter {
            height += font::line_height(list_scale);
        }
    }
    height += pad;

    let panel_x = MARGIN;
    let panel_w = win_w as i32 - MARGIN * 2;
    let panel_y = win_h as i32 - MARGIN - height;
    draw_panel(canvas, panel_x, panel_y, panel_w, height, panel_bg_strong())?;

    let x = panel_x + pad;
    let mut y = panel_y + pad;

    // Input line: prompt, typed text, dim ghost completion, blinking caret.
    let mut cx = x;
    font::draw_text(canvas, cx, y, input_scale, TEXT_ACCENT, "> ")?;
    cx += 2 * font::advance(input_scale);
    font::draw_text(canvas, cx, y, input_scale, TEXT_BRIGHT, &palette.input)?;
    cx += palette.input.chars().count() as i32 * font::advance(input_scale);
    if let Some(ghost) = &ghost {
        font::draw_text(canvas, cx, y, input_scale, TEXT_DIM, ghost)?;
    }
    if blink_on {
        draw_panel(
            canvas,
            cx,
            y,
            input_scale as i32,
            font::GLYPH_H as i32 * input_scale as i32,
            Color::RGB(TEXT_BRIGHT.r, TEXT_BRIGHT.g, TEXT_BRIGHT.b),
        )?;
    }
    y += font::line_height(input_scale);

    if let Some(hint) = &hint {
        font::draw_text(canvas, x, y, list_scale, TEXT_DIM, hint)?;
        y += font::line_height(list_scale);
    }
    if let Some(status) = &palette.status {
        font::draw_text(canvas, x, y, list_scale, TEXT_ERR, status)?;
        y += font::line_height(list_scale);
    }

    if total > 0 {
        y += 6;
        let selected = palette.selected.min(total - 1);
        let scroll = palette.scroll.min(total - visible);
        let arrow_x = panel_x + panel_w - pad - font::advance(list_scale);
        for row in 0..visible {
            let idx = scroll + row;
            let sugg = &suggestions[idx];
            let is_selected = idx == selected;
            if is_selected {
                draw_panel(
                    canvas,
                    panel_x + 4,
                    y - 2,
                    panel_w - 8,
                    font::line_height(list_scale),
                    highlight_bg(),
                )?;
            }
            let (marker, label_color) = if is_selected {
                ("> ", TEXT_BRIGHT)
            } else {
                ("  ", TEXT_DIM)
            };
            let mut sx = x;
            font::draw_text(canvas, sx, y, list_scale, TEXT_ACCENT, marker)?;
            sx += 2 * font::advance(list_scale);
            font::draw_text(canvas, sx, y, list_scale, label_color, &sugg.label)?;
            let detail_x = sx + 20 * font::advance(list_scale);
            font::draw_text(canvas, detail_x, y, list_scale, TEXT_DIM, &sugg.detail)?;
            // Scroll affordances: a caret when there's more above / below the window.
            if row == 0 && scroll > 0 {
                font::draw_text(canvas, arrow_x, y, list_scale, TEXT_ACCENT, "^")?;
            }
            if row + 1 == visible && scroll + visible < total {
                font::draw_text(canvas, arrow_x, y, list_scale, TEXT_ACCENT, "v")?;
            }
            y += font::line_height(list_scale);
        }
        if has_counter {
            let counter = format!(
                "{}-{} of {}   up/dn to scroll",
                scroll + 1,
                scroll + visible,
                total
            );
            font::draw_text(canvas, x, y, list_scale, TEXT_DIM, &counter)?;
        }
    }

    Ok(())
}

fn draw_toast(canvas: &mut WindowCanvas, text: &str) -> Result<()> {
    let (_win_w, win_h) = canvas.output_size().map_err(map_sdl_err)?;
    let scale = 2;
    let pad = 6;
    let w = font::text_width(scale, text) + pad * 2;
    let h = font::line_height(scale) + pad * 2;
    let x = MARGIN;
    let y = win_h as i32 - MARGIN - h;
    draw_panel(canvas, x, y, w, h, panel_bg())?;
    font::draw_text(canvas, x + pad, y + pad, scale, TEXT_OK, text)?;
    Ok(())
}

fn format_time(secs: f64) -> String {
    let secs = secs.max(0.0);
    let minutes = (secs / 60.0).floor() as u64;
    let seconds = secs - (minutes as f64) * 60.0;
    format!("{}:{:04.1}", minutes, seconds)
}

fn short_name(path: &str) -> String {
    path.rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or(path)
        .to_string()
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

fn clear_canvas(canvas: &mut WindowCanvas, color: VizColor) {
    canvas.set_draw_color(to_sdl_color(color));
    canvas.clear();
    canvas.present();
}

fn draw_bars_scene(
    canvas: &mut WindowCanvas,
    frame: &[Channeled<VizFloat>],
    debug_fft_overlay: Option<&[VizFloat]>,
    render_style: RenderStyle,
    high_water_values: &[VizFloat],
) -> Result<()> {
    const BIN_MARGIN: u32 = 3;
    const MIN_HEIGHT: u32 = 4;
    const HIGH_WATER_LINE_HEIGHT: u32 = 1;

    let (width, height) = canvas.output_size().map_err(map_sdl_err)?;

    let avail_height = height.saturating_sub(BIN_MARGIN * 2);
    let n_bins = frame.len() as u32;
    if n_bins == 0 {
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
    fn decay_shrinks_bars_toward_floor() {
        let mut frame = [Channeled::Mono(1.0), Channeled::Stereo(0.8, 0.4)];
        // One half-life of elapsed time should roughly halve every value.
        decay_frame(&mut frame, Duration::from_secs_f64(0.4), 0.4);
        match frame[0] {
            Channeled::Mono(v) => assert!((v - 0.5).abs() < 1e-9),
            _ => panic!("expected mono"),
        }
        match frame[1] {
            Channeled::Stereo(l, r) => {
                assert!((l - 0.4).abs() < 1e-9);
                assert!((r - 0.2).abs() < 1e-9);
            }
            _ => panic!("expected stereo"),
        }
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
