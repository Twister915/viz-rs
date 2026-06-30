//! The "commander": a runtime command palette (open with `/`) that can load/unload songs,
//! switch themes, and tweak *any* pipeline parameter live.
//!
//! This module is deliberately free of SDL so it stays pure and testable: it owns the input
//! string, the autocomplete model, and parsing into a [`Command`]. The visualizer loop is what
//! actually renders the palette and applies the resulting [`Command`].

use crate::pipeline::{VizColor, VizPipelineConfig, parse_hex_color};
use crate::util::VizFloat;

// ---------------------------------------------------------------------------
// Settable parameters
// ---------------------------------------------------------------------------

/// A single live-tunable setting: how to read it, how to parse+write it, and whether changing
/// it forces a pipeline rebuild (everything except colors and the two render-time knobs does).
pub struct ParamDesc {
    pub key: &'static str,
    pub hint: &'static str,
    /// Whether changing this value requires rebuilding the math pipeline.
    pub rebuild: bool,
    get: fn(&VizPipelineConfig) -> String,
    set: fn(&mut VizPipelineConfig, &str) -> Result<(), String>,
}

impl ParamDesc {
    pub fn current(&self, cfg: &VizPipelineConfig) -> String {
        (self.get)(cfg)
    }

    pub fn apply(&self, cfg: &mut VizPipelineConfig, raw: &str) -> Result<(), String> {
        (self.set)(cfg, raw)
    }

    pub fn find(key: &str) -> Option<&'static ParamDesc> {
        PARAMS.iter().find(|p| p.key.eq_ignore_ascii_case(key))
    }

    pub fn matching(prefix: &str) -> impl Iterator<Item = &'static ParamDesc> {
        let prefix = prefix.to_ascii_lowercase();
        PARAMS
            .iter()
            .filter(move |p| p.key.to_ascii_lowercase().starts_with(&prefix))
    }
}

fn fmt_color(c: VizColor) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)
}

fn pf(s: &str) -> Result<VizFloat, String> {
    s.trim()
        .parse::<VizFloat>()
        .map_err(|_| format!("expected a number, got {:?}", s.trim()))
}

fn pu64(s: &str) -> Result<u64, String> {
    s.trim()
        .parse::<u64>()
        .map_err(|_| format!("expected a whole number, got {:?}", s.trim()))
}

fn pusize(s: &str) -> Result<usize, String> {
    s.trim()
        .parse::<usize>()
        .map_err(|_| format!("expected a whole number, got {:?}", s.trim()))
}

fn pu32(s: &str) -> Result<u32, String> {
    s.trim()
        .parse::<u32>()
        .map_err(|_| format!("expected a whole number, got {:?}", s.trim()))
}

/// Every setting the palette can change. Order here is the order shown in the debug UI / help.
pub static PARAMS: &[ParamDesc] = &[
    // --- render-time knobs (no rebuild) ---
    ParamDesc {
        key: "background_color",
        hint: "hex #rrggbb",
        rebuild: false,
        get: |c| fmt_color(c.background_color),
        set: |c, s| {
            c.background_color = parse_hex_color(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "bar_color",
        hint: "hex #rrggbb",
        rebuild: false,
        get: |c| fmt_color(c.bar_color),
        set: |c, s| {
            c.bar_color = parse_hex_color(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "bar_difference_color",
        hint: "hex #rrggbb",
        rebuild: false,
        get: |c| fmt_color(c.bar_difference_color),
        set: |c, s| {
            c.bar_difference_color = parse_hex_color(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "debug_overlay_color",
        hint: "hex #rrggbb",
        rebuild: false,
        get: |c| fmt_color(c.debug_fft_overlay_color),
        set: |c, s| {
            c.debug_fft_overlay_color = parse_hex_color(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "high_water_color",
        hint: "hex #rrggbb",
        rebuild: false,
        get: |c| fmt_color(c.high_water_line.color),
        set: |c, s| {
            c.high_water_line.color = parse_hex_color(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "fall_acceleration",
        hint: "peak fall rate >= 0",
        rebuild: false,
        get: |c| c.high_water_line.fall_acceleration.to_string(),
        set: |c, s| {
            c.high_water_line.fall_acceleration = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "pause_decay",
        hint: "seconds > 0 (bar fall on pause)",
        rebuild: false,
        get: |c| c.pause_decay_secs.to_string(),
        set: |c, s| {
            c.pause_decay_secs = pf(s)?;
            Ok(())
        },
    },
    // --- temporal / dynamics (rebuild) ---
    ParamDesc {
        key: "alpha0",
        hint: "0.0-1.0 (1st time smoothing)",
        rebuild: true,
        get: |c| c.alpha0.to_string(),
        set: |c, s| {
            c.alpha0 = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "alpha1",
        hint: "0.0-1.0 (2nd time smoothing)",
        rebuild: true,
        get: |c| c.alpha1.to_string(),
        set: |c, s| {
            c.alpha1 = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "min_db",
        hint: "dB floor (e.g. -60)",
        rebuild: true,
        get: |c| c.min_db.to_string(),
        set: |c, s| {
            c.min_db = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "max_db",
        hint: "dB ceiling (e.g. -10)",
        rebuild: true,
        get: |c| c.max_db.to_string(),
        set: |c, s| {
            c.max_db = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "fps",
        hint: "frames/sec > 1",
        rebuild: true,
        get: |c| c.fps.to_string(),
        set: |c, s| {
            c.fps = pu64(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "data_window_ms",
        hint: "FFT window ms > 1",
        rebuild: true,
        get: |c| c.data_window_ms.to_string(),
        set: |c, s| {
            c.data_window_ms = pu64(s)?;
            Ok(())
        },
    },
    // --- binning (rebuild) ---
    ParamDesc {
        key: "bins",
        hint: "bar count > 1",
        rebuild: true,
        get: |c| c.binning.bins.to_string(),
        set: |c, s| {
            c.binning.bins = pusize(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "fmin",
        hint: "low edge Hz",
        rebuild: true,
        get: |c| c.binning.fmin.to_string(),
        set: |c, s| {
            c.binning.fmin = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "fmax",
        hint: "high edge Hz",
        rebuild: true,
        get: |c| c.binning.fmax.to_string(),
        set: |c, s| {
            c.binning.fmax = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "gamma",
        hint: "bar spacing curve > 0",
        rebuild: true,
        get: |c| c.binning.gamma.to_string(),
        set: |c, s| {
            c.binning.gamma = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "discrete_levels",
        hint: "quantization levels > 2",
        rebuild: true,
        get: |c| c.binning.discrete_levels.to_string(),
        set: |c, s| {
            c.binning.discrete_levels = pu32(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "bandwidth_weight",
        hint: "0.0-1.0 energy vs density",
        rebuild: true,
        get: |c| c.binning.compensation.bandwidth_weight.to_string(),
        set: |c, s| {
            c.binning.compensation.bandwidth_weight = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "spectral_tilt",
        hint: "dB per octave",
        rebuild: true,
        get: |c| c.binning.compensation.spectral_tilt_db_per_octave.to_string(),
        set: |c, s| {
            c.binning.compensation.spectral_tilt_db_per_octave = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "tilt_reference_hz",
        hint: "Hz > 0",
        rebuild: true,
        get: |c| c.binning.compensation.tilt_reference_hz.to_string(),
        set: |c, s| {
            c.binning.compensation.tilt_reference_hz = pf(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "max_adjustment_db",
        hint: "dB clamp >= 0",
        rebuild: true,
        get: |c| c.binning.compensation.max_adjustment_db.to_string(),
        set: |c, s| {
            c.binning.compensation.max_adjustment_db = pf(s)?;
            Ok(())
        },
    },
    // --- spatial smoothing (rebuild) ---
    ParamDesc {
        key: "smoothing0_window",
        hint: "odd > 2",
        rebuild: true,
        get: |c| c.smoothing0.window_size.to_string(),
        set: |c, s| {
            c.smoothing0.window_size = pu64(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "smoothing0_degree",
        hint: "polynomial degree > 0",
        rebuild: true,
        get: |c| c.smoothing0.degree.to_string(),
        set: |c, s| {
            c.smoothing0.degree = pu64(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "smoothing0_order",
        hint: "derivative order (usually 0)",
        rebuild: true,
        get: |c| c.smoothing0.order.to_string(),
        set: |c, s| {
            c.smoothing0.order = pu64(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "smoothing1_window",
        hint: "odd > 2",
        rebuild: true,
        get: |c| c.smoothing1.window_size.to_string(),
        set: |c, s| {
            c.smoothing1.window_size = pu64(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "smoothing1_degree",
        hint: "polynomial degree > 0",
        rebuild: true,
        get: |c| c.smoothing1.degree.to_string(),
        set: |c, s| {
            c.smoothing1.degree = pu64(s)?;
            Ok(())
        },
    },
    ParamDesc {
        key: "smoothing1_order",
        hint: "derivative order (usually 0)",
        rebuild: true,
        get: |c| c.smoothing1.order.to_string(),
        set: |c, s| {
            c.smoothing1.order = pu64(s)?;
            Ok(())
        },
    },
];

// ---------------------------------------------------------------------------
// Themes
// ---------------------------------------------------------------------------

/// A named color preset (the themes that used to be commented blocks in `config.yml`).
pub struct Theme {
    pub key: &'static str,
    background: &'static str,
    bar: &'static str,
    difference: &'static str,
    overlay: &'static str,
    high_water: &'static str,
}

impl Theme {
    /// Apply the theme's colors onto a config (the only fields a theme touches).
    pub fn apply(&self, cfg: &mut VizPipelineConfig) {
        // Theme constants are known-valid 6-digit hex, so parse failures are impossible here.
        if let Ok(c) = parse_hex_color(self.background) {
            cfg.background_color = c;
        }
        if let Ok(c) = parse_hex_color(self.bar) {
            cfg.bar_color = c;
        }
        if let Ok(c) = parse_hex_color(self.difference) {
            cfg.bar_difference_color = c;
        }
        if let Ok(c) = parse_hex_color(self.overlay) {
            cfg.debug_fft_overlay_color = c;
        }
        if let Ok(c) = parse_hex_color(self.high_water) {
            cfg.high_water_line.color = c;
        }
    }

    pub fn find(key: &str) -> Option<&'static Theme> {
        THEMES.iter().find(|t| t.key.eq_ignore_ascii_case(key))
    }

    pub fn matching(prefix: &str) -> impl Iterator<Item = &'static Theme> {
        let prefix = prefix.to_ascii_lowercase();
        THEMES
            .iter()
            .filter(move |t| t.key.to_ascii_lowercase().starts_with(&prefix))
    }
}

macro_rules! theme {
    ($key:literal, $bg:literal, $bar:literal, $diff:literal, $ov:literal, $hw:literal) => {
        Theme {
            key: $key,
            background: $bg,
            bar: $bar,
            difference: $diff,
            overlay: $ov,
            high_water: $hw,
        }
    };
}

pub static THEMES: &[Theme] = &[
    theme!("calm-blue-sky", "#d6ecfc", "#4f9fd1", "#b9e7ff", "#5f87a5", "#f8fcff"),
    theme!("pale-dawn", "#edf7fb", "#6faed6", "#fde6c8", "#88a7bc", "#fffaf0"),
    theme!("deep-evening-sky", "#102a43", "#7cc7e8", "#d8f3ff", "#9ab6ca", "#f3fbff"),
    theme!("frosted-window", "#e6f1f6", "#7aa6bd", "#c7eef6", "#6f8795", "#ffffff"),
    theme!("rain-washed-glass", "#b6ccd8", "#476f8f", "#a9d8ed", "#33546d", "#eef8ff"),
    theme!("neon-salmon", "#0a0510", "#fa8072", "#ff1b8b", "#5a2740", "#ffd0e0"),
    theme!("dark-songs", "#07080d", "#8a5cf6", "#e43f5a", "#3b4261", "#f5f0ff"),
    theme!("happy-songs", "#fff7b8", "#00a878", "#ffb703", "#5aa9e6", "#ffffff"),
    theme!("fast-songs", "#050b1a", "#00e5ff", "#ff2d75", "#f8ff00", "#ffffff"),
    theme!("slow-songs", "#1c2533", "#7f9bb3", "#d7e8f7", "#586f86", "#edf6ff"),
    theme!("piano-songs", "#f7f3ec", "#111827", "#b28f62", "#7b705f", "#ffffff"),
    theme!("love-songs", "#2b0f1f", "#e85d91", "#ffc2d4", "#a8466f", "#fff1f7"),
    theme!("hot-songs", "#1a0800", "#ff5a1f", "#ffd166", "#c44512", "#fff3c4"),
    theme!("cold-songs", "#06131f", "#5cc8ff", "#d8f7ff", "#2f7ea8", "#f7fdff"),
];

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

pub struct CommandDesc {
    pub key: &'static str,
    pub usage: &'static str,
}

pub static COMMANDS: &[CommandDesc] = &[
    CommandDesc { key: "set", usage: "set <param> <value>  -  change a setting live" },
    CommandDesc { key: "theme", usage: "theme <name>  -  apply a color theme" },
    CommandDesc { key: "load", usage: "load <path.wav>  -  load & play a song" },
    CommandDesc { key: "unload", usage: "unload  -  stop & clear the song (bars fall)" },
    CommandDesc { key: "play", usage: "play  -  resume playback" },
    CommandDesc { key: "pause", usage: "pause  -  pause (bars fall to the floor)" },
    CommandDesc { key: "toggle", usage: "toggle  -  toggle play / pause" },
    CommandDesc { key: "seek", usage: "seek <seconds>  -  jump by +/- seconds" },
    CommandDesc { key: "reload", usage: "reload  -  reload config.yml from disk" },
    CommandDesc { key: "help", usage: "help  -  list commands" },
    CommandDesc { key: "quit", usage: "quit  -  exit the visualizer" },
];

/// A parsed, ready-to-apply command produced by [`Palette::parse`].
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Set { key: String, value: String },
    Theme(String),
    Load(String),
    Unload,
    Play,
    Pause,
    Toggle,
    Seek(f64),
    Reload,
    Help,
    Quit,
}

// ---------------------------------------------------------------------------
// Autocomplete suggestions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// Canonical token inserted when this suggestion is accepted (Tab).
    pub text: String,
    /// Primary label shown in the suggestion list.
    pub label: String,
    /// Dim detail shown after the label (usage, current value, etc.).
    pub detail: String,
    /// Whether accepting should append a space (false for directories, mid-path).
    pub append_space: bool,
}

impl Suggestion {
    fn simple(text: &str, detail: &str) -> Self {
        Suggestion {
            text: text.to_string(),
            label: text.to_string(),
            detail: detail.to_string(),
            append_space: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Palette (input state machine)
// ---------------------------------------------------------------------------

/// How many suggestion rows are visible at once; the rest scroll into view.
pub const MAX_VISIBLE_SUGGESTIONS: usize = 8;

pub struct Palette {
    pub open: bool,
    pub input: String,
    pub selected: usize,
    /// Index of the first visible suggestion row (the scroll window's top).
    pub scroll: usize,
    /// Last result / error message, shown as a small toast below the bars.
    pub status: Option<String>,
}

impl Default for Palette {
    fn default() -> Self {
        Palette {
            open: false,
            input: String::new(),
            selected: 0,
            scroll: 0,
            status: None,
        }
    }
}

impl Palette {
    pub fn open(&mut self) {
        self.open = true;
        self.input.clear();
        self.selected = 0;
        self.scroll = 0;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.input.clear();
        self.selected = 0;
        self.scroll = 0;
    }

    pub fn insert(&mut self, text: &str) {
        for ch in text.chars() {
            if !ch.is_control() {
                self.input.push(ch);
            }
        }
        self.selected = 0;
        self.scroll = 0;
    }

    pub fn backspace(&mut self) {
        self.input.pop();
        self.selected = 0;
        self.scroll = 0;
    }

    pub fn move_selection(&mut self, delta: isize, len: usize) {
        if len == 0 {
            self.selected = 0;
            self.scroll = 0;
            return;
        }
        let signed_len = len as isize;
        let mut next = self.selected as isize + delta;
        next = ((next % signed_len) + signed_len) % signed_len;
        self.selected = next as usize;
        self.ensure_visible(len);
    }

    /// Scroll the visible window the minimum amount needed to keep `selected` on screen.
    fn ensure_visible(&mut self, len: usize) {
        let visible = MAX_VISIBLE_SUGGESTIONS;
        if len <= visible {
            self.scroll = 0;
            return;
        }
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + visible {
            self.scroll = self.selected + 1 - visible;
        }
        self.scroll = self.scroll.min(len - visible);
    }

    /// Split the input into the already-complete leading words and the word being typed now.
    fn split(&self) -> (Vec<&str>, &str) {
        let trailing_space = self
            .input
            .chars()
            .last()
            .map(|c| c.is_whitespace())
            .unwrap_or(false);
        let words: Vec<&str> = self.input.split_whitespace().collect();
        if trailing_space || words.is_empty() {
            (words, "")
        } else {
            let last = words[words.len() - 1];
            (words[..words.len() - 1].to_vec(), last)
        }
    }

    /// Compute the suggestion list for the word currently being typed.
    pub fn suggestions(&self, cfg: &VizPipelineConfig) -> Vec<Suggestion> {
        let (prior, current) = self.split();

        // Completing the command name.
        if prior.is_empty() {
            return COMMANDS
                .iter()
                .filter(|c| c.key.starts_with(&current.to_ascii_lowercase()))
                .map(|c| Suggestion::simple(c.key, c.usage))
                .collect();
        }

        match prior[0].to_ascii_lowercase().as_str() {
            "set" => match prior.len() {
                1 => ParamDesc::matching(current)
                    .map(|p| Suggestion {
                        text: p.key.to_string(),
                        label: p.key.to_string(),
                        detail: format!("now {}  ({})", p.current(cfg), p.hint),
                        append_space: true,
                    })
                    .collect(),
                2 => {
                    // Value stage: offer the current value as an editable starting point.
                    if let Some(p) = ParamDesc::find(prior[1]) {
                        vec![Suggestion {
                            text: p.current(cfg),
                            label: p.current(cfg),
                            detail: format!("current {} ({})", prior[1], p.hint),
                            append_space: true,
                        }]
                    } else {
                        Vec::new()
                    }
                }
                _ => Vec::new(),
            },
            "theme" if prior.len() == 1 => Theme::matching(current)
                .map(|t| Suggestion::simple(t.key, "color theme"))
                .collect(),
            "load" if prior.len() == 1 => file_suggestions(current),
            _ => Vec::new(),
        }
    }

    /// The dim "ghost" completion shown inline after the typed text (the untyped tail of the top
    /// suggestion), mimicking a web placeholder. Returns `None` when there's nothing to ghost.
    pub fn ghost(&self, cfg: &VizPipelineConfig) -> Option<String> {
        let (_, current) = self.split();
        let suggestions = self.suggestions(cfg);
        let top = suggestions.get(self.selected.min(suggestions.len().saturating_sub(1)))?;
        if current.is_empty() {
            // Nothing typed for this word yet: ghost the whole token (e.g. the current value).
            Some(top.text.clone())
        } else if top
            .text
            .to_ascii_lowercase()
            .starts_with(&current.to_ascii_lowercase())
            && top.text.len() > current.len()
        {
            Some(top.text[current.len()..].to_string())
        } else {
            None
        }
    }

    /// A one-line hint describing what the current word expects.
    pub fn hint(&self, _cfg: &VizPipelineConfig) -> Option<String> {
        let (prior, _current) = self.split();
        if prior.is_empty() {
            return Some("type a command, Tab to complete, Enter to run".to_string());
        }
        match prior[0].to_ascii_lowercase().as_str() {
            "set" => match prior.len() {
                1 => Some("which setting?".to_string()),
                2 => ParamDesc::find(prior[1]).map(|p| format!("value: {}", p.hint)),
                _ => None,
            },
            "theme" if prior.len() == 1 => Some("which theme?".to_string()),
            "load" if prior.len() == 1 => Some("path to a .wav file".to_string()),
            "seek" if prior.len() == 1 => Some("seconds, e.g. 30 or -10".to_string()),
            other => COMMANDS
                .iter()
                .find(|c| c.key == other)
                .map(|c| c.usage.to_string()),
        }
    }

    /// Accept the highlighted suggestion (Tab): replace the current word with it.
    pub fn accept(&mut self, cfg: &VizPipelineConfig) {
        let suggestions = self.suggestions(cfg);
        if suggestions.is_empty() {
            return;
        }
        let pick = &suggestions[self.selected.min(suggestions.len() - 1)];
        let trailing_space = self
            .input
            .chars()
            .last()
            .map(|c| c.is_whitespace())
            .unwrap_or(false);
        if !trailing_space {
            // Strip the partial word being typed.
            let keep = self.input.trim_end_matches(|c: char| !c.is_whitespace());
            self.input.truncate(keep.len());
        }
        self.input.push_str(&pick.text);
        if pick.append_space {
            self.input.push(' ');
        }
        self.selected = 0;
    }

    /// Parse the current input into a [`Command`]. Returns a human error string on failure.
    pub fn parse(&self) -> Result<Command, String> {
        let trimmed = self.input.trim();
        if trimmed.is_empty() {
            return Err("type a command (try: help)".to_string());
        }
        let mut words = trimmed.split_whitespace();
        let cmd = words.next().unwrap().to_ascii_lowercase();
        let rest = trimmed[cmd.len()..].trim_start();

        match cmd.as_str() {
            "set" => {
                let mut it = rest.splitn(2, char::is_whitespace);
                let key = it.next().unwrap_or("").trim();
                let value = it.next().unwrap_or("").trim();
                if key.is_empty() {
                    return Err("usage: set <param> <value>".to_string());
                }
                if ParamDesc::find(key).is_none() {
                    return Err(format!("unknown setting {:?}", key));
                }
                if value.is_empty() {
                    return Err(format!("set {} needs a value", key));
                }
                Ok(Command::Set {
                    key: key.to_string(),
                    value: value.to_string(),
                })
            }
            "theme" => {
                if rest.is_empty() {
                    return Err("usage: theme <name>".to_string());
                }
                Ok(Command::Theme(rest.to_string()))
            }
            "load" => {
                if rest.is_empty() {
                    return Err("usage: load <path.wav>".to_string());
                }
                Ok(Command::Load(rest.to_string()))
            }
            "unload" => Ok(Command::Unload),
            "play" => Ok(Command::Play),
            "pause" => Ok(Command::Pause),
            "toggle" => Ok(Command::Toggle),
            "seek" => {
                let secs = rest
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| "usage: seek <seconds> (e.g. 30 or -10)".to_string())?;
                Ok(Command::Seek(secs))
            }
            "reload" => Ok(Command::Reload),
            "help" => Ok(Command::Help),
            "quit" | "exit" => Ok(Command::Quit),
            other => Err(format!("unknown command {:?} (try: help)", other)),
        }
    }
}

/// Filesystem autocomplete for `load <path>`: list entries in the directory of the partial path,
/// preferring `.wav` files and directories.
fn file_suggestions(current: &str) -> Vec<Suggestion> {
    let sep = |c: char| c == '/' || c == '\\';
    let (dir_part, prefix) = match current.rfind(sep) {
        Some(i) => (&current[..=i], &current[i + 1..]),
        None => ("", current),
    };
    let read_path = if dir_part.is_empty() { "." } else { dir_part };
    let prefix_lower = prefix.to_ascii_lowercase();

    let Ok(entries) = std::fs::read_dir(read_path) else {
        return Vec::new();
    };

    let mut out: Vec<Suggestion> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if !name.to_ascii_lowercase().starts_with(&prefix_lower) {
            continue;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let is_wav = name.to_ascii_lowercase().ends_with(".wav");
        if !is_dir && !is_wav {
            continue;
        }
        out.push(Suggestion {
            text: format!("{}{}", dir_part, name),
            label: name.clone(),
            detail: if is_dir { "dir".to_string() } else { "wav".to_string() },
            append_space: !is_dir,
        });
    }
    // Directories first, then wavs, alphabetical within each.
    out.sort_by(|a, b| {
        let a_dir = a.detail == "dir";
        let b_dir = b.detail == "dir";
        b_dir.cmp(&a_dir).then_with(|| a.label.cmp(&b.label))
    });
    out.truncate(40);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::embedded_default_config;

    #[test]
    fn every_param_round_trips_through_get_then_set() {
        let mut cfg = embedded_default_config();
        for p in PARAMS {
            let current = p.current(&cfg);
            p.apply(&mut cfg, &current)
                .unwrap_or_else(|e| panic!("param {} failed to re-apply {:?}: {}", p.key, current, e));
        }
    }

    #[test]
    fn parses_set_command() {
        let mut pal = Palette::default();
        pal.input = "set gamma 3.5".to_string();
        assert_eq!(
            pal.parse().unwrap(),
            Command::Set {
                key: "gamma".to_string(),
                value: "3.5".to_string()
            }
        );
    }

    #[test]
    fn rejects_unknown_setting() {
        let mut pal = Palette::default();
        pal.input = "set wobble 3".to_string();
        assert!(pal.parse().is_err());
    }

    #[test]
    fn ghost_completes_command_prefix() {
        let cfg = embedded_default_config();
        let mut pal = Palette::default();
        pal.input = "th".to_string();
        assert_eq!(pal.ghost(&cfg).as_deref(), Some("eme"));
    }

    #[test]
    fn accept_completes_param_name() {
        let cfg = embedded_default_config();
        let mut pal = Palette::default();
        pal.input = "set alph".to_string();
        pal.accept(&cfg);
        assert_eq!(pal.input, "set alpha0 ");
    }

    #[test]
    fn seek_parses_negative() {
        let mut pal = Palette::default();
        pal.input = "seek -10".to_string();
        assert_eq!(pal.parse().unwrap(), Command::Seek(-10.0));
    }

    #[test]
    fn selection_scrolls_to_stay_visible() {
        let total = MAX_VISIBLE_SUGGESTIONS + 5;
        let mut pal = Palette::default();

        // Walk down past the bottom of the window; the window should follow.
        for _ in 0..MAX_VISIBLE_SUGGESTIONS {
            pal.move_selection(1, total);
        }
        assert_eq!(pal.selected, MAX_VISIBLE_SUGGESTIONS);
        assert_eq!(pal.scroll, 1, "window should have scrolled down by one");
        assert!(pal.selected < pal.scroll + MAX_VISIBLE_SUGGESTIONS);

        // Wrapping from the top back to the last item jumps the window to the end.
        let mut pal = Palette::default();
        pal.move_selection(-1, total);
        assert_eq!(pal.selected, total - 1);
        assert_eq!(pal.scroll, total - MAX_VISIBLE_SUGGESTIONS);

        // A short list never scrolls.
        let mut pal = Palette::default();
        pal.move_selection(1, 3);
        assert_eq!(pal.scroll, 0);
    }
}
