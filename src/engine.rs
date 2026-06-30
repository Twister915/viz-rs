//! The live visual engine: it owns the (rebuildable) math pipeline and the current playback
//! position, so the visualizer loop can change *any* baked-in parameter — bin count, gamma, the
//! Savitzky-Golay matrices, alphas, dB range — by reconstructing the pipeline and re-seeking to
//! the exact same spot. It also supports an *unloaded* state so a song can be loaded at runtime
//! from the command palette.

use crate::channeled::Channeled;
use crate::framed::Framed;
use crate::pipeline::{VizPipelineConfig, create_debug_fft_pipeline, create_viz_pipeline};
use crate::util::VizFloat;
use crate::wav::WavFile;
use anyhow::Result;

const BUF_SIZE: usize = 32768;

type VizFrames = Box<dyn Framed<Item = Channeled<VizFloat>>>;
type DebugFrames = Box<dyn Framed<Item = VizFloat>>;

/// State that only exists while a song is loaded.
struct Loaded {
    file: String,
    frames: VizFrames,
    debug_frames: DebugFrames,
    /// Index of the frame that the next `advance()` will return (absolute position).
    next_index: usize,
    ended: bool,
}

pub struct VizEngine {
    /// The live source of truth for all parameters. Mutate this then call
    /// [`VizEngine::reload_and_refresh`] for any change where [`needs_rebuild`] is true.
    pub config: VizPipelineConfig,
    loaded: Option<Loaded>,
    /// The most recent bar frame (kept across pause/unload so it can decay naturally).
    pub last_frame: Vec<Channeled<VizFloat>>,
    /// The most recent debug-FFT frame.
    pub last_debug_frame: Vec<VizFloat>,
}

impl VizEngine {
    /// An engine with no song loaded yet.
    pub fn empty(config: VizPipelineConfig) -> Self {
        VizEngine {
            config,
            loaded: None,
            last_frame: Vec::new(),
            last_debug_frame: Vec::new(),
        }
    }

    /// An engine with `file` already loaded and playing from the start.
    pub fn with_song(config: VizPipelineConfig, file: &str) -> Result<Self> {
        let mut engine = Self::empty(config);
        engine.load(file)?;
        Ok(engine)
    }

    fn build(file: &str, config: VizPipelineConfig) -> Result<(VizFrames, DebugFrames)> {
        let frames: VizFrames =
            Box::new(create_viz_pipeline(WavFile::open(file, BUF_SIZE)?, config)?);
        let debug_frames: DebugFrames =
            Box::new(create_debug_fft_pipeline(WavFile::open(file, BUF_SIZE)?, config)?);
        Ok((frames, debug_frames))
    }

    /// Load a new song from the start.
    pub fn load(&mut self, file: &str) -> Result<()> {
        let (frames, debug_frames) = Self::build(file, self.config)?;
        self.loaded = Some(Loaded {
            file: file.to_string(),
            frames,
            debug_frames,
            next_index: 0,
            ended: false,
        });
        Ok(())
    }

    /// Drop the current song. The last frame is kept so the bars can fall away naturally.
    pub fn unload(&mut self) {
        self.loaded = None;
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    pub fn is_ended(&self) -> bool {
        self.loaded.as_ref().map(|l| l.ended).unwrap_or(false)
    }

    pub fn loaded_file(&self) -> Option<&str> {
        self.loaded.as_ref().map(|l| l.file.as_str())
    }

    /// Index of the next frame to be produced (the current playback position, in frames).
    pub fn frame_index(&self) -> usize {
        self.loaded.as_ref().map(|l| l.next_index).unwrap_or(0)
    }

    /// Current position in seconds (frames / fps).
    pub fn position_secs(&self) -> f64 {
        self.frame_index() as f64 / self.config.fps.max(1) as f64
    }

    /// Pull the next frame, copying it into `last_frame` / `last_debug_frame`. Returns `false`
    /// when there is no song or the stream has ended.
    pub fn advance(&mut self) -> Result<bool> {
        let Some(l) = self.loaded.as_mut() else {
            return Ok(false);
        };

        // The two pipelines are independent fields, so these borrows don't overlap.
        let next = l.frames.next_frame()?;
        let Some(frame) = next else {
            l.ended = true;
            return Ok(false);
        };
        self.last_frame.clear();
        self.last_frame.extend_from_slice(frame);

        match l.debug_frames.next_frame()? {
            Some(debug_frame) => {
                self.last_debug_frame.clear();
                self.last_debug_frame.extend_from_slice(debug_frame);
            }
            None => {
                l.ended = true;
                return Ok(false);
            }
        }

        l.next_index += 1;
        Ok(true)
    }

    /// Seek to an absolute frame index, moving both pipelines together.
    pub fn seek_to(&mut self, target: usize) -> Result<()> {
        if let Some(l) = self.loaded.as_mut() {
            let delta = target as isize - l.next_index as isize;
            l.frames.seek_frame(delta)?;
            l.debug_frames.seek_frame(delta)?;
            l.next_index = target;
            l.ended = false;
        }
        Ok(())
    }

    /// Seek by a relative number of frames (clamped at the start).
    pub fn seek_frames(&mut self, delta: isize) -> Result<()> {
        let target = (self.frame_index() as isize + delta).max(0) as usize;
        self.seek_to(target)
    }

    /// Rescale the current frame position when `fps` changes, so the *time* position is kept.
    pub fn rescale_position(&mut self, old_fps: u64, new_fps: u64) {
        if let Some(l) = self.loaded.as_mut() {
            if old_fps != 0 {
                l.next_index =
                    ((l.next_index as u64 * new_fps + old_fps / 2) / old_fps) as usize;
            }
        }
    }

    /// Rebuild both pipelines from the current `config` and re-seek to the current position,
    /// re-pulling the displayed frame so it immediately reflects the new parameters. Net
    /// position is preserved (no drift on repeated tweaks).
    pub fn reload_and_refresh(&mut self) -> Result<()> {
        if let Some(l) = self.loaded.as_mut() {
            let file = l.file.clone();
            // The frame currently on screen is one behind the next-to-produce index.
            let display = l.next_index.saturating_sub(1);
            let (frames, debug_frames) = Self::build(&file, self.config)?;
            l.frames = frames;
            l.debug_frames = debug_frames;
            l.frames.seek_frame(display as isize)?;
            l.debug_frames.seek_frame(display as isize)?;
            l.next_index = display;
            l.ended = false;
        } else {
            return Ok(());
        }
        // Re-pull the displayed frame with the new parameters (advance leaves next_index where
        // it started, so position is unchanged).
        self.advance()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::embedded_default_config;
    use std::io::Write;

    /// Write a tiny valid 16-bit mono PCM WAV (a 440 Hz tone) to a temp file so the engine tests
    /// don't depend on any checked-in audio.
    fn write_test_wav() -> std::path::PathBuf {
        let sample_rate: u32 = 44100;
        let n: u32 = 20000;
        let bits: u16 = 16;
        let channels: u16 = 1;
        let block_align: u16 = channels * bits / 8;
        let byte_rate: u32 = sample_rate * block_align as u32;
        let data_len: u32 = n * block_align as u32;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..n {
            let t = i as f64 / sample_rate as f64;
            let v = (0.3 * 32767.0 * (std::f64::consts::TAU * 440.0 * t).sin()) as i16;
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let path = std::env::temp_dir().join("vis_rs_engine_rebuild_test.wav");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        path
    }

    #[test]
    fn rebuilding_pipeline_preserves_position_and_applies_baked_params() {
        let path = write_test_wav();
        let path = path.to_str().unwrap();
        let mut engine = VizEngine::with_song(embedded_default_config(), path).unwrap();

        for _ in 0..5 {
            engine.advance().unwrap();
        }
        let pos = engine.frame_index();
        assert_eq!(pos, 5);
        assert!(!engine.last_frame.is_empty());

        // Change a *baked* parameter (bar count) and reconstruct the pipeline in place.
        let new_bins = 16;
        engine.config.binning.bins = new_bins;
        engine.reload_and_refresh().unwrap();
        assert_eq!(
            engine.frame_index(),
            pos,
            "position must be preserved across a rebuild"
        );
        assert_eq!(
            engine.last_frame.len(),
            new_bins,
            "rebuilt pipeline must use the new bin count"
        );

        // Changing the temporal smoothing (also baked) must not panic or move position.
        engine.config.alpha0 = 0.5;
        engine.reload_and_refresh().unwrap();
        assert_eq!(engine.frame_index(), pos);

        // Unloading clears the song but keeps the last frame so it can decay away.
        engine.unload();
        assert!(!engine.is_loaded());
        assert!(!engine.last_frame.is_empty());
    }

    #[test]
    fn seek_moves_to_absolute_frame() {
        let path = write_test_wav();
        let mut engine =
            VizEngine::with_song(embedded_default_config(), path.to_str().unwrap()).unwrap();
        engine.advance().unwrap();
        engine.seek_to(10).unwrap();
        assert_eq!(engine.frame_index(), 10);
        engine.seek_frames(-3).unwrap();
        assert_eq!(engine.frame_index(), 7);
    }
}
