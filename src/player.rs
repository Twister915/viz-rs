use crate::channeled::Channeled;
use crate::framed::Samples;
use crate::util::VizFloat;
use crate::wav::WavFile;
use anyhow::{Result, bail};
use sdl2::AudioSubsystem;
use sdl2::audio::{AudioCallback, AudioDevice, AudioSpecDesired};
use std::ops::{Mul, Sub};
use std::time::{Duration, Instant};

enum WavStates {
    Empty,
    Ready(WavPlayerInner),
    Playing(AudioDevice<WavCallback>),
}

impl WavStates {
    fn take(&mut self) -> Self {
        let mut next = Self::Empty;
        std::mem::swap(&mut next, self);
        next
    }
}

pub struct WavPlayer {
    state: WavStates,
    sdl_audio: AudioSubsystem,
}

impl WavPlayer {
    pub fn new(sdl_audio: AudioSubsystem, wav: WavFile) -> WavPlayer {
        WavPlayer {
            state: WavStates::Ready(WavPlayerInner {
                source: wav,
                start_playing_at: None,
                at: Duration::from_nanos(0),
                file_at: Duration::from_nanos(0),
            }),
            sdl_audio,
        }
    }

    pub fn play(&mut self) -> Result<()> {
        match self.state.take() {
            WavStates::Empty => bail!("player state was unexpectedly empty"),
            WavStates::Playing(playing) => {
                self.state = WavStates::Playing(playing);
            }
            WavStates::Ready(mut ready) => {
                ready.start_playing_at = Some(Instant::now());
                let dev = self
                    .sdl_audio
                    .open_playback(
                        None,
                        &AudioSpecDesired {
                            freq: Some(ready.source.sample_rate as i32),
                            channels: Some(ready.source.num_channels as u8),
                            samples: None,
                        },
                        move |_| WavCallback { inner: ready },
                    )
                    .map_err(map_sdl_err)?;
                dev.resume();
                self.state = WavStates::Playing(dev);
            }
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        match self.state.take() {
            WavStates::Empty => bail!("player state was unexpectedly empty"),
            WavStates::Ready(ready) => {
                self.state = WavStates::Ready(ready);
            }
            WavStates::Playing(playing) => {
                playing.pause();
                let mut inner = playing.close_and_get_callback().inner;
                if let Some(start_playing_at) = inner.start_playing_at.take() {
                    inner.at += Instant::now().sub(start_playing_at);
                }
                self.state = WavStates::Ready(inner);
            }
        }

        Ok(())
    }

    /// Seek by a signed number of seconds (negative seeks backward), used by the command
    /// palette and the arrow-key jumps.
    pub fn seek_secs(&mut self, secs: f64) -> Result<()> {
        let was_playing = matches!(self.state, WavStates::Playing(_));
        self.stop()?;
        if let WavStates::Ready(player) = &mut self.state {
            let sample_rate = player.source.sample_rate as f64;
            let skip_samples = (secs * sample_rate).round() as isize;
            let skip_time = Duration::from_secs_f64(secs.abs());
            player.source.seek_samples(skip_samples)?;
            if secs >= 0.0 {
                player.at += skip_time;
                player.file_at += skip_time;
            } else {
                player.at = player.at.saturating_sub(skip_time);
                player.file_at = player.file_at.saturating_sub(skip_time);
            }
        } else {
            bail!("player state was not ready after stop")
        }

        if was_playing {
            self.play()?;
        }
        Ok(())
    }
}

struct WavPlayerInner {
    source: WavFile,
    start_playing_at: Option<Instant>,
    at: Duration,
    file_at: Duration,
}

struct WavCallback {
    inner: WavPlayerInner,
}

impl AudioCallback for WavCallback {
    type Channel = f32;

    fn callback(&mut self, data: &mut [Self::Channel]) {
        let mut idx = 0;
        let mut sample_frames = 0u32;
        while idx < data.len() {
            let sample = match self.inner.source.next_sample() {
                Ok(Some(sample)) => sample,
                Ok(None) | Err(_) => break,
            };

            match sample {
                Channeled::Mono(v) => {
                    let v: VizFloat = v.into();
                    let v = v as f32;
                    data[idx] = v;
                    idx += 1;
                }
                Channeled::Stereo(l, r) => {
                    if idx + 1 >= data.len() {
                        break;
                    }
                    let l: VizFloat = l.into();
                    let r: VizFloat = r.into();
                    let l = l as f32;
                    let r = r as f32;
                    data[idx] = l;
                    idx += 1;
                    data[idx] = r;
                    idx += 1;
                }
            }

            sample_frames += 1;
        }

        if sample_frames != 0 {
            self.inner.file_at +=
                Duration::from_nanos(1_000_000_000 / (self.inner.source.sample_rate as u64))
                    .mul(sample_frames);
        }

        data[idx..].iter_mut().for_each(move |v| *v = 0.0);
    }
}

fn map_sdl_err(err: String) -> anyhow::Error {
    anyhow::anyhow!("sdl2: {}", err)
}
