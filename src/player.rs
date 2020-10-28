use sdl2::audio::{AudioDevice, AudioSpecDesired, AudioCallback};
use sdl2::AudioSubsystem;
use anyhow::Result;
use crate::wav::WavFile;
use std::time::{Instant, Duration};
use std::ops::{Sub, Add, Mul};
use crate::framed::{Sampled, Samples};
use crate::channeled::Channeled;

enum WavStates {
    Empty,
    Ready(WavPlayerInner),
    Playing(AudioDevice<WavCallback>)
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
        let mut next_state = WavStates::Empty;
        std::mem::swap(&mut self.state, &mut next_state);
        match next_state {
            WavStates::Empty => {
                panic!("empty when can't be empty")
            },
            WavStates::Playing(playing) => {
                self.state = WavStates::Playing(playing);
            },
            WavStates::Ready(mut ready) => {
                ready.start_playing_at = Some(Instant::now());
                let dev = self.sdl_audio.open_playback(None, &AudioSpecDesired {
                    freq: Some(ready.source.sample_rate as i32),
                    channels: Some(ready.source.num_channels as u8),
                    samples: None,
                }, move |_| WavCallback { inner: ready }).map_err(map_sdl_err)?;
                dev.resume();
                self.state = WavStates::Playing(dev);
            }
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        let mut next_state = WavStates::Empty;
        std::mem::swap(&mut self.state, &mut next_state);
        match next_state {
            WavStates::Empty => {
                panic!("empty when can't be empty")
            },
            WavStates::Ready(ready) => {
                self.state = WavStates::Ready(ready);
            },
            WavStates::Playing(playing) => {
                playing.pause();
                let mut inner = playing.close_and_get_callback().inner;
                inner.at += Instant::now().sub(inner.start_playing_at.take().expect("should exist"));
                self.state = WavStates::Ready(inner);
            }
        }

        Ok(())
    }

    pub fn seek(&mut self, amount: Duration) -> Result<()> {
        let seek_to = Instant::now().add(amount);
        self.stop()?;
        if let WavStates::Ready(player) = &mut self.state {
            let amount = seek_to.sub(Instant::now());
            let skip_samples = player.source.samples_from_dur(amount);
            let skip_time = Duration::from_nanos(1_000_000_000 / (player.source.sample_rate as u64)).mul(skip_samples as u32);
            player.source.seek_samples(skip_samples as isize)?;
            player.at += skip_time;
            player.file_at += skip_time;
        } else {
            panic!("state malfunction, stopped but not in ready state")
        }

        self.play()
    }
}

struct WavPlayerInner {
    source: WavFile,
    start_playing_at: Option<Instant>,
    at: Duration,
    file_at: Duration,
}

struct WavCallback {
    inner: WavPlayerInner
}

impl AudioCallback for WavCallback {
    type Channel = f32;

    fn callback(&mut self, data: &mut [Self::Channel]) {
        let mut idx = 0;
        while let Some(sample) = self.inner.source.next_sample().expect("no err") {
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
                self.inner.file_at += Duration::from_nanos(1_000_000_000 / (self.inner.source.sample_rate as u64)).mul(idx as u32);
                return
            }
        }
    }
}

fn map_sdl_err(err: String) -> anyhow::Error {
    anyhow::anyhow!("sdl2: {}", err)
}