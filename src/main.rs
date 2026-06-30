use crate::viz::visualize;

mod binner;
mod channeled;
mod command;
mod engine;
mod exponential_smoothing;
mod fft;
mod font;
mod framed;
mod pipeline;
mod player;
mod savitzky_golay;
mod sliding;
mod timer;
mod util;
mod viz;
mod wav;
mod window;

fn main() -> anyhow::Result<()> {
    // A file argument is optional now: with no song, the visualizer starts idle and a song can be
    // loaded at runtime from the command palette (press `/`, then `load <path.wav>`).
    let target = std::env::args().nth(1);
    visualize(target.as_deref())?;

    Ok(())
}
