use crate::viz::visualize;

mod binner;
mod channeled;
mod exponential_smoothing;
mod fft;
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
    if let Some(target) = std::env::args().nth(1) {
        visualize(target.as_str())?;
    } else {
        eprintln!("err: specify target file as first arg!")
    }

    Ok(())
}
