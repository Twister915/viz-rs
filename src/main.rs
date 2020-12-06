#![feature(div_duration)]
#![feature(trusted_len)]
#![feature(associated_type_defaults)]

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

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

fn main() {
    if let Some(target) = std::env::args().nth(1) {
        match visualize(target.as_str()) {
            Ok(()) => {}
            Err(err) => panic!("got error: {:?}", err),
        }
    } else {
        eprintln!("err: specify target file as first arg!")
    }
}
