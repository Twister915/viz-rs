#![feature(div_duration)]
#![feature(trusted_len)]

use crate::viz::visualize;

mod binner;
mod channeled;
mod db;
mod exponential_smoothing;
mod fft;
mod fraction;
mod framed;
mod player;
mod savitzky_golay;
mod sliding;
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
