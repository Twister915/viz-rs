use anyhow::Result;
use std::iter::FusedIterator;
use std::ops::Sub;
use std::time::{Duration, Instant};

pub type VizFloat = f64;
pub type VizComplex = fftw::types::c64;
pub type VizFftPlan = fftw::plan::R2CPlan64;

pub fn slice_copy_from<T, I>(slice: &mut [T], mut iter: I) -> &mut [T]
where
    I: Iterator<Item = T>,
{
    let n = slice.len();
    let mut idx = 0;

    loop {
        if idx >= n {
            return &mut slice[..idx];
        }

        if let Some(v) = iter.next() {
            slice[idx] = v;
        } else {
            return &mut slice[..idx];
        }

        idx += 1;
    }
}

pub fn try_use_iter<I, T, F>(source: I, mut consumer: F) -> Result<()>
where
    I: Iterator<Item = Result<T>>,
    F: for<'a> FnMut(&'a mut TryUseValueIter<I>) -> (),
{
    let mut out = TryUseValueIter { source, err: None };
    consumer(&mut out);
    if let Some(err) = out.err.take() {
        Err(err)
    } else {
        Ok(())
    }
}

pub struct TryUseValueIter<I> {
    source: I,
    err: Option<anyhow::Error>,
}

impl<I, R> Iterator for TryUseValueIter<I>
where
    I: Iterator<Item = Result<R>>,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        if self.err.is_some() {
            None
        } else {
            match self.source.next() {
                None => None,
                Some(Err(err)) => {
                    self.err = Some(err);
                    None
                }
                Some(Ok(v)) => Some(v),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.source.size_hint();
        (0, max)
    }
}

pub fn log_timed<F, R>(name: String, f: F) -> R
where
    F: FnOnce() -> R,
{
    println!("start {}", name);
    let (dur, out) = timed(f);
    println!("done {}, took {:?}", name, dur);
    out
}

pub fn timed<F, R>(f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let start_at = Instant::now();
    let result = f();
    let time_taken = Instant::now().sub(start_at);
    (time_taken, result)
}

impl<I, R> FusedIterator for TryUseValueIter<I> where I: Iterator<Item = Result<R>> + FusedIterator {}
