use crate::channeled::Channeled;
use crate::delegate_impls;
use crate::framed::Framed;
use anyhow::{anyhow, Result};

pub struct ExponentialSmoothing<S> {
    source: S,
    cur: Vec<Channeled<f64>>,
    previous: Vec<Vec<Channeled<f64>>>,
    n_prev: usize,
    alpha: f64,
}

impl<S> ExponentialSmoothing<S>
where
    S: Framed<Channeled<f64>>,
{
    pub fn new(source: S, seek_back_limit: usize, alpha: f64) -> Self {
        let cap = source.full_frame_size();
        Self {
            source,
            cur: Vec::with_capacity(cap),
            previous: Vec::with_capacity(seek_back_limit),
            n_prev: seek_back_limit,
            alpha,
        }
    }
}

impl<S> Framed<Channeled<f64>> for ExponentialSmoothing<S>
where
    S: Framed<Channeled<f64>>,
{
    fn seek_frame(&mut self, n: isize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        if n < 0 {
            let max_prev = self.previous.len() - 1;
            if max_prev < (-n as usize) {
                return Err(anyhow!("cannot seek further than {} frames back", max_prev));
            }

            // if we want to seek 1 back, we drop 0 from prev and replace cur with remove of prev
            if n < 1 {
                self.previous.drain(1..(-n as usize));
            }

            self.cur = self.previous.remove(0);
            self.source.seek_frame(n)?;
        } else {
            for _ in 0..n {
                self.next_frame()?;
            }
        }

        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<&[Channeled<f64>]>> {
        let cap = self.source.full_frame_size();
        let next = if let Some(next) = self.source.next_frame()? {
            next
        } else {
            return Ok(None);
        };

        if !self.cur.is_empty() {
            // move cur to prev
            let mut new_prev = if self.previous.len() < self.n_prev {
                Vec::with_capacity(cap)
            } else {
                let mut tail = self.previous.remove(self.previous.len() - 1);
                tail.clear();
                tail
            };

            new_prev.extend_from_slice(self.cur.as_slice());
            self.cur.clear();
            self.previous.insert(0, new_prev);
        }

        if let Some(prev) = self.previous.get(0) {
            let alpha = self.alpha;
            let alpha_inv = 1.0 - alpha;

            self.cur.extend(
                next.iter()
                    .zip(prev.iter().copied())
                    .map(move |(new, pre)| new.zip(pre).expect("mono/stereo should match"))
                    .map(move |zipped| zipped.map(move |(n, p)| (n * alpha_inv) + (p * alpha))),
            );
        } else {
            self.cur.extend_from_slice(next);
        }

        Ok(Some(self.cur.as_slice()))
    }

    fn num_frames(&self) -> usize {
        self.source.num_frames()
    }

    fn num_frames_remain(&self) -> usize {
        self.source.num_frames_remain()
    }

    fn num_full_frames(&self) -> usize {
        self.source.num_full_frames()
    }

    fn full_frame_size(&self) -> usize {
        self.source.full_frame_size()
    }
}

delegate_impls!(ExponentialSmoothing<S>, S, source);
