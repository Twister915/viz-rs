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
    at: usize,
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
            at: 0,
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

        let new_at = (self.at as isize) + n;
        let max_at = self.num_full_frames();
        if new_at < 0 || new_at >= (max_at as isize) {
            return Err(anyhow!("cannot seek past bounds"));
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

        let new_at = (self.at as isize) + n;
        if new_at < 0 {
            return Err(anyhow!("cannot seek to before start!"));
        }
        self.at = new_at as usize;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<&[Channeled<f64>]>> {
        if self.at == self.num_frames() {
            return Ok(None);
        }

        let cap = self.source.full_frame_size();
        let next = if let Some(next) = self.source.next_frame()? {
            next
        } else {
            return Ok(None);
        };

        if self.at > 0 {
            // move cur to prev
            let mut new_prev = if self.previous.len() < self.n_prev {
                Vec::with_capacity(cap)
            } else {
                let mut tail = self.previous.remove(self.previous.len() - 1);
                tail.clear();
                tail
            };

            new_prev.extend(self.cur.drain(..));
            self.previous.insert(0, new_prev);
        }

        use Channeled::*;
        self.cur.extend_from_slice(next);
        if self.at > 0 {
            let alpha_inv = 1.0 - self.alpha;
            for (idx, prev) in self
                .previous
                .get(0)
                .expect("we should have prev if at > 0")
                .iter()
                .enumerate()
            {
                let cur = self.cur.get_mut(idx).expect("should have same len");
                match cur {
                    Stereo(ca, cb) => {
                        if let Stereo(pa, pb) = prev {
                            *ca = (*ca * alpha_inv) + (*pa * self.alpha);
                            *cb = (*cb * alpha_inv) + (*pb * self.alpha);
                        } else {
                            return Err(anyhow!(
                                "mismatch, expected stereo got mono for prev, mixed data!"
                            ));
                        }
                    }
                    Mono(cv) => {
                        if let Mono(pv) = prev {
                            *cv = (*cv * alpha_inv) + (*pv * self.alpha);
                        } else {
                            return Err(anyhow!(
                                "mismatch, expected stereo got mono for prev, mixed data!"
                            ));
                        }
                    }
                }
            }
        }

        self.at += 1;
        Ok(Some(self.cur.as_slice()))
    }

    fn num_frames(&self) -> usize {
        self.source.num_frames()
    }

    fn num_frames_remain(&self) -> usize {
        self.num_frames() - self.at
    }

    fn num_full_frames(&self) -> usize {
        self.source.num_full_frames()
    }

    fn full_frame_size(&self) -> usize {
        self.source.full_frame_size()
    }
}

delegate_impls!(ExponentialSmoothing<S>, S, source);
