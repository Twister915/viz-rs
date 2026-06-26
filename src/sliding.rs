use crate::delegate_impls;
use crate::framed::{Framed, Samples};
use crate::util::try_use_iter;
use anyhow::Result;

pub struct SlidingFrame<S, T> {
    source: S,
    buf: Vec<T>,
    cur_buf: Vec<T>,
    size: usize,
    stride: usize,
}

impl<S, T> SlidingFrame<S, T>
where
    S: Samples<Item = T>,
{
    pub fn new(source: S, size: usize, mut stride: usize) -> Self {
        if stride == 0 {
            stride = 1;
        }
        Self {
            source,
            buf: Vec::with_capacity(size),
            cur_buf: Vec::with_capacity(size),
            size,
            stride,
        }
    }
}

impl<S, T> Framed for SlidingFrame<S, T>
where
    S: Samples<Item = T>,
    T: Copy,
{
    type Item = T;

    fn seek_frame(&mut self, n: isize) -> Result<()> {
        let sample_delta = n.saturating_mul(self.stride as isize);
        if sample_delta < 0 {
            let buf_len = self.buf.len() as isize;
            self.buf.clear();
            self.source.seek_samples(sample_delta - buf_len)?;
        } else if sample_delta > 0 {
            let buf_len = self.buf.len();
            let to_remove = std::cmp::min(buf_len, sample_delta as usize);
            self.buf.drain(0..to_remove);
            if to_remove < sample_delta as usize {
                self.source
                    .seek_samples((sample_delta as usize - to_remove) as isize)?;
            }
        }
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<&mut [Self::Item]>> {
        if !self.buf.is_empty() {
            if self.buf.len() < self.stride {
                self.buf.clear();
                return Ok(None);
            }

            self.buf.drain(0..self.stride);
        }

        self.ensure_buf_filled()?;

        if self.buf.is_empty() {
            return Ok(None);
        }

        self.cur_buf.clear();
        self.cur_buf.extend_from_slice(self.buf.as_slice());
        Ok(Some(self.cur_buf.as_mut_slice()))
    }

    fn full_frame_size(&self) -> usize {
        self.size
    }
}

delegate_impls!(SlidingFrame<S, T>, S, source);

impl<S, T> SlidingFrame<S, T>
where
    S: Samples<Item = T>,
    T: Copy,
{
    fn ensure_buf_filled(&mut self) -> Result<()> {
        let source = &mut self.source;
        let buf = &mut self.buf;
        let n_load = std::cmp::min(source.num_samples_remain(), self.size - buf.len());
        if n_load == 0 {
            return Ok(());
        }

        try_use_iter(
            std::iter::repeat_with(|| source.next_sample()).take(n_load),
            move |iter| buf.extend(iter.take_while(move |v| v.is_some()).flatten()),
        )
    }
}
