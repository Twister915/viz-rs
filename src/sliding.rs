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
    S: Samples<T>,
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

impl<S, T> Framed<T> for SlidingFrame<S, T>
where
    S: Samples<T>,
    T: Copy,
{
    fn seek_frame(&mut self, n: isize) -> Result<()> {
        if n < 0 {
            let buf_len = self.buf.len();
            let to_remove = std::cmp::min(buf_len, -n as usize);
            let first_remove = buf_len - to_remove;
            self.buf.drain(first_remove..buf_len);
        } else if n > 0 {
            let buf_len = self.buf.len();
            let to_remove = std::cmp::min(buf_len, n as usize);
            self.buf.drain(0..to_remove);
        }

        self.source.seek_samples(n)?;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<&mut [T]>> {
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

    fn num_frames(&self) -> usize {
        self.source.num_samples() / self.stride
    }

    fn num_frames_remain(&self) -> usize {
        self.source.num_samples_remain() / self.stride
    }

    fn num_full_frames(&self) -> usize {
        let samples = self.source.num_samples();
        let non_full_samples = self.size - 1;
        if non_full_samples > samples {
            0
        } else {
            (samples - non_full_samples) / self.stride
        }
    }

    fn full_frame_size(&self) -> usize {
        self.size
    }
}

delegate_impls!(SlidingFrame<S, T>, S, source);

impl<S, T> SlidingFrame<S, T>
where
    S: Samples<T>,
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
            move |iter| {
                buf.extend(
                    iter.take_while(move |v| v.is_some())
                        .map(move |v| v.unwrap()),
                )
            },
        )
    }
}
