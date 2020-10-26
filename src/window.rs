use crate::framed::{ChanneledMapperWrapper, FramedMapper, MapperToChanneled};
use anyhow::Result;
use std::f64::consts::PI;
use std::marker::PhantomData;

pub trait WindowingFunction {
    fn coefficient(idx: f64, count: f64) -> f64;

    fn apply(idx: f64, count: f64, value: f64) -> f64 {
        Self::coefficient(idx, count) * value
    }

    fn framed_mapper(size: usize) -> WindowingMapper<Self>
    where
        Self: Sized,
    {
        WindowingMapper {
            buf: Vec::with_capacity(size),
            size,
            _windowing_func: PhantomData,
        }
    }

    fn memoized_mapper(size: usize) -> MemoizedWindowingMapper {
        let mut coefficients = Vec::with_capacity(size);
        for i in 0..size {
            coefficients.push(Self::coefficient(i as f64, size as f64));
        }
        MemoizedWindowingMapper {
            coefficients,
            buf: Vec::with_capacity(size),
        }
    }
}

#[derive(Copy, Clone)]
pub struct BlackmanNuttall;

impl WindowingFunction for BlackmanNuttall {
    fn coefficient(idx: f64, count: f64) -> f64 {
        const A0: f64 = 0.3635819;
        const A1: f64 = 0.4891775;
        const A2: f64 = 0.1365995;
        const A3: f64 = 0.0106411;
        const TWOPI: f64 = PI * 2.0;
        const FOURPI: f64 = PI * 4.0;
        const SIXPI: f64 = PI * 6.0;

        let count_minus_one = count - 1.0;
        let a1t = A1 * f64::cos((TWOPI * idx) / count_minus_one);
        let a2t = A2 * f64::cos((FOURPI * idx) / count_minus_one);
        let a3t = A3 * f64::cos((SIXPI * idx) / count_minus_one);

        A0 - a1t + a2t - a3t
    }
}

pub struct WindowingMapper<W> {
    buf: Vec<f64>,
    size: usize,
    _windowing_func: PhantomData<W>,
}

impl<W> FramedMapper<f64, f64> for WindowingMapper<W>
where
    W: WindowingFunction,
{
    fn map(&mut self, input: &[f64]) -> Result<Option<&[f64]>> {
        self.buf.clear();
        let count = self.size as f64;
        let idx_offset = count - (input.len() as f64);
        for (idx, datum) in input.iter().enumerate() {
            let idx = (idx as f64) + idx_offset;
            self.buf.push(W::apply(idx, count, *datum));
        }

        Ok(Some(self.buf.as_slice()))
    }
}

impl<W> MapperToChanneled<f64, f64> for WindowingMapper<W>
where
    W: WindowingFunction,
{
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let size = self.size;
        self.channeled(size)
    }
}

pub struct MemoizedWindowingMapper {
    coefficients: Vec<f64>,
    buf: Vec<f64>,
}

impl FramedMapper<f64, f64> for MemoizedWindowingMapper {
    fn map(&mut self, input: &[f64]) -> Result<Option<&[f64]>> {
        self.buf.clear();
        for (idx, v) in input.iter().enumerate() {
            self.buf.push(self.coefficients[idx] * *v);
        }
        Ok(Some(self.buf.as_slice()))
    }
}

impl MapperToChanneled<f64, f64> for MemoizedWindowingMapper {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let size = self.coefficients.len();
        self.channeled(size)
    }
}
