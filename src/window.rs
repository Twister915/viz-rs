use crate::framed::{ChanneledMapperWrapper, FramedMapper, MapperToChanneled};
use anyhow::Result;
use std::f64::consts::PI;

pub trait WindowingFunction {
    fn coefficient(idx: f64, count: f64) -> f64;

    fn apply(idx: f64, count: f64, value: f64) -> f64 {
        Self::coefficient(idx, count) * value
    }

    fn mapper(size: usize) -> MemoizedWindowingMapper {
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

pub struct MemoizedWindowingMapper {
    coefficients: Vec<f64>,
    buf: Vec<f64>,
}

impl FramedMapper<f64, f64> for MemoizedWindowingMapper {
    fn map(&mut self, input: &[f64]) -> Result<Option<&[f64]>> {
        self.buf.clear();

        self.buf.extend(
            input
                .iter()
                .copied()
                .zip(self.coefficients.iter().copied())
                .map(move |(v, cf)| cf * v),
        );

        Ok(Some(self.buf.as_slice()))
    }
}

impl MapperToChanneled<f64, f64> for MemoizedWindowingMapper {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let size = self.coefficients.len();
        self.channeled(size)
    }
}
