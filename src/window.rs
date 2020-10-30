use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::log_timed;
use anyhow::Result;
use std::f64::consts::PI;

pub trait WindowingFunction {
    fn coefficient(idx: f64, count: f64) -> f64;

    fn apply(idx: f64, count: f64, value: f64) -> f64 {
        Self::coefficient(idx, count) * value
    }

    fn mapper(size: usize) -> MemoizedWindowingMapper {
        log_timed(
            format!("compute windowing function values for size {}", size),
            || MemoizedWindowingMapper {
                coefficients: {
                    let mut out = Vec::with_capacity(size);
                    for i in 0..size {
                        out.push(Self::coefficient(i as f64, size as f64));
                    }
                    out
                },
            },
        )
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
}

impl FramedMapper<Channeled<f64>, Channeled<f64>> for MemoizedWindowingMapper {
    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<f64>],
    ) -> Result<Option<&'a mut [Channeled<f64>]>> {
        input
            .iter_mut()
            .zip(self.coefficients.iter().copied())
            .for_each(move |(v, cf)| v.as_mut_ref().for_each(move |v| *v = cf * *v));

        Ok(Some(input))
    }
}
