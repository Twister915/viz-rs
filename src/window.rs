use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::{VizFloat, log_timed};
use anyhow::Result;

pub trait WindowingFunction {
    fn coefficient(idx: VizFloat, count: VizFloat) -> VizFloat;

    fn mapper(size: usize) -> MemoizedWindowingMapper {
        let sz = size as VizFloat;
        log_timed(
            format!("compute windowing function values for size {}", size),
            || {
                let mut coefficients = (0..size)
                    .into_iter()
                    .map(move |i| i as VizFloat)
                    .map(move |i| Self::coefficient(i, sz))
                    .collect::<Vec<_>>();
                let sum = coefficients.iter().copied().sum::<VizFloat>();
                if sum.is_finite() && sum > 0.0 {
                    let scale = 2.0 / sum;
                    coefficients.iter_mut().for_each(move |cf| *cf *= scale);
                }
                MemoizedWindowingMapper { coefficients }
            },
        )
    }
}

#[derive(Copy, Clone)]
pub struct BlackmanNuttall;

impl WindowingFunction for BlackmanNuttall {
    fn coefficient(idx: VizFloat, count: VizFloat) -> VizFloat {
        const TAU: VizFloat = std::f64::consts::TAU;
        const A0: VizFloat = 0.3635819;
        const A1: VizFloat = 0.4891775;
        const A2: VizFloat = 0.1365995;
        const A3: VizFloat = 0.0106411;
        const FOURPI: VizFloat = TAU * 2.0;
        const SIXPI: VizFloat = FOURPI + TAU;

        let count_minus_one = count - 1.0;
        let a1t = A1 * VizFloat::cos((TAU * idx) / count_minus_one);
        let a2t = A2 * VizFloat::cos((FOURPI * idx) / count_minus_one);
        let a3t = A3 * VizFloat::cos((SIXPI * idx) / count_minus_one);

        A0 - a1t + a2t - a3t
    }
}

pub struct MemoizedWindowingMapper {
    coefficients: Vec<VizFloat>,
}

impl FramedMapper for MemoizedWindowingMapper {
    type Input = Channeled<VizFloat>;
    type Output = Channeled<VizFloat>;

    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<VizFloat>],
    ) -> Result<Option<&'a mut [Channeled<VizFloat>]>> {
        input
            .iter_mut()
            .zip(self.coefficients.iter())
            .for_each(move |(v, cf)| v.as_mut_ref().for_each(move |v| *v *= *cf));

        Ok(Some(input))
    }
}
