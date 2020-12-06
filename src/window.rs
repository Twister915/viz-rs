use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::{log_timed, VizFloat};
use anyhow::Result;
use itertools::Itertools;

pub trait WindowingFunction {
    fn coefficient(idx: VizFloat, count: VizFloat) -> VizFloat;

    fn apply(idx: VizFloat, count: VizFloat, value: VizFloat) -> VizFloat {
        Self::coefficient(idx, count) * value
    }

    fn mapper(size: usize) -> MemoizedWindowingMapper {
        let sz = size as VizFloat;
        log_timed(
            format!("compute windowing function values for size {}", size),
            || MemoizedWindowingMapper {
                coefficients: (0..size)
                    .into_iter()
                    .map(move |i| i as VizFloat)
                    .map(move |i| Self::coefficient(i, sz))
                    .collect_vec(),
            },
        )
    }
}

#[derive(Copy, Clone)]
pub struct BlackmanNuttall;

impl WindowingFunction for BlackmanNuttall {
    fn coefficient(idx: VizFloat, count: VizFloat) -> VizFloat {
        const TAU: VizFloat = 6.28318530717958647692528676655900577;
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

impl FramedMapper<Channeled<VizFloat>, Channeled<VizFloat>> for MemoizedWindowingMapper {
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
