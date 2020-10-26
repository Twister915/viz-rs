use crate::fraction::Fraction;
use crate::framed::{ChanneledMapperWrapper, FramedMapper, MapperToChanneled};
use anyhow::Result;

// thanks to: https://github.com/arntanguy/gram_savitzky_golay/tree/master/src
// thanks to: https://github.com/mirkov/savitzky-golay/blob/master/gram-poly.lisp
// thanks to: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/48493/versions/1/previews/Functions/grampoly.m/index.html
fn gram_poly(i: Fraction, m: Fraction, k: Fraction, s: Fraction) -> Fraction {
    if k > 0 {
        let r0 = gram_poly(i, m, k - 1, s);
        let r1 = gram_poly(i, m, k - 1, s - 1);
        let r2 = gram_poly(i, m, k - 2, s);
        // jesus forgive me
        ((((k * 4) - 2) / (k * (m * 2 - k + 1))) * (i * r0 + s * r1))
            - ((k - 1) * (m * 2 + k)) / (k * (m * 2 - k + 1)) * r2
    } else {
        if k == 0 && s == 0 {
            1.into()
        } else {
            0.into()
        }
    }
}

fn gen_fact(a: Fraction, b: Fraction) -> Fraction {
    let mut gf = 1.into();
    let mut j = (a - b) + 1;
    while j <= a {
        gf *= j;
        j += 1;
    }

    gf
}

fn weight(i: Fraction, t: Fraction, m: Fraction, n: Fraction, s: Fraction) -> Fraction {
    let mut w = 0.into();
    let mut k = 0.into();
    while k <= n {
        let fact0 = gen_fact(m * 2, k);
        let fact1 = gen_fact(m * 2 + k + 1, k + 1);
        let p0 = gram_poly(i, m, k, 0.into());
        let p1 = gram_poly(t, m, k, s);
        w += (k * 2 + 1) * (fact0 / fact1) * p0 * p1;
        k += 1;
    }

    w
}

fn weights(m: i64, t: Fraction, n: Fraction, s: Fraction) -> Vec<Fraction> {
    let count = (2 * m) + 1;
    let mut fracs = Vec::with_capacity(count as usize);
    for i in 0..count {
        let weight = weight(((i - m) as i64).into(), t, (m as i64).into(), n, s);
        fracs.push(weight);
    }

    let sum = fracs
        .iter()
        .copied()
        .fold(0.into(), move |sm: Fraction, elem| sm + elem);

    fracs.into_iter().map(move |f| f / sum).collect()
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash)]
pub struct SavitzkyGolayConfig {
    pub window_size: u64,
    pub polynomial_order: u64,
}

impl SavitzkyGolayConfig {
    pub fn compute_coefficients(&self) -> SavitzkyGolayCoefficients {
        if self.window_size % 2 == 0 || self.window_size < 3 {
            panic!("invalid window size {}", self.window_size)
        }

        let half_window_size = (self.window_size as i64) / 2;
        let start_time_delta = -half_window_size;
        let end_time_delta = half_window_size;

        let mut out = Vec::with_capacity(self.window_size as usize);
        for delta in start_time_delta..=end_time_delta {
            let weights = weights(
                half_window_size,
                delta.into(),
                self.polynomial_order.into(),
                0.into(),
            );
            out.push(TimeStepCoefficients {
                time_offset: delta,
                coefficients: weights,
            })
        }

        SavitzkyGolayCoefficients { coefficients: out }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct SavitzkyGolayCoefficients {
    pub coefficients: Vec<TimeStepCoefficients>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct TimeStepCoefficients {
    pub time_offset: i64,
    pub coefficients: Vec<Fraction>,
}

#[derive(Debug)]
pub struct SavitzkyGolayMapper {
    buf: Vec<f64>,
    cap: usize,
    coefficients: SavitzkyGolayCoefficients,
}

impl SavitzkyGolayMapper {
    pub fn new(size: usize, config: SavitzkyGolayConfig) -> SavitzkyGolayMapper {
        Self {
            buf: Vec::with_capacity(size),
            cap: size,
            coefficients: config.compute_coefficients(),
        }
    }
}

impl FramedMapper<f64, f64> for SavitzkyGolayMapper {
    fn map(&mut self, input: &[f64]) -> Result<Option<&[f64]>> {
        self.buf.clear();
        let n = input.len() as isize;
        let coefficients = self.coefficients.coefficients.as_slice();
        let half_window_size = (coefficients.len() / 2) as isize;
        let window_size = (half_window_size * 2) + 1;

        for idx in 0..n {
            let (coefficients_idx, start, stop) = {
                let def_coefficients_idx = half_window_size;
                let neg_offset = idx - half_window_size;
                if neg_offset < 0 {
                    (def_coefficients_idx + neg_offset, 0, window_size)
                } else {
                    let until_end = (n - idx) - 1;
                    if until_end < half_window_size {
                        (
                            def_coefficients_idx + (half_window_size - until_end),
                            n - window_size,
                            n,
                        )
                    } else {
                        (
                            def_coefficients_idx,
                            idx - half_window_size,
                            idx + half_window_size,
                        )
                    }
                }
            };

            let coefficients_idx = coefficients_idx as usize;
            let start = start as usize;
            let stop = stop as usize;
            let mut ci = 0;
            let mut sum = 0.0;
            for i in start..stop {
                let v = input[i];
                let cf = coefficients[coefficients_idx].coefficients[ci];
                sum += cf * v;
                ci += 1;
            }
            self.buf.push(sum);
        }

        Ok(Some(&self.buf[..]))
    }
}

impl MapperToChanneled<f64, f64> for SavitzkyGolayMapper {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let cap = self.cap;
        self.channeled(cap)
    }
}
