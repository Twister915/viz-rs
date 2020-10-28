/// # Savitzky Golay Smoothing
///
/// This smoothing algorthim is applied to a single frame of data, and will try to fit the curve of
/// the data to a polynomial.
///
/// The algorithm has the following dials to turn:
/// * The degree of the polynomial
/// * The "window size," which is the number of data-points which surround any given point that the
///   algorithm considers.
/// * The order of the polynomial. This should almost always be 0, but in cases where it is not, then
///   the result of smoothing some data is to compute a smoothed Nth derivative of the data with respect
///   to the X axis.
///
/// The math assumes an evenly spaced input, and produces an evenly spaced output.
///
/// ## Usage
///
/// You can create an instance of the `SavitzkyGolayConfig` with the three values explained above.
/// There is no recommendation for the window size, except that it must be an odd number greater than 1.
/// The order should often be set to 0. The degree is best when it is set to 2, 3, or 4.
///
/// You can take a `SavitzkyGolyConfig` and get a `SavitzkyGolayMapper`, which can be used with the
/// `Framed` interface to map each frame of data in your pipeline, by calling `.into_mapper()` on your
/// config.
///
/// ## Math During Runtime
///
/// Given some config, a set of coefficients are computed. These coefficients are a N by N matrix where
/// N is the window size.
///
/// Given the following definitions:
/// * `N = window size`
/// * `M = (N - 1) / 2`
/// * `K = length of input and output`
///
/// To smooth input data frames, we convolve the input data and the coefficients. Literally, this means we:
/// * Iterate through the input data (index will be referred to as `i`)
/// * Select one of the rows from the coefficients matrix (defined later)
/// * Multiply & sum pointwise the slices `input[(i - M)..(i + M)]` and `coefficients[0..N]`.
/// * Replace the point at i with the sum you just computed
///
/// In cases where `i - M` would be < 0, or `i + M` would be >= `K`, then we simply slide our window
/// as far as we can, while keeping it's size constant. This means that in the first case, where we
/// are very far to the left, we always select points `0..N` from the input, and on the far right
/// side of the input we always select points `(K - N)..K`.
///
/// Typically, when our input window is not on the extreme right or left, we will use the coefficient
/// row in the middle of the matrix (that is coefficient row with index M). However, in cases where
/// we are on the extreme left, we will use coefficient rows 0..M. For the furthest left, we use row
/// 0, then we use row 1, etc until we reach row M. Once we reach row M, then we use row M until we
/// hit the far right side. Once we approach the far right side, we use row M+1, M+2, M+3 until the
/// input data ends (when we use coefficient row N - 1).
///
/// We refer to the middle coefficient row as `row t=0`. The row to the left of it is `row t=-1`. The
/// row to the right of it is `row t=1`. In general, you can say: Use row t=0, except when the point
/// of interest is not centered in the input window. In those cases you use the coefficient row
/// corresponding with the offset of point of interest from the center of the input window.
///
/// ## Math During Setup
///
/// The coefficient computation is beyond my powers of explanation. However, there are a few things
/// to note:
/// * The setup time scales with some power >2 of the window size, it seems. Keep in mind that the
///   window size is the number of points considered when smoothing any given point. The higher this
///   number, the smoother the data will be. This scales linearly with the input data, but the setup
///   time scales differently. It is important to keep this tradeoff in mind.
/// * The coefficients computed by the black box code (the math I don't understand) does not sum to 1,
///   but since we're multiplying and summing the data by these coefficients, and we don't want to
///   scale the input data at all, we must normalize each coefficient row so that it sums to 1.
///

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
    /// The number of coefficients to compute (the number of nearby points to convolve when computing any given point)
    pub window_size: u64,
    /// What degree to use for the polynomials
    pub degree: u64,
    /// What smoothed derivative to compute (0 means just smooth the data)
    pub order: u64,
}

impl SavitzkyGolayConfig {
    pub fn into_mapper(self, size: usize) -> SavitzkyGolayMapper {
        SavitzkyGolayMapper::new(size, self)
    }

    pub fn compute_coefficients(&self) -> Vec<Vec<Fraction>> {
        if self.window_size % 2 == 0 || self.window_size < 3 {
            panic!("invalid window size {}", self.window_size)
        }

        let half_window_size = (self.window_size as i64) / 2;
        (-half_window_size..=half_window_size)
            .map(move |time_offset| weights(
                half_window_size,
                time_offset.into(),
                self.degree.into(),
                self.order.into(),
            ))
            .collect()
    }
}

#[derive(Debug)]
pub struct SavitzkyGolayMapper {
    buf: Vec<f64>,
    cap: usize,
    coefficients: Vec<Vec<Fraction>>,
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
        let coefficients = self.coefficients.as_slice();
        let n_coefficients = coefficients.len();
        let half_size = n_coefficients / 2;

        self.buf.clear();
        // convolution!
        // slides a window of fixed size along the input data
        //
        // each window has a point of interest which is at some offset from center of window
        // input data for each window is window.start..window.end and point of interest is at
        // window.start + window.offset + half_size
        self.buf.extend(
            SlidingWindow::new(coefficients.len(), input.len())
                // grab the input data we need and the coefficient row we need
                .map(move |win| (
                    &input[win.start..win.end],
                    &coefficients[(win.offset + (half_size as isize)) as usize],
                ))
                // pointwise multiply the data, and get the sum
                .map(move |(input, coefficients)|
                    Iterator::zip(input.iter().copied(), coefficients.iter().copied())
                        .map(move |(i, cf)| cf * i)
                        .sum::<f64>()));

        Ok(Some(&self.buf[..]))
    }
}

impl MapperToChanneled<f64, f64> for SavitzkyGolayMapper {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let cap = self.cap;
        self.channeled(cap)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct SlidingWindow {
    // configurable inputs
    window: usize,
    size: usize,

    // cached constant
    half_window: usize,

    // state
    at: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct WindowPointer {
    start: usize,
    end: usize,
    offset: isize,
}

impl SlidingWindow {
    fn new(window: usize, size: usize) -> Self {
        Self {
            window,
            size,
            half_window: window / 2,
            at: 0,
        }
    }
}

impl Iterator for SlidingWindow {
    type Item = WindowPointer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.size {
            return None;
        }

        let next = self.at;
        self.at += 1;
        let tail_at = (self.size - self.half_window) - 1;
        Some(if next <= self.half_window {
            WindowPointer {
                start: 0,
                end: self.window,
                offset: (next as isize) - (self.half_window as isize),
            }
        } else if next > tail_at {
            WindowPointer {
                start: self.size - self.window,
                end: self.size,
                offset: (next - tail_at) as isize,
            }
        } else {
            WindowPointer {
                start: next - self.half_window,
                end: next + self.half_window + 1,
                offset: 0,
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }

    fn count(self) -> usize where
        Self: Sized,
    {
        self.size
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let amount_left = self.size - self.at;
        if amount_left < n {
            self.at = self.size;
            None
        } else {
            self.at += n - 1;
            self.next()
        }
    }
}