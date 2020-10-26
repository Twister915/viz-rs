use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use anyhow::{anyhow, Result};
use fftw::array::AlignedVec;
use fftw::plan::{R2CPlan, R2CPlan64};
use fftw::types::{c64, Flag};

pub struct FramedFft {
    plan: R2CPlan64,
    bufs: Option<Channeled<Bufs>>,
    out: Vec<Channeled<f64>>,
    n_out: usize,
    n_in: usize,
}

struct Bufs {
    input: AlignedVec<f64>,
    output: AlignedVec<c64>,
    real_output: Vec<f64>,
}

impl Bufs {
    fn new(in_size: usize) -> Self {
        let out_size = (in_size / 2) + 1;
        Self {
            input: AlignedVec::new(in_size),
            output: AlignedVec::new(out_size),
            real_output: Vec::with_capacity(out_size - 1),
        }
    }
}

impl FramedFft {
    pub fn new(cap: usize) -> Result<Self> {
        let n_out = cap / 2;
        Ok(Self {
            plan: R2CPlan64::aligned(&[cap], Flag::MEASURE | Flag::DESTROYINPUT)
                .map_err(map_fftw_error)?,
            bufs: None,
            out: Vec::with_capacity(n_out),
            n_out,
            n_in: cap,
        })
    }
}

impl FramedMapper<Channeled<f64>, Channeled<f64>> for FramedFft {
    fn map(&mut self, input: &[Channeled<f64>]) -> Result<Option<&[Channeled<f64>]>> {
        let bufs = &mut self.bufs;
        if bufs.is_none() {
            *bufs = Some(match &input[0] {
                Channeled::Stereo(_, _) => {
                    Channeled::Stereo(Bufs::new(self.n_in), Bufs::new(self.n_in))
                }
                Channeled::Mono(_) => Channeled::Mono(Bufs::new(self.n_in)),
            })
        }

        let bufs = bufs.as_mut().unwrap();
        consume_input(bufs, input)?;
        perform_transform(bufs, &mut self.plan, &mut self.out)?;
        Ok(Some(&self.out))
    }

    fn map_frame_size(&self, _: usize) -> usize {
        self.n_out
    }
}

fn consume_input(bufs: &mut Channeled<Bufs>, input: &[Channeled<f64>]) -> Result<()> {
    use Channeled::*;

    for (idx, elem) in input.iter().enumerate() {
        match elem {
            Stereo(a, b) => {
                if let Stereo(buf_a, buf_b) = bufs {
                    buf_a.input[idx] = *a;
                    buf_b.input[idx] = *b;
                } else {
                    return Err(anyhow!("mixed mono/stereo?"));
                }
            }
            Mono(v) => {
                if let Mono(buf) = bufs {
                    buf.input[idx] = *v;
                } else {
                    return Err(anyhow!("mixed mono/stereo?"));
                }
            }
        }
    }

    let input_len = input.len();
    match bufs {
        Stereo(a, b) => {
            fill_rest_zero(input_len, &mut a.input);
            fill_rest_zero(input_len, &mut b.input);
        }
        Mono(v) => {
            fill_rest_zero(input_len, &mut v.input);
        }
    };

    Ok(())
}

fn fill_rest_zero(input_len: usize, target: &mut AlignedVec<f64>) {
    for i in input_len..target.len() {
        target[i] = 0.0;
    }
}

fn perform_transform(
    bufs: &mut Channeled<Bufs>,
    plan: &mut R2CPlan64,
    to: &mut Vec<Channeled<f64>>,
) -> Result<()> {
    use Channeled::*;
    match bufs {
        Stereo(a, b) => {
            perform_transform_once(plan, a)?;
            perform_transform_once(plan, b)?;
            zip_stereo_output(a, b, to);
        }
        Mono(v) => {
            perform_transform_once(plan, v)?;
            zip_mono_output(v, to);
        }
    }

    Ok(())
}

fn zip_stereo_output(a: &mut Bufs, b: &mut Bufs, to: &mut Vec<Channeled<f64>>) {
    to.clear();
    for idx in 0..a.real_output.len() {
        let av = a.real_output[idx];
        let bv = b.real_output[idx];
        to.push(Channeled::Stereo(av, bv))
    }
}

fn zip_mono_output(v: &mut Bufs, to: &mut Vec<Channeled<f64>>) {
    to.clear();
    to.extend(v.real_output.drain(..).map(move |v| Channeled::Mono(v)))
}

fn perform_transform_once(plan: &mut R2CPlan64, buf: &mut Bufs) -> Result<()> {
    plan.r2c(buf.input.as_slice_mut(), buf.output.as_slice_mut())
        .map_err(map_fftw_error)?;
    buf.real_output.clear();
    for result in buf.output.iter().skip(1) {
        buf.real_output.push(result.norm());
    }

    Ok(())
}

fn map_fftw_error(err: fftw::error::Error) -> anyhow::Error {
    anyhow!("fftw: {:?}", err)
}
