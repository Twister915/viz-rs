use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::log_timed;
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
}

impl Bufs {
    fn new(in_size: usize) -> Self {
        let out_size = (in_size / 2) + 1;
        Self {
            input: AlignedVec::new(in_size),
            output: AlignedVec::new(out_size),
        }
    }
}

impl FramedFft {
    pub fn new(cap: usize) -> Result<Self> {
        // fft is defined as having (N / 2) + 1 outputs but we skip
        // DC at index 0 so N / 2
        let n_out = cap / 2;
        let plan = log_timed(format!("plan fft for size {}", cap), || {
            R2CPlan64::aligned(&[cap], Flag::ESTIMATE | Flag::DESTROYINPUT).map_err(map_fftw_error)
        })?;
        Ok(Self {
            plan,
            bufs: None,
            out: Vec::with_capacity(n_out),
            n_out,
            n_in: cap,
        })
    }
}

impl FramedMapper<Channeled<f64>, Channeled<f64>> for FramedFft {
    fn map(&mut self, input: &[Channeled<f64>]) -> Result<Option<&[Channeled<f64>]>> {
        // lazily setup the bufs
        let bufs = if let Some(buf) = self.bufs.as_mut() {
            buf
        } else {
            // stereo needs two bufs, mono needs one buf, so this map will handle creating one for
            // each, depending on whether or not input[0] is mono or stereo
            let created = (&input[0]).map(|_| Bufs::new(self.n_in));
            self.bufs = Some(created);
            self.bufs.as_mut().unwrap()
        };

        // load input into the buffers:
        bufs.as_mut_ref()
            .map(move |v| v.input.iter_mut()) // Channeled<IterMut<f64>>
            .into_iter() // Iter<Channeled<&mut f64>> basically
            .zip(input.iter()) // Iter<(Channeled<&mut f64>, Channeled<f64>)>
            .for_each(move |(dest, input)| {
                dest.zip(input.as_ref()) // Channeled<(&mut f64, f64)>
                    .expect("mixed mono/stereo?")
                    .for_each(move |(d, i)| *d = *i)
            });

        // fill any un-filled input with 0s
        let input_len = input.len();
        bufs.as_mut_ref()
            .map(move |v| &mut v.input)
            .for_each(move |input| {
                (&mut input[input_len..])
                    .iter_mut()
                    .for_each(move |t| *t = 0.0)
            });

        let plan = &mut self.plan;
        self.out.clear();
        self.out.extend(bufs.as_mut_ref().try_map(move |buf| {
            // transform input data in buf: &mut Bufs
            // input is in buf.input
            // output (complex) will be in buf.output
            let i = buf.input.as_slice_mut();
            let o = buf.output.as_slice_mut();
            plan.r2c(i, o).map_err(map_fftw_error)?;

            // return an iterator over the output which skips the DC component (skip(1)) and
            // converts complex data to real data using norm() (magnitude of complex number)
            Ok(o.iter().skip(1).map(move |v| v.norm()))
        })?);

        Ok(Some(&self.out))
    }

    fn map_frame_size(&self, _: usize) -> usize {
        self.n_out
    }
}

fn map_fftw_error(err: fftw::error::Error) -> anyhow::Error {
    anyhow!("fftw: {:?}", err)
}
