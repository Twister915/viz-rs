use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::{log_timed, slice_copy_from, VizFloat, VizComplex, VizFftPlan};
use anyhow::{anyhow, Result};
use fftw::array::AlignedVec;
use fftw::plan::R2CPlan;
use fftw::types::Flag;

pub struct FramedFft {
    plan: VizFftPlan,
    bufs: Option<Channeled<Bufs>>,
    n_out: usize,
    n_in: usize,
}

struct Bufs {
    input: AlignedVec<VizFloat>,
    output: AlignedVec<VizComplex>,
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
            VizFftPlan::aligned(&[cap], Flag::ESTIMATE | Flag::DESTROYINPUT).map_err(map_fftw_error)
        })?;
        Ok(Self {
            plan,
            bufs: None,
            n_out,
            n_in: cap,
        })
    }
}

impl FramedMapper<Channeled<VizFloat>, Channeled<VizFloat>> for FramedFft {
    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<VizFloat>],
    ) -> Result<Option<&'a mut [Channeled<VizFloat>]>> {
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
            .map(move |v| v.input.iter_mut()) // Channeled<IterMut<VizFloat>>
            .into_iter() // Iter<Channeled<&mut VizFloat>> basically
            .zip(input.iter()) // Iter<(Channeled<&mut VizFloat>, Channeled<VizFloat>)>
            .for_each(move |(dest, input)| {
                dest.zip(input.as_ref()) // Channeled<(&mut VizFloat, VizFloat)>
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

        let updated = slice_copy_from(
            input,
            bufs.as_mut_ref()
                .try_map(move |buf| {
                    // transform input data in buf: &mut Bufs
                    // input is in buf.input
                    // output (complex) will be in buf.output
                    let i = buf.input.as_slice_mut();
                    let o = buf.output.as_slice_mut();
                    plan.r2c(i, o).map_err(map_fftw_error)?;

                    // return an iterator over the output which skips the DC component (skip(1)) and
                    // converts complex data to real data using norm() (magnitude of complex number)
                    Ok(o.iter().skip(1).map(move |v| v.norm()))
                })?
                .into_iter(),
        );
        Ok(Some(updated))
    }

    fn map_frame_size(&self, _: usize) -> usize {
        self.n_out
    }
}

fn map_fftw_error(err: fftw::error::Error) -> anyhow::Error {
    anyhow!("fftw: {:?}", err)
}
