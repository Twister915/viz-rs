use crate::framed::{ChanneledMapperWrapper, FramedMapper, MapperToChanneled};
use crate::util::two_dimensional_vec;
use anyhow::Result;

pub struct Binner {
    bufs: Vec<Vec<f64>>,
    out_buf: Vec<f64>,
    indexes: Vec<usize>,
    n_bins: usize,
    in_size: usize,
}

impl Binner {
    pub fn new(config: BinConfig) -> Self {
        let indexes = compute_bin_indexes(&config);
        let n_bins = indexes.len() - 1;
        let sizes = compute_bin_sizes(&indexes);
        let bufs = two_dimensional_vec(&sizes);
        let out_buf = Vec::with_capacity(n_bins);
        let in_size = config.input_size;
        Self {
            bufs,
            out_buf,
            indexes,
            n_bins,
            in_size,
        }
    }

    fn aggregate_bufs(&mut self) {
        self.out_buf.clear();
        for elem in self.bufs.iter_mut() {
            self.out_buf.push(elem.drain(..).sum());
        }
    }
}

impl FramedMapper<f64, f64> for Binner {
    fn map(&mut self, input: &[f64]) -> Result<Option<&[f64]>> {
        if input.len() != self.in_size {
            return Ok(None);
        }

        let mut bin_idx = 0usize;
        let bufs = self.bufs.as_mut_slice();
        let idx_slice = self.indexes.as_slice();
        for (idx, elem) in input.iter().enumerate() {
            let this_bin_start_at = &idx_slice[bin_idx];
            if idx < *this_bin_start_at {
                continue;
            }

            let next_bin_start_at = &idx_slice[bin_idx + 1];
            if idx >= *next_bin_start_at {
                bin_idx += 1;
            }

            if bin_idx >= self.n_bins {
                break;
            }

            let elem = *elem;
            if elem.is_finite() {
                bufs[bin_idx].push(elem);
            }
        }

        self.aggregate_bufs();
        Ok(Some(self.out_buf.as_slice()))
    }

    fn map_frame_size(&self, _: usize) -> usize {
        self.n_bins
    }
}

impl MapperToChanneled<f64, f64> for Binner {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let in_size = self.in_size;
        let out_size = self.n_bins;
        ChanneledMapperWrapper::new(self, in_size, out_size)
    }
}

pub struct BinConfig {
    pub bins: usize,
    pub input_size: usize,
    pub sample_rate: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub gamma: f64,
}

fn compute_bin_sizes(indexes: &Vec<usize>) -> Vec<usize> {
    let mut out = Vec::with_capacity(indexes.len() - 1);
    let mut last_start_at = None;
    for cur_start_at in indexes.iter().copied() {
        if let Some(last_start_at) = last_start_at {
            out.push(cur_start_at - last_start_at);
        }

        last_start_at = Some(cur_start_at);
    }

    out
}

fn compute_bin_indexes(config: &BinConfig) -> Vec<usize> {
    let total_max_freq = (config.sample_rate as f64) / 2.0;
    let bandwidth_per_src_bin = total_max_freq / (config.input_size as f64);
    let gamma_inv = 1.0 / config.gamma;
    let n_bins = config.bins as f64;
    let mut out = vec![None; config.bins + 1];
    for i in 0..config.input_size {
        let ifl = i as f64;
        let f_start = ifl * bandwidth_per_src_bin;
        if f_start < config.fmin {
            continue;
        }

        let mut bin_idx = ((f_start / total_max_freq).powf(gamma_inv) * n_bins).round() as isize;
        if bin_idx < 0 {
            continue;
        }

        let is_last = bin_idx >= (config.bins as isize);
        if is_last {
            bin_idx = config.bins as isize;
        }

        let bin_idx = bin_idx as usize;
        match &mut out[bin_idx] {
            Some(existing) => {
                if *existing > i {
                    *existing = i;
                }
            }
            None => {
                out[bin_idx] = Some(i);
            }
        }

        if is_last {
            break;
        }
    }

    let mut has_any = false;
    let mut fin_out = Vec::with_capacity(out.len());
    for (idx, elem) in out.drain(..).enumerate() {
        if let Some(v) = elem {
            fin_out.push(v);
            has_any = true;
        } else {
            // skip this bin if it's before the 1st bin
            if has_any {
                panic!("did not discover good range for bin {}", idx)
            }
        }
    }

    fin_out
}
