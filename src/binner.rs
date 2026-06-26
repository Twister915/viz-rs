use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::{VizFloat, log_timed};
use anyhow::Result;

pub struct Binner {
    layout: BinLayout,
    out: Vec<Channeled<VizFloat>>,
}

impl Binner {
    pub fn from_layout(layout: BinLayout) -> Self {
        let n_bins = layout.len();
        Self {
            layout,
            out: Vec::with_capacity(n_bins),
        }
    }
}

impl FramedMapper for Binner {
    type Input = Channeled<VizFloat>;
    type Output = Channeled<VizFloat>;

    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<VizFloat>],
    ) -> Result<Option<&'a mut [Channeled<VizFloat>]>> {
        if input.len() != self.layout.input_size() {
            return Ok(None);
        }

        if input.is_empty() {
            return Ok(None);
        }

        let zero = input[0].map(move |_| 0.0);
        self.out.clear();
        self.out.resize(self.layout.len(), zero);

        for bin_idx in 0..self.layout.len() {
            let start = self.layout.indexes[bin_idx].min(input.len());
            let end = self.layout.indexes[bin_idx + 1].min(input.len());
            if start >= end {
                continue;
            }

            let mut count = 0usize;
            let mut power_sum = zero;
            for elem in input[start..end].iter().copied() {
                if elem.map(move |elem| elem.is_finite()).and() {
                    let power = elem.map(move |v| v * v);
                    power_sum = power_sum
                        .zip(power)
                        .expect("mixed stereo/mono?")
                        .map(move |(sum, v)| sum + v);
                    count += 1;
                }
            }

            if count != 0 {
                let count = count as VizFloat;
                self.out[bin_idx] = power_sum.map(move |v| (v / count).sqrt());
            }
        }

        Ok(Some(self.out.as_mut_slice()))
    }

    fn map_frame_size(&self, _: usize) -> usize {
        self.layout.len()
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct BinConfig {
    pub bins: usize,
    pub input_size: usize,
    pub sample_rate: usize,
    pub fmin: VizFloat,
    pub fmax: VizFloat,
    pub gamma: VizFloat,
}

#[derive(Clone, Debug)]
pub struct BinLayout {
    indexes: Vec<usize>,
    input_size: usize,
    bandwidth_per_src_bin: VizFloat,
}

impl BinLayout {
    pub fn new(config: BinConfig) -> Self {
        log_timed(format!("compute bin constants for {:?}", config), || {
            let indexes = compute_bin_indexes(&config);
            Self {
                indexes,
                input_size: config.input_size,
                bandwidth_per_src_bin: bandwidth_per_src_bin(&config),
            }
        })
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn len(&self) -> usize {
        self.indexes.len().saturating_sub(1)
    }

    pub fn bin_size(&self, bin_idx: usize) -> usize {
        self.indexes
            .get(bin_idx..=bin_idx + 1)
            .and_then(move |win| win.first().zip(win.get(1)))
            .map(move |(start, end)| end.saturating_sub(*start))
            .unwrap_or(0)
    }

    pub fn center_hz(&self, bin_idx: usize) -> VizFloat {
        let Some(start) = self.indexes.get(bin_idx).copied() else {
            return 0.0;
        };
        let Some(end) = self.indexes.get(bin_idx + 1).copied() else {
            return 0.0;
        };
        if start >= end {
            return 0.0;
        }

        let low = self.hz_for_idx(start);
        let high = self.hz_for_idx(end - 1);
        if low > 0.0 && high > 0.0 {
            (low * high).sqrt()
        } else {
            (low + high) / 2.0
        }
    }

    #[cfg(test)]
    fn indexes(&self) -> &[usize] {
        &self.indexes
    }

    fn hz_for_idx(&self, idx: usize) -> VizFloat {
        ((idx + 1) as VizFloat) * self.bandwidth_per_src_bin
    }
}

fn bandwidth_per_src_bin(config: &BinConfig) -> VizFloat {
    if config.input_size == 0 {
        0.0
    } else {
        ((config.sample_rate as VizFloat) / 2.0) / (config.input_size as VizFloat)
    }
}

fn compute_bin_indexes(config: &BinConfig) -> Vec<usize> {
    if config.input_size == 0 || config.bins == 0 {
        return vec![0];
    }

    let nyquist = (config.sample_rate as VizFloat) / 2.0;
    let bandwidth_per_src_bin = bandwidth_per_src_bin(config);
    let fmin = config.fmin.max(0.0).min(nyquist);
    let fmax = config.fmax.max(fmin).min(nyquist);
    let idx_for_edge = move |hz: VizFloat| -> usize {
        if hz >= nyquist {
            config.input_size
        } else if hz <= bandwidth_per_src_bin {
            0
        } else {
            ((hz / bandwidth_per_src_bin).ceil() as usize)
                .saturating_sub(1)
                .min(config.input_size)
        }
    };

    let mut start = idx_for_edge(fmin);
    if start >= config.input_size {
        start = config.input_size - 1;
    }

    let mut end = idx_for_edge(fmax);
    if end <= start {
        end = (start + 1).min(config.input_size);
    }

    let available_bins = end.saturating_sub(start);
    let n_bins = config.bins.min(available_bins.max(1));
    if n_bins < config.bins {
        println!(
            "using {} bins for {} desired bins; only {} FFT bins are available in range",
            n_bins, config.bins, available_bins
        );
    }

    let mut out = Vec::with_capacity(n_bins + 1);
    for bin in 0..=n_bins {
        let idx = if bin == 0 {
            start
        } else if bin == n_bins {
            end
        } else {
            let t = (bin as VizFloat) / (n_bins as VizFloat);
            let hz = fmin + ((fmax - fmin) * t.powf(config.gamma));
            idx_for_edge(hz)
        };
        out.push(idx.min(config.input_size));
    }

    out[0] = start;
    out[n_bins] = end;
    for idx in 1..n_bins {
        let min_idx = out[idx - 1] + 1;
        let max_idx = end - (n_bins - idx);
        out[idx] = out[idx].max(min_idx).min(max_idx);
    }

    let hz_for_idx = move |idx: usize| ((idx + 1) as VizFloat) * bandwidth_per_src_bin;
    let sizes = out
        .windows(2)
        .map(move |win| win[1] - win[0])
        .collect::<Vec<usize>>();

    sizes
        .iter()
        .copied()
        .zip(out.windows(2).map(move |win| {
            let from = win[0];
            let to = win[1];
            let from_hz = hz_for_idx(from.min(config.input_size - 1));
            let to_hz = if to == 0 {
                0.0
            } else {
                hz_for_idx((to - 1).min(config.input_size - 1))
            };
            ((from, from_hz), (to, to_hz))
        }))
        .enumerate()
        .for_each(move |(idx, (size, ((from, from_hz), (to, to_hz))))| {
            println!(
                "bin[{}] size={} :: {}..{} {:.2}Hz..{:.2}Hz",
                idx, size, from, to, from_hz, to_hz,
            )
        });

    let total_size = sizes.iter().copied().sum::<usize>();
    println!(
        "total size :: {} (/ {}) -> {}",
        total_size, config.input_size, n_bins
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framed::FramedMapper;

    #[test]
    fn bin_indexes_account_for_skipped_dc_bin() {
        let indexes = compute_bin_indexes(&BinConfig {
            bins: 2,
            input_size: 4,
            sample_rate: 8,
            fmin: 2.0,
            fmax: 4.0,
            gamma: 1.0,
        });

        assert_eq!(indexes, vec![1, 2, 4]);
    }

    #[test]
    fn bin_layout_reports_widths_and_centers() {
        let layout = BinLayout::new(BinConfig {
            bins: 2,
            input_size: 4,
            sample_rate: 8,
            fmin: 1.0,
            fmax: 4.0,
            gamma: 1.0,
        });

        assert_eq!(layout.indexes(), &[0, 2, 4]);
        assert_eq!(layout.bin_size(0), 2);
        assert_eq!(layout.bin_size(1), 2);
        assert!((layout.center_hz(0) - VizFloat::sqrt(2.0)).abs() < 1e-12);
        assert!((layout.center_hz(1) - VizFloat::sqrt(12.0)).abs() < 1e-12);
    }

    #[test]
    fn bins_are_rms_averaged_by_their_own_width() {
        let mut binner = Binner::from_layout(BinLayout::new(BinConfig {
            bins: 2,
            input_size: 4,
            sample_rate: 8,
            fmin: 1.0,
            fmax: 4.0,
            gamma: 1.0,
        }));
        let mut input = [
            Channeled::Mono(1.0),
            Channeled::Mono(1.0),
            Channeled::Mono(3.0),
            Channeled::Mono(3.0),
        ];

        let out = binner.map(&mut input).unwrap().unwrap();

        assert_eq!(out.len(), 2);
        assert!(
            (match out[0] {
                Channeled::Mono(v) => v,
                Channeled::Stereo(_, _) => panic!("expected mono"),
            } - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            (match out[1] {
                Channeled::Mono(v) => v,
                Channeled::Stereo(_, _) => panic!("expected mono"),
            } - 3.0)
                .abs()
                < 1e-12
        );
    }
}
