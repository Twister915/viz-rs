use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use anyhow::Result;

pub struct ExponentialSmoothing {
    previous: Vec<Vec<Channeled<f64>>>,
    n_prev: usize,
    alpha: f64,
}

impl ExponentialSmoothing {
    pub fn new(seek_back_limit: usize, alpha: f64) -> Self {
        Self {
            previous: Vec::with_capacity(seek_back_limit),
            n_prev: seek_back_limit,
            alpha,
        }
    }
}

impl FramedMapper<Channeled<f64>, Channeled<f64>> for ExponentialSmoothing {
    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<f64>],
    ) -> Result<Option<&'a mut [Channeled<f64>]>> {
        if let Some(prev) = self.previous.get(0) {
            let alpha = self.alpha;
            let alpha_inv = 1.0 - alpha;

            input
                .iter_mut()
                .map(move |c| c.as_mut_ref())
                .zip(prev.iter().copied())
                .map(move |(new, pre)| new.zip(pre).expect("mono/stereo should match"))
                .for_each(move |zipped| {
                    zipped.for_each(move |(new, prev)| *new = (*new * alpha_inv) + (prev * alpha))
                })
        }

        // copy the computed data into the prev vec
        let mut new_prev = if self.previous.len() < self.n_prev {
            Vec::with_capacity(input.len())
        } else {
            let mut tail = self.previous.remove(self.previous.len() - 1);
            tail.clear();
            tail
        };

        new_prev.extend_from_slice(input);
        self.previous.insert(0, new_prev);

        Ok(Some(input))
    }
}
