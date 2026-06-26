use crate::channeled::Channeled;
use crate::framed::FramedMapper;
use crate::util::VizFloat;
use anyhow::Result;

pub struct ExponentialSmoothing {
    previous: Vec<Vec<Channeled<VizFloat>>>,
    n_prev: usize,
    alpha: VizFloat,
}

impl ExponentialSmoothing {
    pub fn new(seek_back_limit: usize, alpha: VizFloat) -> Self {
        Self {
            previous: Vec::with_capacity(seek_back_limit),
            n_prev: seek_back_limit,
            alpha,
        }
    }
}

impl FramedMapper for ExponentialSmoothing {
    type Input = Channeled<VizFloat>;
    type Output = Channeled<VizFloat>;

    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<VizFloat>],
    ) -> Result<Option<&'a mut [Channeled<VizFloat>]>> {
        if let Some(prev) = self.previous.first() {
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
