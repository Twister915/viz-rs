use anyhow::Result;
use std::iter::{FusedIterator, TrustedLen};

pub fn two_dimensional_vec<E>(sizes: &Vec<usize>) -> Vec<Vec<E>> {
    sizes
        .iter()
        .copied()
        .map(move |size| Vec::with_capacity(size))
        .collect()
}

pub fn try_use_iter<I, T, F>(source: I, mut consumer: F) -> Result<()>
where
    I: Iterator<Item = Result<T>>,
    F: for<'a> FnMut(&'a mut TryUseValueIter<I>) -> (),
{
    let mut out = TryUseValueIter { source, err: None };
    consumer(&mut out);
    if let Some(err) = out.err.take() {
        Err(err)
    } else {
        Ok(())
    }
}

pub struct TryUseValueIter<I> {
    source: I,
    err: Option<anyhow::Error>,
}

impl<I, R> Iterator for TryUseValueIter<I>
where
    I: Iterator<Item = Result<R>>,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        if self.err.is_some() {
            None
        } else {
            match self.source.next() {
                None => None,
                Some(Err(err)) => {
                    self.err = Some(err);
                    None
                }
                Some(Ok(v)) => Some(v),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.source.size_hint();
        (0, max)
    }
}

unsafe impl<I, R> TrustedLen for TryUseValueIter<I> where I: Iterator<Item = Result<R>> + TrustedLen {}
impl<I, R> ExactSizeIterator for TryUseValueIter<I> where
    I: Iterator<Item = Result<R>> + ExactSizeIterator
{
}
impl<I, R> FusedIterator for TryUseValueIter<I> where I: Iterator<Item = Result<R>> + FusedIterator {}
