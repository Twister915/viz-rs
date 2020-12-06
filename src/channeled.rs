use anyhow::Result;
use std::fmt;
use std::iter::{FusedIterator, TrustedLen};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channeled<T> {
    Mono(T),
    Stereo(T, T),
}

impl<T> fmt::Display for Channeled<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Channeled::*;
        match self {
            Mono(v) => v.fmt(f),
            Stereo(a, b) => write!(f, "({}, {})", a, b),
        }
    }
}

impl<T> Default for Channeled<T>
where
    T: Default,
{
    fn default() -> Self {
        Channeled::Mono(T::default())
    }
}

impl<T> Channeled<T> {
    pub fn map<F, R>(self, mut f: F) -> Channeled<R>
    where
        F: FnMut(T) -> R,
    {
        use Channeled::*;
        match self {
            Stereo(a, b) => Stereo(f(a), f(b)),
            Mono(a) => Mono(f(a)),
        }
    }

    pub fn try_map<F, R>(self, f: F) -> Result<Channeled<R>>
    where
        F: FnMut(T) -> Result<R>,
    {
        self.map(f).invert_result()
    }

    pub fn for_each<F>(self, f: F)
    where
        F: FnMut(T) -> (),
    {
        self.map(f);
    }

    pub fn as_mut_ref(&mut self) -> Channeled<&mut T> {
        use Channeled::*;
        match self {
            Stereo(a, b) => Stereo(a, b),
            Mono(v) => Mono(v),
        }
    }

    pub fn as_ref(&self) -> Channeled<&T> {
        use Channeled::*;
        match self {
            Stereo(a, b) => Stereo(a, b),
            Mono(v) => Mono(v),
        }
    }

    pub fn zip<O>(self, other: Channeled<O>) -> Option<Channeled<(T, O)>> {
        use Channeled::*;
        match (self, other) {
            (Stereo(al, ar), Stereo(bl, br)) => Some(Stereo((al, bl), (ar, br))),
            (Mono(a), Mono(b)) => Some(Mono((a, b))),
            _ => None
        }
    }
}

impl<R, X> Channeled<Result<R, X>> {
    pub fn invert_result(self) -> Result<Channeled<R>, X> {
        use Channeled::*;
        match self {
            Stereo(Ok(lv), Ok(rv)) => Ok(Stereo(lv, rv)),
            Mono(Ok(v)) => Ok(Mono(v)),
            Stereo(Err(err), _) | Stereo(_, Err(err)) | Mono(Err(err)) => Err(err),
        }
    }
}

impl Channeled<bool> {
    pub fn and(self) -> bool {
        use Channeled::*;
        match self {
            Stereo(a, b) => a && b,
            Mono(v) => v,
        }
    }
}

impl<T> IntoIterator for Channeled<T>
where
    T: IntoIterator,
{
    type Item = Channeled<T::Item>;
    type IntoIter = ChanneledIter<T::IntoIter>;

    fn into_iter(self) -> ChanneledIter<T::IntoIter> {
        ChanneledIter {
            iters: self.map(move |i| i.into_iter()),
        }
    }
}

pub struct ChanneledIter<I> {
    iters: Channeled<I>,
}

impl<I> Iterator for ChanneledIter<I>
where
    I: Iterator,
{
    type Item = Channeled<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        use Channeled::*;
        match self.iters.as_mut_ref().map(move |v| v.next()) {
            Stereo(Some(vl), Some(vr)) => Some(Stereo(vl, vr)),
            Mono(Some(v)) => Some(Mono(v)),
            _ => None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        use Channeled::*;
        match &self.iters {
            Stereo(a, b) => {
                let (al, ah) = a.size_hint();
                let (bl, bh) = b.size_hint();
                (
                    std::cmp::min(al, bl),
                    ah.and_then(move |ah| bh.map(move |bh| std::cmp::max(ah, bh))),
                )
            }
            Mono(v) => v.size_hint(),
        }
    }
}

unsafe impl<I> TrustedLen for ChanneledIter<I> where I: Iterator + TrustedLen {}

impl<I> ExactSizeIterator for ChanneledIter<I> where I: Iterator + ExactSizeIterator {}

impl<I> FusedIterator for ChanneledIter<I> where I: Iterator + FusedIterator {}
