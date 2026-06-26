use crate::channeled::Channeled;
use crate::util::try_use_iter;
use anyhow::Result;
use num_rational::Rational64;
use std::time::Duration;

pub trait Framed {
    type Item;

    fn apply_mapper<M>(self, mapper: M) -> MappedFramed<Self, M>
    where
        Self: Sized,
        M: FramedMapper<Input = Self::Item>,
    {
        MappedFramed {
            source: self,
            mapper,
        }
    }

    fn lift<F, M>(self, factory: F) -> MappedFramed<Self, M>
    where
        Self: Sized,
        F: FnOnce(usize) -> M,
        M: FramedMapper<Input = Self::Item>,
    {
        let size = self.full_frame_size();
        self.apply_mapper(factory(size))
    }

    fn try_lift<F, M, X>(self, factory: F) -> Result<MappedFramed<Self, M>, X>
    where
        Self: Sized,
        F: FnOnce(usize) -> Result<M, X>,
        M: FramedMapper<Input = Self::Item>,
    {
        let size = self.full_frame_size();
        Ok(self.apply_mapper(factory(size)?))
    }

    fn compose<F, O>(self, f: F) -> O
    where
        F: FnOnce(Self) -> O,
        Self: Sized,
    {
        f(self)
    }

    fn seek_frame(&mut self, n: isize) -> Result<()>;

    fn next_frame(&mut self) -> Result<Option<&mut [Self::Item]>>;

    fn full_frame_size(&self) -> usize;

    fn map<F, R>(self, mapper: F) -> MapFramed<Self, F, R>
    where
        Self: Sized,
        F: Fn(&Self::Item) -> R,
    {
        let cap = self.full_frame_size();
        MapFramed {
            source: self,
            mapper,
            buf: Vec::with_capacity(cap),
        }
    }

    fn map_mut<F>(self, mapper: F) -> MapMutFramed<Self, F>
    where
        Self: Sized,
        F: FnMut(&mut Self::Item),
    {
        MapMutFramed {
            source: self,
            mapper,
        }
    }
}

pub trait Samples: Sampled {
    type Item;

    fn compose<F, O>(self, f: F) -> O
    where
        F: FnOnce(Self) -> O,
        Self: Sized,
    {
        f(self)
    }

    fn seek_samples(&mut self, n: isize) -> Result<()>;

    fn next_sample(&mut self) -> Result<Option<Self::Item>>;

    fn num_samples_remain(&self) -> usize;

    fn has_more_samples(&self) -> bool {
        self.num_samples_remain() != 0
    }

    fn map<F, R>(self, mapper: F) -> MappedSamples<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> R,
    {
        MappedSamples::new(self, mapper)
    }
}

pub trait Sampled {
    fn samples_from_dur(&self, dur: Duration) -> usize {
        *((Rational64::new(self.sample_rate() as i64, 1_000_000_000)) * (dur.as_nanos() as i64))
            .round()
            .numer() as usize
    }

    fn sample_rate(&self) -> usize;
}

#[macro_export]
macro_rules! delegate_impls {
    ($ty:ident <$($g: ident),+>, $s: ident, $fld: ident) => {
        impl<$($g),+> $crate::framed::Sampled for $ty<$($g),+> where $s: $crate::framed::Sampled {
            fn sample_rate(&self) -> usize {
                self.$fld.sample_rate()
            }
        }
    }
}

pub trait FramedMapper {
    type Input;
    type Output;

    fn map<'a>(
        &'a mut self,
        input: &'a mut [Self::Input],
    ) -> Result<Option<&'a mut [Self::Output]>>;

    fn map_frame_size(&self, orig: usize) -> usize {
        orig
    }
}

pub struct MapMutFramed<S, F> {
    source: S,
    mapper: F,
}

impl<S, F> Framed for MapMutFramed<S, F>
where
    S: Framed,
    F: FnMut(&mut S::Item),
{
    type Item = S::Item;

    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [Self::Item]>> {
        let Some(input) = self.source.next_frame()? else {
            return Ok(None);
        };
        input.iter_mut().for_each(&mut self.mapper);
        Ok(Some(input))
    }

    fn full_frame_size(&self) -> usize {
        self.source.full_frame_size()
    }
}

delegate_impls!(MapMutFramed<S, F>, S, source);

pub struct MapFramed<S, F, R> {
    source: S,
    mapper: F,
    buf: Vec<R>,
}

impl<S, F, R> Framed for MapFramed<S, F, R>
where
    S: Framed,
    F: Fn(&S::Item) -> R,
{
    type Item = R;

    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [Self::Item]>> {
        let Some(input) = self.source.next_frame()? else {
            return Ok(None);
        };
        self.buf.clear();
        let mapper = &self.mapper;
        self.buf.extend(input.iter().map(mapper));
        Ok(Some(self.buf.as_mut_slice()))
    }

    fn full_frame_size(&self) -> usize {
        self.source.full_frame_size()
    }
}

delegate_impls!(MapFramed<S, F, R>, S, source);

pub struct MappedFramed<S, M> {
    source: S,
    mapper: M,
}

impl<S, M> Framed for MappedFramed<S, M>
where
    S: Framed<Item = M::Input>,
    M: FramedMapper,
{
    type Item = M::Output;

    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [Self::Item]>> {
        if let Some(data) = self.source.next_frame()? {
            self.mapper.map(data)
        } else {
            Ok(None)
        }
    }

    fn full_frame_size(&self) -> usize {
        self.mapper.map_frame_size(self.source.full_frame_size())
    }
}

delegate_impls!(MappedFramed<S, M>, S, source);

pub struct MappedSamples<S, M> {
    source: S,
    mapper: M,
}

impl<S, M> MappedSamples<S, M> {
    pub fn new(source: S, mapper: M) -> Self {
        Self { source, mapper }
    }
}

delegate_impls!(MappedSamples<S, M>, S, source);

impl<S, M, R> Samples for MappedSamples<S, M>
where
    S: Samples + Sampled,
    M: Fn(S::Item) -> R,
{
    type Item = R;

    fn seek_samples(&mut self, n: isize) -> Result<()> {
        self.source.seek_samples(n)
    }

    fn next_sample(&mut self) -> Result<Option<Self::Item>> {
        Ok(if let Some(next) = self.source.next_sample()? {
            let mapper = &self.mapper;
            Some(mapper(next))
        } else {
            None
        })
    }

    fn num_samples_remain(&self) -> usize {
        self.source.num_samples_remain()
    }
}

pub struct ChanneledMapperWrapper<M, T, R> {
    mapper: M,
    in_buf: Vec<Channeled<T>>,
    out_buf: Vec<R>,
}

impl<T, R, M> FramedMapper for ChanneledMapperWrapper<M, T, R>
where
    M: FramedMapper<Input = Channeled<T>, Output = Channeled<R>>,
    T: Copy,
    R: Copy,
{
    type Input = T;
    type Output = R;

    fn map<'a>(&'a mut self, input: &'a mut [T]) -> Result<Option<&'a mut [R]>> {
        self.in_buf.clear();
        self.in_buf
            .extend(input.iter().copied().map(move |i| Channeled::Mono(i)));
        if let Some(next) = self.mapper.map(&mut self.in_buf)? {
            let out = &mut self.out_buf;
            out.clear();

            try_use_iter(
                next.iter().map(move |v| match v {
                    Channeled::Mono(v) => Ok(*v),
                    _ => Err(anyhow::anyhow!("mono return from stereo data")),
                }),
                |itr| out.extend(itr),
            )?;

            Ok(Some(out.as_mut_slice()))
        } else {
            Ok(None)
        }
    }

    fn map_frame_size(&self, orig: usize) -> usize {
        self.mapper.map_frame_size(orig)
    }
}

pub trait SplitChanneledFramedMapper<T, R>:
    FramedMapper<Input = Channeled<T>, Output = Channeled<R>> + Sized
{
    fn split_channeled(self, cap: usize) -> ChanneledMapperWrapper<Self, T, R> {
        let cap_mapped = self.map_frame_size(cap);
        ChanneledMapperWrapper {
            mapper: self,
            in_buf: Vec::with_capacity(cap),
            out_buf: Vec::with_capacity(cap_mapped),
        }
    }
}

impl<T, R, M> SplitChanneledFramedMapper<T, R> for M where
    M: FramedMapper<Input = Channeled<T>, Output = Channeled<R>> + Sized
{
}
