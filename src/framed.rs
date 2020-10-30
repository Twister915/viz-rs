use crate::channeled::Channeled;
use crate::util::try_use_iter;
use anyhow::Result;
use num_rational::Rational64;
use std::marker::PhantomData;
use std::time::Duration;

pub trait Framed<E> {
    fn apply_mapper<M, R>(self, mapper: M) -> MappedFramed<Self, M, E, R>
    where
        Self: Sized,
        M: FramedMapper<E, R>,
    {
        MappedFramed {
            source: self,
            mapper,
            _src_typ: PhantomData,
            _dst_typ: PhantomData,
        }
    }

    fn lift<F, M, R>(self, factory: F) -> MappedFramed<Self, M, E, R>
    where
        Self: Sized,
        F: FnOnce(usize) -> M,
        M: FramedMapper<E, R>,
    {
        let size = self.full_frame_size();
        self.apply_mapper(factory(size))
    }

    fn try_lift<F, M, X, R>(self, factory: F) -> Result<MappedFramed<Self, M, E, R>, X>
    where
        Self: Sized,
        F: FnOnce(usize) -> Result<M, X>,
        M: FramedMapper<E, R>,
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

    fn next_frame(&mut self) -> Result<Option<&mut [E]>>;

    fn num_frames(&self) -> usize;

    fn num_frames_remain(&self) -> usize;

    fn num_full_frames(&self) -> usize;

    fn num_full_frames_remain(&self) -> usize {
        let total_full_frames = self.num_full_frames();
        let frames_consumed = self.num_frames() - self.num_frames_remain();
        if frames_consumed > total_full_frames {
            0
        } else {
            total_full_frames - frames_consumed
        }
    }

    fn full_frame_size(&self) -> usize;

    fn map<F, R>(self, mapper: F) -> MappedFramed<Self, FramedMapFn<E, R, F>, E, R>
    where
        Self: Sized,
        F: Fn(&E) -> R,
    {
        self.lift(move |cap| FramedMapFn {
            mapper,
            buf: Vec::with_capacity(cap),
            _in_typ: PhantomData,
        })
    }

    fn map_mut<F>(self, mapper: F) -> MappedFramed<Self, FramedMutMapFn<E, F>, E, E>
    where
        Self: Sized,
        F: FnMut(&mut E) -> (),
    {
        self.lift(move |_| FramedMutMapFn {
            mapper,
            _in_typ: PhantomData,
        })
    }

    fn collect(mut self) -> Result<Vec<Vec<E>>>
    where
        Self: Sized,
        E: Copy,
    {
        let mut big_out_buf = Vec::with_capacity(self.num_frames_remain());
        while let Some(frame) = self.next_frame()? {
            big_out_buf.push(frame.iter().copied().collect::<Vec<_>>());
        }

        Ok(big_out_buf)
    }
}

pub trait Samples<T>: Sampled {
    fn compose<F, O>(self, f: F) -> O
    where
        F: FnOnce(Self) -> O,
        Self: Sized,
    {
        f(self)
    }

    fn seek_samples(&mut self, n: isize) -> Result<()>;

    fn next_sample(&mut self) -> Result<Option<T>>;

    fn num_samples_remain(&self) -> usize;

    fn has_more_samples(&self) -> bool {
        self.num_samples_remain() != 0
    }

    fn map<F, R>(self, mapper: F) -> MappedSamples<Self, F, T, R>
    where
        Self: Sized,
        F: Fn(T) -> R,
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

    fn num_samples(&self) -> usize;

    fn duration(&self) -> Duration {
        let samples_per_ms = self.sample_rate() as f64;
        let num_samples = self.num_samples() as f64;
        let num_sec = num_samples / samples_per_ms;
        let num_nano = num_sec * 1_000_000_000.0;
        let num_nano = num_nano.floor() as u64;
        Duration::from_nanos(num_nano)
    }
}

pub trait AudioSource: Sampled {
    fn num_channels(&self) -> usize;
}

#[macro_export]
macro_rules! delegate_impls {
    ($ty:ident <$($g: ident),+>, $s: ident, $fld: ident) => {
        impl<$($g),+> crate::framed::AudioSource for $ty<$($g),+> where $s: crate::framed::AudioSource {
            fn num_channels(&self) -> usize {
                self.$fld.num_channels()
            }
        }

        impl<$($g),+> crate::framed::Sampled for $ty<$($g),+> where $s: crate::framed::Sampled {
            fn sample_rate(&self) -> usize {
                self.$fld.sample_rate()
            }

            fn num_samples(&self) -> usize {
                self.$fld.num_samples()
            }

            fn duration(&self) -> std::time::Duration {
                self.$fld.duration()
            }
        }
    }
}

pub trait FramedMapper<T, R> {
    fn map<'a>(&'a mut self, input: &'a mut [T]) -> Result<Option<&'a mut [R]>>;

    fn map_frame_size(&self, orig: usize) -> usize {
        orig
    }
}

pub struct FramedMutMapFn<T, F> {
    mapper: F,
    _in_typ: PhantomData<T>,
}

impl<T, F> FramedMapper<T, T> for FramedMutMapFn<T, F>
where
    F: FnMut(&mut T) -> (),
{
    fn map<'a>(&'a mut self, input: &'a mut [T]) -> Result<Option<&'a mut [T]>> {
        input.iter_mut().for_each(&mut self.mapper);
        Ok(Some(input))
    }
}

pub struct FramedMapFn<T, R, F> {
    mapper: F,
    buf: Vec<R>,
    _in_typ: PhantomData<T>,
}

impl<T, R, F> FramedMapper<T, R> for FramedMapFn<T, R, F>
where
    F: Fn(&T) -> R,
{
    fn map<'a>(&'a mut self, input: &'a mut [T]) -> Result<Option<&'a mut [R]>> {
        self.buf.clear();
        let mapper = &self.mapper;
        self.buf.extend(input.iter().map(mapper));
        Ok(Some(self.buf.as_mut_slice()))
    }
}

pub struct MappedFramed<S, M, T, R> {
    source: S,
    mapper: M,
    _src_typ: PhantomData<T>,
    _dst_typ: PhantomData<R>,
}

impl<S, M, T, R> Framed<R> for MappedFramed<S, M, T, R>
where
    S: Framed<T>,
    M: FramedMapper<T, R>,
{
    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [R]>> {
        if let Some(data) = self.source.next_frame()? {
            self.mapper.map(data)
        } else {
            Ok(None)
        }
    }

    fn num_frames(&self) -> usize {
        self.source.num_frames()
    }

    fn num_frames_remain(&self) -> usize {
        self.source.num_frames_remain()
    }

    fn num_full_frames(&self) -> usize {
        self.source.num_full_frames()
    }

    fn full_frame_size(&self) -> usize {
        self.mapper.map_frame_size(self.source.full_frame_size())
    }
}

delegate_impls!(MappedFramed<S, M, T, R>, S, source);

pub struct MappedSamples<S, M, T, R> {
    source: S,
    mapper: M,

    _src_typ: PhantomData<T>,
    _dst_typ: PhantomData<R>,
}

impl<S, M, T, R> MappedSamples<S, M, T, R>
where
    S: Samples<T> + Sampled,
    M: Fn(T) -> R,
{
    pub fn new(source: S, mapper: M) -> Self {
        Self {
            source,
            mapper,
            _src_typ: PhantomData,
            _dst_typ: PhantomData,
        }
    }
}

delegate_impls!(MappedSamples<S, M, T, R>, S, source);

impl<S, M, T, R> Samples<R> for MappedSamples<S, M, T, R>
where
    S: Samples<T> + Sampled,
    M: Fn(T) -> R,
{
    fn seek_samples(&mut self, n: isize) -> Result<()> {
        self.source.seek_samples(n)
    }

    fn next_sample(&mut self) -> Result<Option<R>> {
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

impl<T, R, M> FramedMapper<T, R> for ChanneledMapperWrapper<M, T, R>
where
    M: FramedMapper<Channeled<T>, Channeled<R>>,
    T: Copy,
    R: Copy,
{
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
    FramedMapper<Channeled<T>, Channeled<R>> + Sized
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
    M: FramedMapper<Channeled<T>, Channeled<R>> + Sized
{
}
