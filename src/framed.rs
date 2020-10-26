use crate::channeled::Channeled;
use crate::fraction::Fraction;
use anyhow::{anyhow, Result};
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

    fn next_frame(&mut self) -> Result<Option<&[E]>>;

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
            cap,
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
        ((Fraction::new(self.sample_rate() as i64, 1_000_000_000).unwrap())
            * (dur.as_nanos() as i64))
            .rounded() as usize
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
    fn map(&mut self, input: &[T]) -> Result<Option<&[R]>>;

    fn map_frame_size(&self, orig: usize) -> usize {
        orig
    }
}

pub trait MapperToChanneled<T, R>: Sized {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, T, R>;

    fn channeled(self, size: usize) -> ChanneledMapperWrapper<Self, T, R> {
        ChanneledMapperWrapper::new(self, size, size)
    }
}

pub struct FramedMapFn<T, R, F> {
    mapper: F,
    buf: Vec<R>,
    cap: usize,
    _in_typ: PhantomData<T>,
}

impl<T, R, F> FramedMapper<T, R> for FramedMapFn<T, R, F>
where
    F: Fn(&T) -> R,
{
    fn map(&mut self, input: &[T]) -> Result<Option<&[R]>> {
        self.buf.clear();
        let mapper = &self.mapper;
        for elem in input {
            self.buf.push(mapper(elem));
        }

        Ok(Some(self.buf.as_slice()))
    }
}

impl<T, R, F> MapperToChanneled<T, R> for FramedMapFn<T, R, F>
where
    F: Fn(&T) -> R,
{
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, T, R> {
        let cap = self.cap;
        self.channeled(cap)
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

    fn next_frame(&mut self) -> Result<Option<&[R]>> {
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

pub struct ChanneledMapperWrapper<M, T, R> {
    mapper: M,
    in_bufs: ChanneledBufs<T>,
    out_bufs: ChanneledBufs<R>,
    out: Vec<Channeled<R>>,
}

impl<M, T, R> ChanneledMapperWrapper<M, T, R> {
    pub fn new(mapper: M, in_size: usize, out_size: usize) -> Self {
        Self {
            mapper,
            in_bufs: ChanneledBufs::new(in_size),
            out_bufs: ChanneledBufs::new(out_size),
            out: Vec::with_capacity(out_size),
        }
    }
}

struct ChanneledBufs<T> {
    primary: Vec<T>,
    secondary: Option<Vec<T>>,
    size: usize,
}

impl<T> ChanneledBufs<T> {
    fn new(size: usize) -> Self {
        Self {
            primary: Vec::with_capacity(size),
            secondary: None,
            size,
        }
    }

    fn clear(&mut self) {
        self.primary.clear();
        if let Some(secondary) = self.secondary.as_mut() {
            secondary.clear();
        }
    }

    fn primary(&mut self) -> &mut Vec<T> {
        &mut self.primary
    }

    fn secondary(&mut self) -> &mut Vec<T> {
        let secondary = &mut self.secondary;
        if secondary.is_none() {
            *secondary = Some(Vec::with_capacity(self.size));
        }
        secondary.as_mut().unwrap()
    }
}

impl<M, T, R> FramedMapper<Channeled<T>, Channeled<R>> for ChanneledMapperWrapper<M, T, R>
where
    M: FramedMapper<T, R>,
    T: Clone,
    R: Clone,
{
    fn map(&mut self, input: &[Channeled<T>]) -> Result<Option<&[Channeled<R>]>> {
        use Channeled::*;

        self.in_bufs.clear();
        let mut is_mono = false;
        let mut is_stereo = false;
        for entry in input {
            match entry {
                Stereo(a, b) => {
                    if is_mono {
                        return Err(anyhow!("inconsistent mono/stereo data"));
                    }

                    is_stereo = true;
                    self.in_bufs.primary().push(a.clone());
                    self.in_bufs.secondary().push(b.clone());
                }
                Mono(v) => {
                    if is_stereo {
                        return Err(anyhow!("inconsistent mono/stereo data"));
                    }

                    is_mono = true;
                    self.in_bufs.primary().push(v.clone());
                }
            }
        }

        self.out.clear();
        self.out_bufs.clear();
        if let Some(result) = self.mapper.map(self.in_bufs.primary.as_slice())? {
            self.out_bufs.primary().extend_from_slice(result);
        } else {
            return Ok(None);
        }

        if is_stereo {
            let secondary = self
                .in_bufs
                .secondary
                .as_ref()
                .expect("it's there")
                .as_slice();
            let count = if let Some(result) = self.mapper.map(secondary)? {
                let a_count = self.out_bufs.primary.len();
                let b_count = result.len();
                if a_count != b_count {
                    return Err(anyhow!(
                        "left chan had {} outs but right had {} outs",
                        a_count,
                        b_count
                    ));
                }

                self.out_bufs.secondary().extend_from_slice(result);
                b_count
            } else {
                return Ok(None);
            };

            let primary = &mut self.out_bufs.primary;
            let secondary = self.out_bufs.secondary.as_mut().unwrap();

            for i in (0..count).rev() {
                self.out
                    .push(Stereo(primary.remove(i), secondary.remove(i)));
            }
        } else {
            let primary = &mut self.out_bufs.primary;
            let count = primary.len();
            for i in (0..count).rev() {
                self.out.push(Mono(primary.remove(i)));
            }
        }

        self.out.reverse();
        Ok(Some(self.out.as_slice()))
    }

    fn map_frame_size(&self, orig: usize) -> usize {
        self.mapper.map_frame_size(orig)
    }
}

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
