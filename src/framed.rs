use crate::channeled::Channeled;
use crate::fraction::Fraction;
use anyhow::Result;
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
            cap,
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
    fn map<'a>(&'a mut self, input: &'a mut [T]) -> Result<Option<&'a mut [R]>>;

    fn map_frame_size(&self, orig: usize) -> usize {
        orig
    }
}

pub trait MapperToChanneled<T: Copy, R: Copy>: Sized + FramedMapper<T, R> {
    type Channeled: FramedMapper<Channeled<T>, Channeled<R>> = ChanneledMapperWrapper<Self, T, R>;

    fn into_channeled(self) -> Self::Channeled;

    fn channeled(self, size: usize) -> ChanneledMapperWrapper<Self, T, R> {
        ChanneledMapperWrapper::new(self, size, size)
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
    cap: usize,
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

impl<T, R, F> MapperToChanneled<T, R> for FramedMapFn<T, R, F>
where
    F: Fn(&T) -> R,
    T: Copy,
    R: Copy,
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

pub struct ChanneledMapperWrapper<M, T, R> {
    mapper: M,
    zip_buf: Vec<Channeled<R>>,
    out_bufs: Option<Channeled<Vec<R>>>,
    in_bufs: Option<Channeled<Vec<T>>>,
    in_size: usize,
    out_size: usize,
}

impl<M, T, R> ChanneledMapperWrapper<M, T, R> {
    pub fn new(mapper: M, in_size: usize, out_size: usize) -> Self {
        Self {
            mapper,
            in_bufs: None,
            out_bufs: None,
            zip_buf: Vec::with_capacity(out_size),
            in_size,
            out_size,
        }
    }
}

impl<M, T, R> FramedMapper<Channeled<T>, Channeled<R>> for ChanneledMapperWrapper<M, T, R>
where
    M: FramedMapper<T, R>,
    T: Clone + Copy,
    R: Clone + Copy,
{
    fn map<'a>(
        &'a mut self,
        input: &'a mut [Channeled<T>],
    ) -> Result<Option<&'a mut [Channeled<R>]>> {
        // initialize buffer for input lazily
        let in_bufs = if let Some(in_bufs) = self.in_bufs.as_mut() {
            in_bufs
        } else {
            if let Some(i0) = input.get(0) {
                let bufs = i0.as_ref().map(|_| Vec::with_capacity(self.in_size));
                self.in_bufs = Some(bufs);
                self.in_bufs.as_mut().unwrap()
            } else {
                return Ok(Some(&mut []));
            }
        };

        // clear the input buffers, and push the input data into them
        in_bufs.as_mut_ref().for_each(move |buf| buf.clear());
        input.iter().for_each(|iv| {
            iv.zip(in_bufs.as_mut_ref())
                .expect("mixed mono/stereo?")
                .for_each(move |(iv, buf)| buf.push(iv))
        });

        let out_bufs = if let Some(out_bufs) = self.out_bufs.as_mut() {
            out_bufs
        } else {
            let out_size = self.out_size;
            let bufs = in_bufs.as_ref().map(|_| Vec::with_capacity(out_size));
            self.out_bufs = Some(bufs);
            self.out_bufs.as_mut().unwrap()
        };

        let mapper = &mut self.mapper;
        let zip_buf = &mut self.zip_buf;
        // go through input data
        in_bufs
            .as_mut_ref()
            // zip with an out buf for each channel
            .zip(out_bufs.as_mut_ref())
            .expect("mix stereo/mono?")
            // for each channel...
            .try_map(|(in_buf, out_buf)| {
                // map the data using the mapper, copy output to the out buf
                // this returns Result<Option<()>> where it's Some(()) if mapper give a value
                // and it's None if mapper did not give a value
                mapper.map(in_buf).map(move |option| {
                    option.map(move |out| {
                        out_buf.clear();
                        out_buf.extend_from_slice(out);
                    })
                })
            })
            .map(move |result| {
                // flatten Channel<Option<()>> to Option<Channeled<()>>
                result
                    .flatten_option()
                    // if the option is present, zip the output buffers, and return them
                    .map(move |_| {
                        zip_buf.clear();
                        zip_buf.extend(out_bufs.as_ref().into_iter().copy_elements());
                        zip_buf.as_mut_slice()
                    })
            })
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
