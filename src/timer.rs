use crate::delegate_impls;
use crate::framed::Framed;
use crate::util::timed;
use anyhow::Result;
use std::marker::PhantomData;

pub struct FramedTimed<S, T, I> {
    source: S,
    every_nth: usize,
    counter: usize,

    _in_typ: PhantomData<T>,
    _inner_typ: PhantomData<I>,
}

impl<S, T, I> FramedTimed<S, T, I>
where
    S: Framed<T, I>,
{
    pub fn new(source: S, every_nth: usize) -> Self {
        Self {
            source,
            every_nth,
            counter: 0,
            _in_typ: PhantomData,
            _inner_typ: PhantomData,
        }
    }
}

impl<S, T, I> Framed<T, I> for FramedTimed<S, T, I>
where
    S: Framed<T, I>,
{
    fn into_deep_inner(self) -> I {
        return self.source.into_deep_inner()
    }

    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [T]>> {
        let source = &mut self.source;
        let result = if self.counter % self.every_nth == 0 {
            let (dur, out) = timed(move || source.next_frame());
            if let Ok(Some(_)) = &out {
                println!("frame computed in {:?}", dur);
            }

            out
        } else {
            source.next_frame()
        };

        self.counter += 1;
        result
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
        self.source.full_frame_size()
    }
}

delegate_impls!(FramedTimed<S, T, I>, S, source);
