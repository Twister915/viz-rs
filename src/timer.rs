use crate::delegate_impls;
use crate::framed::Framed;
use crate::util::timed;
use anyhow::Result;
use std::marker::PhantomData;

pub struct FramedTimed<S, T> {
    source: S,
    every_nth: usize,
    counter: usize,

    _in_typ: PhantomData<T>,
}

impl<S, T> FramedTimed<S, T>
where
    S: Framed<T>,
{
    pub fn new(source: S, every_nth: usize) -> Self {
        Self {
            source,
            every_nth,
            counter: 0,
            _in_typ: PhantomData,
        }
    }
}

impl<S, T> Framed<T> for FramedTimed<S, T>
where
    S: Framed<T>,
{
    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [T]>> {
        let result = if self.counter % self.every_nth == 0 {
            let source = &mut self.source;
            let (dur, out) = timed(move || source.next_frame());
            if let Ok(Some(_)) = &out {
                println!("frame computed in {:?}", dur);
            }

            out
        } else {
            self.source.next_frame()
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

delegate_impls!(FramedTimed<S, T>, S, source);
