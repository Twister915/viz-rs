use crate::delegate_impls;
use crate::framed::Framed;
use crate::util::timed;
use anyhow::Result;

pub struct FramedTimed<S> {
    source: S,
    every_nth: usize,
    counter: usize,
}

impl<S> FramedTimed<S>
where
    S: Framed,
{
    pub fn new(source: S, every_nth: usize) -> Self {
        Self {
            source,
            every_nth,
            counter: 0,
        }
    }
}

impl<S> Framed for FramedTimed<S>
where
    S: Framed,
{
    type Item = S::Item;

    fn seek_frame(&mut self, n: isize) -> Result<()> {
        self.source.seek_frame(n)
    }

    fn next_frame(&mut self) -> Result<Option<&mut [Self::Item]>> {
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

    fn full_frame_size(&self) -> usize {
        self.source.full_frame_size()
    }
}

delegate_impls!(FramedTimed<S>, S, source);
