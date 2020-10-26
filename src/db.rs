use crate::framed::{ChanneledMapperWrapper, FramedMapper, MapperToChanneled};
use anyhow::Result;

pub struct DbMapper {
    buf: Vec<f64>,
    cap: usize,
}

impl DbMapper {
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            cap,
        }
    }
}

impl FramedMapper<f64, f64> for DbMapper {
    fn map(&mut self, input: &[f64]) -> Result<Option<&[f64]>> {
        self.buf.clear();
        for elem in input {
            self.buf.push(20.0 * (*elem).log10());
        }

        Ok(Some(self.buf.as_slice()))
    }
}

impl MapperToChanneled<f64, f64> for DbMapper {
    fn into_channeled(self) -> ChanneledMapperWrapper<Self, f64, f64> {
        let cap = self.cap;
        self.channeled(cap)
    }
}
