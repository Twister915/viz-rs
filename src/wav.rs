// supports only: PCM, 8 or 16 bits per sample

use crate::channeled::Channeled;
use crate::framed::{AudioSource, Sampled, Samples};
use anyhow::*;
use std::cmp;
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::str::from_utf8;
use crate::util::VizFloat;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleRaw {
    OneByte(u8),
    TwoBytes(i16),
}

impl Default for SampleRaw {
    fn default() -> Self {
        SampleRaw::OneByte(0u8)
    }
}

impl Into<VizFloat> for SampleRaw {
    fn into(self) -> VizFloat {
        use SampleRaw::*;

        match self {
            OneByte(b) => ((b as VizFloat / 255.0) * 2.0) - 1.0,
            TwoBytes(b) => ((b as VizFloat) / 65535.0) * 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrdering {
    LittleEndian,
    BigEndian,
}

impl ByteOrdering {
    fn read_u32<R>(&self, reader: &mut R, buf: &mut [u8]) -> Result<u32>
    where
        R: Read,
    {
        let bytes = self.read_n(reader, buf, 4)?;
        use ByteOrdering::*;
        Ok(match self {
            LittleEndian => u32::from_le_bytes(bytes.try_into().expect("should work")),
            BigEndian => u32::from_be_bytes(bytes.try_into().expect("should work")),
        })
    }

    fn read_u16<R>(&self, reader: &mut R, buf: &mut [u8]) -> Result<u16>
    where
        R: Read,
    {
        let bytes = self.read_n(reader, buf, 2)?;
        use ByteOrdering::*;
        Ok(match self {
            LittleEndian => u16::from_le_bytes(bytes.try_into().expect("should work")),
            BigEndian => u16::from_be_bytes(bytes.try_into().expect("should work")),
        })
    }

    fn i16_from<'a>(&self, buf: &'a [u8]) -> Result<(i16, &'a [u8])> {
        if buf.len() < 2 {
            return Err(anyhow!("EOF"));
        }

        let (data, rest) = buf.split_at(2);
        use ByteOrdering::*;
        let u16 = match self {
            LittleEndian => u16::from_le_bytes(data.try_into().expect("should work")),
            BigEndian => u16::from_be_bytes(data.try_into().expect("should work")),
        };

        Ok((u16 as i16, rest))
    }

    fn read_n<'a, R>(&self, reader: &mut R, buf: &'a mut [u8], n: usize) -> Result<&'a [u8]>
    where
        R: Read,
    {
        if buf.len() < n {
            return Err(anyhow!("buf too small to read data, {} < {}", buf.len(), n));
        }

        let buf = &mut buf[..n];
        reader.read_exact(buf)?;
        Ok(buf)
    }
}

#[derive(Debug)]
pub struct WavFile {
    pub ordering: ByteOrdering,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub bits_per_sample: u16,
    // per channel
    pub num_samples: usize,
    pub block_align: u16,

    f: BufReader<File>,
    data_starts_at: u64,

    sample_at: usize,
}

impl WavFile {
    pub fn open<P>(at: P, buf_size: usize) -> Result<WavFile>
    where
        P: AsRef<Path>,
    {
        let f = File::open(at)?;
        let mut f = BufReader::with_capacity(buf_size, f);
        let mut buf = [0u8; 8];

        let ordering = match read_str_exact(&mut f, &mut buf[..4])? {
            "RIFF" => ByteOrdering::LittleEndian,
            "RIFX" => ByteOrdering::BigEndian,
            other => {
                return Err(anyhow!("invalid chunk id {}", other));
            }
        };
        // skip chunk size
        f.seek(SeekFrom::Current(4))?;
        check_str_tag(&mut f, "WAVE", &mut buf[..])?;
        seek_to_chunk(&mut f, &ordering, "fmt ", &mut buf[..])?;

        match ordering.read_u16(&mut f, &mut buf[..])? {
            0x01 => {}
            other => {
                return Err(anyhow!("not PCM audio data, got format id {}", other));
            }
        }

        let num_channels = ordering.read_u16(&mut f, &mut buf[..])?;
        let sample_rate = ordering.read_u32(&mut f, &mut buf[..])?;
        let _ = ordering.read_u32(&mut f, &mut buf[..])?;
        let block_align = ordering.read_u16(&mut f, &mut buf[..])?;
        let bits_per_sample = ordering.read_u16(&mut f, &mut buf[..])?;

        let len = seek_to_chunk(&mut f, &ordering, "data", &mut buf[..])?;
        let num_samples = len / (block_align as usize);
        let data_starts_at = f.seek(SeekFrom::Current(0))?;

        Ok(Self {
            ordering,
            sample_rate,
            num_channels,
            bits_per_sample,
            num_samples,
            block_align,
            f,
            data_starts_at,
            sample_at: 0,
        })
    }

    fn read_one_channel_sample(&mut self) -> Result<SampleRaw> {
        match self.bits_per_sample {
            8 => {
                let mut buf = [0u8; 1];
                self.f.read_exact(&mut buf[..])?;
                let sample = SampleRaw::OneByte(buf[0]);
                Ok(sample)
            }
            16 => {
                let mut buf = [0u8; 2];
                self.f.read_exact(&mut buf[..])?;
                let (raw_sample, _) = self.ordering.i16_from(&buf[..2])?;
                let sample = SampleRaw::TwoBytes(raw_sample);
                Ok(sample)
            }
            other => {
                return Err(anyhow!(
                    "bits per sample must be 8 or 16, no support for other formats (got {})!",
                    other
                ));
            }
        }
    }

    fn does_sample_exist(&self, sample: isize) -> bool {
        sample >= 0 && sample < (self.num_samples() as isize)
    }
}

impl Samples<Channeled<SampleRaw>, WavFile> for WavFile {
    fn into_deep_inner(self) -> WavFile {
        self
    }

    fn seek_samples(&mut self, n: isize) -> Result<(), Error> {
        let new_sample_at = (self.sample_at as isize) + n;
        if self.does_sample_exist(new_sample_at) {
            let byte_offset = (n * (self.block_align as isize)) as i64;
            self.f.seek(SeekFrom::Current(byte_offset))?;
            self.sample_at = new_sample_at as usize;
        }

        Ok(())
    }

    fn next_sample(&mut self) -> Result<Option<Channeled<SampleRaw>>, Error> {
        if !self.has_more_samples() {
            return Ok(None);
        }

        let out = match self.num_channels {
            1 => Channeled::Mono(self.read_one_channel_sample()?),
            2 => Channeled::Stereo(
                self.read_one_channel_sample()?,
                self.read_one_channel_sample()?,
            ),
            other => {
                return Err(anyhow!("bad number of channels (unsupported): {}", other));
            }
        };

        self.sample_at += 1;

        Ok(Some(out))
    }

    fn num_samples_remain(&self) -> usize {
        self.num_samples - self.sample_at
    }
}

impl Sampled for WavFile {
    fn sample_rate(&self) -> usize {
        self.sample_rate as usize
    }

    fn num_samples(&self) -> usize {
        self.num_samples as usize
    }
}

impl AudioSource for WavFile {
    fn num_channels(&self) -> usize {
        self.num_channels as usize
    }
}

fn seek_to_chunk<R>(
    reader: &mut R,
    ordering: &ByteOrdering,
    id: &str,
    buf: &mut [u8],
) -> Result<usize>
where
    R: Read + Seek,
{
    loop {
        let chunk_id = read_str_exact(reader, &mut buf[..id.len()])?;
        if chunk_id == id {
            return Ok(ordering.read_u32(reader, &mut buf[..])? as usize);
        }

        let seek_by = SeekFrom::Current((ordering.read_u32(reader, &mut buf[..])?) as i64);
        reader.seek(seek_by)?;
    }
}

fn check_str_tag<R>(reader: &mut R, tag: &str, buf: &mut [u8]) -> Result<()>
where
    R: Read,
{
    let mut unseen_tag_bytes = tag.as_bytes();
    while unseen_tag_bytes.len() > 0 {
        let max_n_read = cmp::min(buf.len(), unseen_tag_bytes.len());
        let buf = &mut buf[..max_n_read];

        let n_read = reader.read(buf)?;
        if n_read == 0 {
            return Err(anyhow!("eof while looking for tag {}", tag));
        }

        for i in 0..n_read {
            if buf[i] != unseen_tag_bytes[i] {
                return Err(anyhow!("did not find tag {}", tag));
            }
        }

        unseen_tag_bytes = &unseen_tag_bytes[n_read..];
    }

    Ok(())
}

fn read_str_exact<'a, R>(reader: &mut R, buf: &'a mut [u8]) -> Result<&'a str>
where
    R: Read,
{
    reader.read_exact(buf)?;
    Ok(from_utf8(buf)?)
}

#[cfg(test)]
pub mod tests {
    use crate::framed::{Sampled, Samples};
    use crate::wav::WavFile;

    #[test]
    fn open_wav_file() {
        let mut file = WavFile::open("skyline.wav", 8192).expect("should open");

        println!("file {:?}", file);
        println!("dur = {:?}", file.duration());
        loop {
            match file.next_sample() {
                Ok(opt) => match opt {
                    Some(c) => {
                        println!("got {:?}", c);
                    }
                    None => break,
                },
                Err(err) => {
                    panic!("failed while reading file {:?}", err);
                }
            }
        }
        println!("done!");
    }
}
