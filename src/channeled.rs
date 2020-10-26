use std::fmt;

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
    pub fn map<'a, F, R>(&'a self, f: F) -> Channeled<R>
    where
        F: Fn(&'a T) -> R,
    {
        use Channeled::*;
        match self {
            Stereo(a, b) => Stereo(f(a), f(b)),
            Mono(a) => Mono(f(a)),
        }
    }
}
