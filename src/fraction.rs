use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::{
    fmt,
    ops::{Add, Div, Mul, Neg, Sub},
};

macro_rules! impl_ops_default {
    ($t: ident) => {
        impl<T> std::ops::Sub<T> for $t
        where
            T: Into<$t>,
        {
            type Output = Self;

            fn sub(self, rhs: T) -> Self::Output {
                self.add(rhs.into().neg())
            }
        }

        impl<T> std::ops::Div<T> for $t
        where
            T: Into<$t>,
        {
            type Output = Self;

            fn div(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                if rhs.is_zero() {
                    panic!("divide by zero");
                }

                self.mul(rhs.reciprocal())
            }
        }

        impl<T, R> std::ops::MulAssign<T> for $t
        where
            Self: std::ops::Mul<T, Output = R>,
            R: Into<$t>,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self = self.mul(rhs).into()
            }
        }

        impl<T, R> std::ops::DivAssign<T> for $t
        where
            Self: std::ops::Div<T, Output = R>,
            R: Into<$t>,
        {
            fn div_assign(&mut self, rhs: T) {
                *self = self.div(rhs).into()
            }
        }

        impl<T, R> std::ops::AddAssign<T> for $t
        where
            Self: std::ops::Add<T, Output = R>,
            R: Into<$t>,
        {
            fn add_assign(&mut self, rhs: T) {
                *self = self.add(rhs).into()
            }
        }

        impl<T, R> std::ops::SubAssign<T> for $t
        where
            Self: std::ops::Sub<T, Output = R>,
            R: Into<$t>,
        {
            fn sub_assign(&mut self, rhs: T) {
                *self = self.sub(rhs).into()
            }
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Fraction {
    numerator: i64,
    denominator: u64,
    simplified: bool,
}

impl Fraction {
    pub fn new(numerator: i64, divisor: i64) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        let sig = numerator.signum() * divisor.signum();
        let numerator = numerator.abs() * sig;
        let divisor = divisor.abs() as u64;
        Some(Self {
            numerator,
            denominator: divisor,
            simplified: false,
        })
    }

    pub fn rounded(&self) -> i64 {
        let (whole, fractional) = self.mixed_number();
        if let Some(fractional) = fractional {
            let half_denom = fractional.denominator / 2;
            if half_denom == 0 {
                whole
            } else {
                let num = fractional.numerator.abs();
                if num < (half_denom as i64) {
                    whole
                } else {
                    whole + fractional.numerator.signum()
                }
            }
        } else {
            whole
        }
    }

    pub fn mixed_number(&self) -> (i64, Option<Fraction>) {
        let denom = self.denominator as i64;
        let num = self.numerator.abs();
        let sign = self.numerator.signum();
        let remainder = num % denom;
        let result = (num / denom) * sign;
        let fractional_part = if remainder != 0 {
            Some(
                Fraction {
                    numerator: remainder * sign,
                    denominator: self.denominator,
                    simplified: false,
                }
                .simplify(),
            )
        } else {
            None
        };

        (result, fractional_part)
    }

    pub fn is_zero(&self) -> bool {
        self.numerator == 0
    }

    pub fn reciprocal(self) -> Self {
        if self.numerator == 0 {
            Self {
                numerator: 0,
                denominator: 1,
                simplified: true,
            }
        } else {
            let numerator_sign = self.numerator.signum();
            Self {
                numerator: (self.denominator as i64) * numerator_sign,
                denominator: (self.numerator.abs()) as u64,
                simplified: self.simplified,
            }
            .simplify()
        }
    }

    pub fn abs(self) -> Self {
        Self {
            numerator: self.numerator.abs(),
            denominator: self.denominator,
            simplified: self.simplified,
        }
    }

    pub fn simplify(mut self) -> Self {
        if self.simplified {
            self
        } else {
            let abs_numerator = self.numerator.abs();
            if self.denominator == 1 || abs_numerator < 2 {
                self.simplified = true;
                self
            } else {
                let gcd = gcd(abs_numerator, self.denominator as i64);
                if gcd == 1 {
                    self.simplified = true;
                    self
                } else {
                    Self {
                        numerator: (abs_numerator / gcd) * self.numerator.signum(),
                        denominator: self.denominator / (gcd as u64),
                        simplified: true,
                    }
                }
            }
        }
    }

    fn make_same_divisor(self, other: Self) -> (Self, Self) {
        if self.denominator == other.denominator {
            (self, other)
        } else if self.numerator == 0 {
            (
                Fraction {
                    numerator: 0,
                    denominator: other.denominator,
                    simplified: true,
                },
                Fraction {
                    numerator: other.numerator,
                    denominator: other.denominator,
                    simplified: other.simplified,
                },
            )
        } else if other.numerator == 0 {
            other.make_same_divisor(self)
        } else {
            let lcm = lcm(self.denominator as i64, other.denominator as i64);
            let lcm_div_self = lcm / (self.denominator as i64);
            let lcm_div_other = lcm / (other.denominator as i64);

            let out_self = Self {
                numerator: self.numerator * lcm_div_self,
                denominator: lcm as u64,
                simplified: false,
            };

            let out_other = Self {
                numerator: other.numerator * lcm_div_other,
                denominator: lcm as u64,
                simplified: false,
            };

            (out_self, out_other)
        }
    }

    pub fn is_whole(&self) -> bool {
        self.denominator == 1
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if a == b {
        a
    } else if a > b {
        gcd(a - b, b)
    } else {
        gcd(a, b - a)
    }
}

fn lcm(a: i64, b: i64) -> i64 {
    if a == 1 {
        b
    } else if b == 1 {
        a
    } else {
        (a * b) / gcd(a, b)
    }
}

impl Default for Fraction {
    fn default() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
            simplified: true,
        }
    }
}

impl Neg for Fraction {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.numerator == 0 {
            self
        } else {
            Self {
                numerator: self.numerator.neg(),
                denominator: self.denominator,
                simplified: self.simplified,
            }
        }
    }
}

impl<T> Mul<T> for Fraction
where
    T: Into<Fraction>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        Self {
            numerator: self.numerator * rhs.numerator,
            denominator: self.denominator * rhs.denominator,
            simplified: false,
        }
        .simplify()
    }
}

impl<T> Add<T> for Fraction
where
    T: Into<Fraction>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        let (a, b) = self.make_same_divisor(rhs);
        Self {
            numerator: a.numerator + b.numerator,
            denominator: a.denominator,
            simplified: false,
        }
        .simplify()
    }
}

impl Into<Fraction> for i64 {
    fn into(self) -> Fraction {
        Fraction {
            numerator: self,
            denominator: 1,
            simplified: true,
        }
    }
}

impl Into<Fraction> for u64 {
    fn into(self) -> Fraction {
        Fraction {
            numerator: self as i64,
            denominator: 1,
            simplified: true,
        }
    }
}

impl Into<Fraction> for i32 {
    fn into(self) -> Fraction {
        (self as i64).into()
    }
}

impl Into<Fraction> for u32 {
    fn into(self) -> Fraction {
        (self as u64).into()
    }
}

impl Into<Fraction> for i16 {
    fn into(self) -> Fraction {
        (self as i64).into()
    }
}

impl Into<Fraction> for u16 {
    fn into(self) -> Fraction {
        (self as u64).into()
    }
}

impl Into<Fraction> for i8 {
    fn into(self) -> Fraction {
        (self as i64).into()
    }
}

impl Into<Fraction> for u8 {
    fn into(self) -> Fraction {
        (self as u64).into()
    }
}

impl Into<f64> for Fraction {
    fn into(self) -> f64 {
        let s = self.simplify();
        (s.numerator as f64) / (s.denominator as f64)
    }
}

impl Into<f32> for Fraction {
    fn into(self) -> f32 {
        let s = self.simplify();
        (s.numerator as f32) / (s.denominator as f32)
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.simplify();
        if s.is_whole() {
            fmt::Display::fmt(&s.numerator, f)
        } else {
            if s.numerator.is_negative() {
                f.write_str("-")?;
            }
            f.write_str("[")?;
            let abs_num = s.numerator.abs();
            fmt::Display::fmt(&abs_num, f)?;
            f.write_str(" / ")?;
            fmt::Display::fmt(&s.denominator, f)?;
            f.write_str("]")?;
            Ok(())
        }
    }
}

impl<T> PartialEq<T> for Fraction
where
    T: Into<Fraction> + Copy,
{
    fn eq(&self, other: &T) -> bool {
        let self_simp = self.simplify();
        let other_simp = (*other).into().simplify();
        self_simp.numerator == other_simp.numerator
            && self_simp.denominator == other_simp.denominator
    }
}

impl Eq for Fraction {}

impl<T> PartialOrd<T> for Fraction
where
    T: Into<Fraction> + Copy,
{
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        let (a, b) = self.make_same_divisor((*other).into());
        a.numerator.partial_cmp(&b.numerator)
    }
}

impl Ord for Fraction {
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b) = self.make_same_divisor(*other);
        a.numerator.cmp(&b.numerator)
    }
}

impl Hash for Fraction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let s = (*self).simplify();
        state.write_i64(s.numerator);
        state.write_u64(s.denominator);
    }
}

impl Mul<f64> for Fraction {
    type Output = f64;

    fn mul(self, rhs: f64) -> Self::Output {
        ((self.numerator as f64) * rhs) / (self.denominator as f64)
    }
}

impl_ops_default!(Fraction);

#[cfg(test)]
pub mod tests {
    use crate::fraction::Fraction;

    #[test]
    fn test_add() {
        let a = Fraction::new(1, 2).unwrap();
        let b = Fraction::new(1, 3).unwrap();
        let sum = a + b;
        assert_eq!(sum, Fraction::new(5, 6).unwrap());
    }

    #[test]
    fn test_mul() {
        let a = Fraction::new(2, 4).unwrap();
        let b = Fraction::new(-2, 1).unwrap();
        let mul_result = a * b;
        assert_eq!(mul_result, Fraction::new(-1, 1).unwrap());
    }
}
