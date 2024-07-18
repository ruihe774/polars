use arrow::array::{PrimitiveArray as PArr, StaticArray};
use arrow::compute::utils::{combine_validities_and, combine_validities_and3};
use strength_reduce::*;

use super::PrimitiveArithmeticKernelImpl;
use crate::arity::{prim_binary_values, prim_unary_values, StoreIntrinsic};
use crate::comparisons::TotalEqKernel;

macro_rules! impl_unsigned_arith_kernel {
    ($T:ty, $StrRed:ty) => {
        impl PrimitiveArithmeticKernelImpl for $T {
            type TrueDivT = f64;

            fn prim_wrapping_abs<S: StoreIntrinsic>(lhs: PArr<$T>, _: S) -> PArr<$T> {
                lhs
            }

            fn prim_wrapping_neg<S: StoreIntrinsic>(lhs: PArr<$T>, s: S) -> PArr<$T> {
                prim_unary_values(lhs, |x| x.wrapping_neg(), s)
            }

            fn prim_wrapping_add<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                other: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, other, |a, b| a.wrapping_add(b), s)
            }

            fn prim_wrapping_sub<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                other: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, other, |a, b| a.wrapping_sub(b), s)
            }

            fn prim_wrapping_mul<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                other: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_binary_values(lhs, other, |a, b| a.wrapping_mul(b), s)
            }

            fn prim_wrapping_floor_div<S: StoreIntrinsic>(
                mut lhs: PArr<$T>,
                mut other: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                let mask = other.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and3(
                    lhs.take_validity().as_ref(),   // Take validity so we don't
                    other.take_validity().as_ref(), // compute combination twice.
                    Some(&mask),
                );
                let ret = prim_binary_values(lhs, other, |a, b| if b != 0 { a / b } else { 0 }, s);
                ret.with_validity(valid)
            }

            fn prim_wrapping_trunc_div<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                Self::prim_wrapping_floor_div(lhs, rhs, s)
            }

            fn prim_wrapping_mod<S: StoreIntrinsic>(
                mut lhs: PArr<$T>,
                mut other: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                let mask = other.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and3(
                    lhs.take_validity().as_ref(),   // Take validity so we don't
                    other.take_validity().as_ref(), // compute combination twice.
                    Some(&mask),
                );
                let ret = prim_binary_values(lhs, other, |a, b| if b != 0 { a % b } else { 0 }, s);
                ret.with_validity(valid)
            }

            fn prim_wrapping_add_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                prim_unary_values(lhs, |x| x.wrapping_add(rhs), s)
            }

            fn prim_wrapping_sub_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                Self::prim_wrapping_add_scalar(lhs, rhs.wrapping_neg(), s)
            }

            fn prim_wrapping_sub_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                prim_unary_values(rhs, |x| lhs.wrapping_sub(x), s)
            }

            fn prim_wrapping_mul_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                if rhs == 0 {
                    lhs.fill_with(0)
                } else if rhs == 1 {
                    lhs
                } else if rhs & (rhs - 1) == 0 {
                    // Power of two.
                    let shift = rhs.trailing_zeros();
                    prim_unary_values(lhs, |x| x << shift, s)
                } else {
                    prim_unary_values(lhs, |x| x.wrapping_mul(rhs), s)
                }
            }

            fn prim_wrapping_floor_div_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                if rhs == 0 {
                    PArr::full_null(lhs.len(), lhs.data_type().clone())
                } else if rhs == 1 {
                    lhs
                } else {
                    let red = <$StrRed>::new(rhs);
                    prim_unary_values(lhs, |x| x / red, s)
                }
            }

            fn prim_wrapping_floor_div_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                if lhs == 0 {
                    return rhs.fill_with(0);
                }

                let mask = rhs.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and(rhs.validity(), Some(&mask));
                let ret = prim_unary_values(rhs, |x| if x != 0 { lhs / x } else { 0 }, s);
                ret.with_validity(valid)
            }

            fn prim_wrapping_trunc_div_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                Self::prim_wrapping_floor_div_scalar(lhs, rhs, s)
            }

            fn prim_wrapping_trunc_div_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                Self::prim_wrapping_floor_div_scalar_lhs(lhs, rhs, s)
            }

            fn prim_wrapping_mod_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<$T> {
                if rhs == 0 {
                    PArr::full_null(lhs.len(), lhs.data_type().clone())
                } else if rhs == 1 {
                    lhs.fill_with(0)
                } else {
                    let red = <$StrRed>::new(rhs);
                    prim_unary_values(lhs, |x| x % red, s)
                }
            }

            fn prim_wrapping_mod_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<$T> {
                if lhs == 0 {
                    return rhs.fill_with(0);
                }

                let mask = rhs.tot_ne_kernel_broadcast(&0);
                let valid = combine_validities_and(rhs.validity(), Some(&mask));
                let ret = prim_unary_values(rhs, |x| if x != 0 { lhs % x } else { 0 }, s);
                ret.with_validity(valid)
            }

            fn prim_true_div<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                other: PArr<$T>,
                s: S,
            ) -> PArr<Self::TrueDivT> {
                prim_binary_values(lhs, other, |a, b| a as f64 / b as f64, s)
            }

            fn prim_true_div_scalar<S: StoreIntrinsic>(
                lhs: PArr<$T>,
                rhs: $T,
                s: S,
            ) -> PArr<Self::TrueDivT> {
                let inv = 1.0 / rhs as f64;
                prim_unary_values(lhs, |x| x as f64 * inv, s)
            }

            fn prim_true_div_scalar_lhs<S: StoreIntrinsic>(
                lhs: $T,
                rhs: PArr<$T>,
                s: S,
            ) -> PArr<Self::TrueDivT> {
                prim_unary_values(rhs, |x| lhs as f64 / x as f64, s)
            }
        }
    };
}

impl_unsigned_arith_kernel!(u8, StrengthReducedU8);
impl_unsigned_arith_kernel!(u16, StrengthReducedU16);
impl_unsigned_arith_kernel!(u32, StrengthReducedU32);
impl_unsigned_arith_kernel!(u64, StrengthReducedU64);
impl_unsigned_arith_kernel!(u128, StrengthReducedU128);
