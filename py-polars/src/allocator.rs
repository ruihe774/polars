#[cfg(all(
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
use jemallocator::Jemalloc;
#[cfg(all(
    not(debug_assertions),
    not(allocator = "default"),
    any(not(target_family = "unix"), allocator = "mimalloc"),
))]
use mimalloc::MiMalloc;

use crate::memory::AlignedAllocator;
#[cfg(all(
    debug_assertions,
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
use crate::memory::TracemallocAllocator;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(allocator = "mimalloc"),
    not(allocator = "default"),
    target_family = "unix",
))]
static ALLOC: AlignedAllocator<Jemalloc> = AlignedAllocator::new(Jemalloc);

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(allocator = "default"),
    any(not(target_family = "unix"), allocator = "mimalloc"),
))]
static ALLOC: AlignedAllocator<MiMalloc> = AlignedAllocator::new(MiMalloc);

// On Windows tracemalloc does work. However, we build abi3 wheels, and the
// relevant C APIs are not part of the limited stable CPython API. As a result,
// linking breaks on Windows if we use tracemalloc C APIs. So we only use this
// on Unix for now.
#[global_allocator]
#[cfg(all(
    debug_assertions,
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
static ALLOC: AlignedAllocator<TracemallocAllocator<Jemalloc>> =
    AlignedAllocator::new(TracemallocAllocator::new(Jemalloc));
