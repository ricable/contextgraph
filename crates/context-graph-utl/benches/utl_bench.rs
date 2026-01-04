//! UTL benchmark suite.

use criterion::{criterion_group, criterion_main, Criterion};

fn utl_benchmarks(_c: &mut Criterion) {
    // TODO: Add UTL computation benchmarks
}

criterion_group!(benches, utl_benchmarks);
criterion_main!(benches);
