pub fn two_dimensional_vec<E>(sizes: &Vec<usize>) -> Vec<Vec<E>> {
    sizes
        .iter()
        .copied()
        .map(move |size| Vec::with_capacity(size))
        .collect()
}
