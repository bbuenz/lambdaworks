use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};

use super::prover::ProvingError;
use super::traits::AIR;

/// The evaluation domains of a STARK: the trace domain (the `trace_length`-th roots of
/// unity) and the LDE domain (a coset of the `trace_length * blowup_factor`-th roots of
/// unity, shifted by `coset_offset`).
///
/// The prover materialises the LDE domain ([`new_domain`] / [`Domain::new`]) because it
/// evaluates polynomials on every point of it. The verifier only ever needs a handful of points
/// and two membership tests, so [`new_verifier_domain`] leaves the vectors empty and
/// [`Domain::lde_point`], [`Domain::is_in_lde_coset`] and [`Domain::is_in_trace_domain`]
/// compute what is needed by exponentiation. This keeps the verifier polylogarithmic in
/// the trace length instead of paying `5 * trace_length` field multiplications up front.
pub struct Domain<F: IsFFTField> {
    pub(crate) root_order: u32,
    /// Points of the LDE coset in natural order; empty for a verifier domain.
    pub(crate) lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    pub(crate) trace_primitive_root: FieldElement<F>,
    pub(crate) coset_offset: FieldElement<F>,
    pub(crate) blowup_factor: usize,
    pub(crate) interpolation_domain_size: usize,
    /// Generator of the LDE domain, a primitive `lde_domain_size`-th root of unity.
    pub(crate) lde_primitive_root: FieldElement<F>,
    /// `trace_length * blowup_factor`.
    pub(crate) lde_domain_size: usize,
}

/// Everything about a domain that does not require materialising it.
struct DomainScalars<F: IsFFTField> {
    root_order: u32,
    trace_primitive_root: FieldElement<F>,
    coset_offset: FieldElement<F>,
    blowup_factor: usize,
    interpolation_domain_size: usize,
    lde_primitive_root: FieldElement<F>,
    lde_domain_size: usize,
}

fn domain_scalars<F, A>(air: &A) -> Result<DomainScalars<F>, ProvingError>
where
    F: IsFFTField,
    A: AIR<Field = F> + ?Sized,
{
    let blowup_factor = air.options().blowup_factor as usize;
    let coset_offset = FieldElement::from(air.options().coset_offset);
    let trace_length = air.trace_length();

    // The trace length must be a power of two (required for FFT).
    if trace_length == 0 || !trace_length.is_power_of_two() {
        return Err(ProvingError::InvalidTraceLength(trace_length));
    }

    let root_order = trace_length.trailing_zeros();
    let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64)
        .map_err(|_| ProvingError::PrimitiveRootNotFound(root_order as u64))?;

    let lde_domain_size = trace_length * blowup_factor;
    let lde_root_order = lde_domain_size.trailing_zeros();
    let lde_primitive_root = F::get_primitive_root_of_unity(lde_root_order as u64)
        .map_err(|_| ProvingError::PrimitiveRootNotFound(lde_root_order as u64))?;

    Ok(DomainScalars {
        root_order,
        trace_primitive_root,
        coset_offset,
        blowup_factor,
        interpolation_domain_size: trace_length,
        lde_primitive_root,
        lde_domain_size,
    })
}

impl<F: IsFFTField> Domain<F> {
    /// Creates a fully materialised domain from an AIR.
    ///
    /// # Panics
    /// Panics if the trace length is not a positive power of two or if the roots of unity
    /// cannot be generated. For fallible construction, use [`new_domain`].
    pub fn new<A>(air: &A) -> Self
    where
        A: AIR<Field = F>,
    {
        Self::eager(air)
            .expect("trace_length must be a positive power of two and roots of unity must exist")
    }

    fn from_scalars(
        scalars: DomainScalars<F>,
        lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    ) -> Self {
        Self {
            root_order: scalars.root_order,
            lde_roots_of_unity_coset,
            trace_primitive_root: scalars.trace_primitive_root,
            coset_offset: scalars.coset_offset,
            blowup_factor: scalars.blowup_factor,
            interpolation_domain_size: scalars.interpolation_domain_size,
            lde_primitive_root: scalars.lde_primitive_root,
            lde_domain_size: scalars.lde_domain_size,
        }
    }

    fn eager<A>(air: &A) -> Result<Self, ProvingError>
    where
        A: AIR<Field = F> + ?Sized,
    {
        let scalars = domain_scalars(air)?;
        let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
            scalars.lde_domain_size.trailing_zeros() as u64,
            scalars.lde_domain_size,
            &scalars.coset_offset,
        )?;
        Ok(Self::from_scalars(scalars, lde_roots_of_unity_coset))
    }

    fn lazy<A>(air: &A) -> Result<Self, ProvingError>
    where
        A: AIR<Field = F> + ?Sized,
    {
        let scalars = domain_scalars(air)?;
        Ok(Self::from_scalars(scalars, Vec::new()))
    }

    /// Number of points of the LDE domain, `trace_length * blowup_factor`.
    pub fn lde_domain_size(&self) -> usize {
        self.lde_domain_size
    }

    /// The `index`-th point of the LDE coset in natural order, `coset_offset * g^index`.
    /// Reads the materialised vector when present and exponentiates otherwise.
    pub fn lde_point(&self, index: usize) -> FieldElement<F> {
        match self.lde_roots_of_unity_coset.get(index) {
            Some(point) => point.clone(),
            None => &self.coset_offset * self.lde_primitive_root.pow(index),
        }
    }

    /// Whether `z` lies in the LDE coset: `z = coset_offset * u` with `u^n = 1` for
    /// `n = lde_domain_size`, i.e. `z^n = coset_offset^n`.
    pub fn is_in_lde_coset<E>(&self, z: &FieldElement<E>) -> bool
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        z.pow(self.lde_domain_size)
            == self
                .coset_offset
                .pow(self.lde_domain_size)
                .to_extension::<E>()
    }

    /// Whether `z` is a `trace_length`-th root of unity.
    pub fn is_in_trace_domain<E>(&self, z: &FieldElement<E>) -> bool
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        z.pow(self.interpolation_domain_size) == FieldElement::<E>::one()
    }

    /// Samples the out-of-domain point `z` from the transcript, rejecting points of the
    /// LDE coset and of the trace domain. Equivalent to
    /// `IsStarkTranscript::sample_z_ood` over the materialised domains, since the two
    /// membership tests characterise those sets exactly, but costs two exponentiations
    /// per candidate instead of a scan over `5 * trace_length` points.
    pub fn sample_z_ood<E>(&self, transcript: &mut impl IsStarkTranscript<E, F>) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        loop {
            let z = transcript.sample_field_element();
            if !self.is_in_lde_coset(&z) && !self.is_in_trace_domain(&z) {
                return z;
            }
        }
    }
}

/// Creates a fully materialised domain from an AIR, for the prover.
pub fn new_domain<Field, FieldExtension, PI>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
) -> Result<Domain<Field>, ProvingError>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    Domain::eager(air)
}

/// Creates a domain that materialises nothing, for the verifier: points are computed on
/// demand with [`Domain::lde_point`] and membership is tested by exponentiation.
pub fn new_verifier_domain<Field, FieldExtension, PI>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
) -> Result<Domain<Field>, ProvingError>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    Domain::lazy(air)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::simple_fibonacci::{FibonacciAIR, FibonacciPublicInputs};
    use crate::proof::options::ProofOptions;
    use crate::transcript::StoneProverTranscript;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    type FE = FieldElement<Stark252PrimeField>;

    fn domains() -> (Domain<Stark252PrimeField>, Domain<Stark252PrimeField>) {
        let mut options = ProofOptions::default_test_options();
        options.blowup_factor = 4;
        options.coset_offset = 3;
        let pub_inputs = FibonacciPublicInputs {
            a0: FE::one(),
            a1: FE::one(),
        };
        let air = FibonacciAIR::<Stark252PrimeField>::new(16, &pub_inputs, &options);
        (
            new_domain(&air).unwrap(),
            new_verifier_domain(&air).unwrap(),
        )
    }

    /// The trace roots of unity, which no longer live in the domain.
    fn trace_roots(domain: &Domain<Stark252PrimeField>) -> Vec<FE> {
        get_powers_of_primitive_root_coset(
            domain.root_order as u64,
            domain.interpolation_domain_size,
            &FE::one(),
        )
        .unwrap()
    }

    #[test]
    fn verifier_domain_points_match_the_materialised_coset() {
        let (eager, lazy) = domains();
        assert!(lazy.lde_roots_of_unity_coset.is_empty());
        assert_eq!(lazy.lde_domain_size(), eager.lde_roots_of_unity_coset.len());
        assert_eq!(lazy.lde_domain_size(), 64);
        for (i, point) in eager.lde_roots_of_unity_coset.iter().enumerate() {
            assert_eq!(&lazy.lde_point(i), point);
            assert_eq!(&eager.lde_point(i), point);
        }
    }

    #[test]
    fn membership_tests_characterise_the_domains() {
        let (eager, lazy) = domains();
        for point in &eager.lde_roots_of_unity_coset {
            assert!(lazy.is_in_lde_coset(point));
            assert!(!lazy.is_in_trace_domain(point));
        }
        for point in &trace_roots(&eager) {
            assert!(lazy.is_in_trace_domain(point));
            assert!(!lazy.is_in_lde_coset(point));
        }
        let outside = FE::from(12345u64);
        assert!(!lazy.is_in_lde_coset(&outside));
        assert!(!lazy.is_in_trace_domain(&outside));
    }

    #[test]
    fn sampling_agrees_with_the_scan_based_sampler() {
        let (eager, lazy) = domains();
        let mut scan_transcript = StoneProverTranscript::new(b"z");
        let mut fast_transcript = StoneProverTranscript::new(b"z");
        let scanned =
            scan_transcript.sample_z_ood(&eager.lde_roots_of_unity_coset, &trace_roots(&eager));
        let fast = lazy.sample_z_ood(&mut fast_transcript);
        assert_eq!(scanned, fast);
    }
}
