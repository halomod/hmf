# Physical test inventory

This file maps every physical model or quantity in `src/hmf` to the tests that
check it against physics, as `AGENTS.md` ("Testing expectations") requires. It
does not count tests that only exercise code, pin regression data, or rebuild
the formula under test.

A *physical* test compares against one of:

- an analytic solution;
- a limit or special case whose answer is known;
- a number quoted in the defining paper;
- an independent code or method;
- a physically required bound or trend.

**Status values:**

- **added**: new physical tests in `tests/test_physical_*.py`, in the file named.
- **existing**: physical coverage was already adequate.
- **exposes bug**: a physical test fails because of a library bug. The test is
  left out and the bug is reported to the maintainer (see "Bugs found" below).
- **fixed**: a bug that a physical test exposed (or a review found) has been fixed,
  and the fix PR's regression test is physical. The PR and the test are cited.
- **none possible**: no meaningful physical test exists. The reason is given.

"Non-physical" tags existing tests that are regression, coverage or rewrite
tests.

## Mass-function fitting functions (`mass_function/fitting_functions.py`)

New tests are in `test_physical_fits.py` unless stated otherwise.

### Tests that apply to every fit

The PS, SMT, Jenkins, Warren, Reed03/07, Peacock, Angulo, Watson, Crocce,
Courtin, Bhattacharya, Tinker08/10, Behroozi, Pillepich, Manera, Ishiyama,
Bocquet\* and Yung24 fits are all covered by these checks:

- f(σ) is finite and positive.
- An exponential (Gaussian) cutoff at high ν: the log-slope is below −20 at σ = 0.2.
- f(σ) has a single peak in 0.15 < σ < 8. The Bocquet c-variants are excluded
  because their amplitude depends on mass.
- Massive haloes become rarer with redshift.
- The mass fraction is at most 1, over a finite σ range for normalised fits and
  over the common calibration range for unnormalised fits.

The existing `test_allfits` (max f ≤ PS max, single peak on one grid) is
partially physical.

**Status:** added.

### Fit-specific tests

| Model | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| PS | `test_fcoll.py`: ρ(>M)/ρ̄ = erfc(ν/√2) (5%). `test_mgtm`: ρ(>M→0) ≈ ρ̄ (10%). | ∫f dlnν = 1. Closed-form dn/dM, n(>M) and erfc for power-law P(k) (`test_physical_sigma.py`). | added |
| SMT / ST | None. `test_genmf` is a regression against genmf output. | ∫f dlnν = 1. The analytic A reproduces the published 0.3222. SMT(a=1, p=0) ≡ PS. | added |
| Manera | None. | ∫f dlnν = 1 (it uses SMT's normalisation). | added |
| Peacock | `test_fcoll.py`: ρ(>M) vs the analytic Peacock F(>ν) (5%, 40% for wide ranges). | ∫f dlnν = 1. | added |
| Peacock `cutmask` | `test_peacock_cutmask_calibrated_range` (`test_fitting_functions_extra.py`, #359): the mask is True exactly inside Warren+06's calibrated 1e10–1e15 M☉/h, which Peacock (2007) fits. It replaced `test_peacock_cutmask_all_false`, which pinned the always-empty mask. | — | fixed (B6, #359) |
| Tinker10 | None. The existing tests are parameter validation and interpolation. | ∫f dlnν = 1 at z > 0 (computed α) to 1e-6, and with the published z = 0 α (Table 4) to 0.5%. Reduces exactly to SMT for η=0, β=√a, γ=a, φ=p. | added |
| Tinker08 | `test_tinker08_matches_colossus`: independent code. `test_tinker08_native_so_definition_consistency`: 200c ≡ 200m/Ω_m(z). | n(>M) falls with increasing overdensity. | added |
| Watson (SO) | `test_published_fits.py` (#367): eqs 14–16 at z → 0⁺ reproduce Table 2 of the published (v4) paper (A, α, β to 0.01), and at Ω_m = 0.27 the z > 0 fit agrees with the z = 0 fit to the paper's ~10%. `test_watson_gamma_and_fsigma_branches` is coverage. | Γ(Δ=178) = 1 at all z. n(>M) falls with overdensity. | existing + added |
| Watson redshift trend | — | The paper's separate z = 0 fit is not joined to its z > 0 form. At Watson's Ω_m = 0.27 the jump in f(σ) is 0–9%, but because Ω_m(z) multiplies A, α and β in eqs 14–16, it grows to a factor 1.4–2.7 at Ω_m = 0.31 (the default cosmology), which violates the trend between z = 0 and z = 0⁺. This is the paper's model, documented in the `Watson` docstring, not a bug. The trend is tested for z > 0 only. The α/β labelling (B9) is fixed. | added (z > 0) |
| Watson_FoF, Jenkins, Warren, Reed03, Reed07, Angulo, AnguloBound, Crocce, Courtin, Pillepich, Ishiyama | None beyond `test_allfits`. | The checks shared by every fit (above). These fits are not normalised and their papers quote no f(σ) values to compare against. A redshift trend is added for the z-dependent ones (Crocce). | added (generic only) |
| Bhattacharya | `test_bhattacharya_normed_conserves_mass` (∫f dlnν = 1 to 1e-3 at z = 0, 1) and `test_bhattacharya_normed_q1_reduces_to_smt` (A = 0.3222, f ≡ SMT), both in `test_fitting_functions_extra.py` (#359). `test_bhattacharya_normed_uses_norm` only checks finiteness. | — | fixed (B1, #359) |
| Behroozi | `test_behroozi_ngtm` reproduces the paper's Fig. 23. Theta ≥ 1, monotonic, → 1 at z = 0. `test_behroozi_cumulative_is_theta_times_tinker` (#359): n_B(>M) = θ(M) n_T08(>M), the definition in Behroozi+13 App. G, to 5e-4 at z = 1, 4, 8; it fails at z = 4 and 8 if dθ/dM is taken in M☉ while dn/dM is in M☉/h (the h-units bug). (`test_behroozi_theta_matches_closed_form` is a rewrite test.) | Shared checks, including the redshift trend. | existing + added (h units fixed, #359) |
| Bocquet16 (all six) | `test_change_dndm_bocquet` only checks that the dndm and fsigma ratios agree (non-physical). | For the four critical-overdensity fits (200c and 500c, DM-only and Hydro), M500c/M200m and M200c/M200m agree with a direct NFW + Duffy08 solve to 7% (the paper quotes "few percent"), and the ordering 0 < M500c/M200m < M200c/M200m < 1 holds. The 200m fits need no conversion. | added |
| Bocquet16 parameters | `test_bocquet16_defaults_match_published_table` (`test_published_fits.py`, #367): the defaults are Table 2 of the published (v3) paper, typed in from it. `test_bocquet16_mass_conversion_uses_msun` (#367) compares with eqs 6/A2 typed in from the paper, so on its own it is close to a rewrite test. | `test_bocquet_mass_conversion_depends_on_mass_in_msun`: the mass ratio of eqs 6/A2 depends on ln(M/M☉), Ω_m and z only, so the same physical halo gets the same ratio for h = 0.5 and 0.9 at fixed Ω_m (1e-12). With ln M in M☉/h it is off by ~1%. | fixed (B10, #367) + added |
| Yung24 | `test_yung24_*_matches_paper_fit`: digitised Fig. A1 of the paper. | Shared checks, plus a redshift trend within 6 ≤ z ≤ 19. | existing + added |

## Transfer functions (`density_field/transfer_models.py`, `transfer.py`)

New tests are in `test_physical_transfer.py`.

| Model | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| EH_BAO | `test_ehnobao`, `test_bondefs`, `test_bbks_sugiyama` are coverage. | T(k→0) → 1. Agrees with EH_NoBAO at k < 3e-4 h/Mpc and at k > 1 h/Mpc (2%). The Eq. 26 sound horizon agrees with Eq. 6 to 2% (EH98). r_drag is within 5% of Planck 2018. Agrees with CAMB to 6% in P(k). | added |
| EH_NoBAO | None. | T → 1, monotonic, agrees with EH_BAO in both limits. | added |
| BBKS | `test_bbks_sugiyama` (non-physical). | T → 1, monotonic. With zero baryons it matches the EH98 shape to 10%. | added |
| BBKS Liddle baryons | `test_bbks.py` (#367): `test_liddle_equals_sugiyama_at_h_half` (at h = 0.5, √(2h) = 1, so the published and preprint Sugiyama forms coincide, to 1e-12); both corrections vanish as Ω_b → 0; T → 1 at low k. | — | fixed (B11, #367) |
| BondEfs | `test_physical_transfer.py` (#371): T depends only on k/Γ with Γ = Ω_m h (EBW92 eq. 7); matches the EH98 zero-baryon shape to 7% for Ω_m = 0.2–0.4; reproduces BE84's own Ω = 1, h = 0.75 fit (Table 1) to 1%. All failed with the Ω = 0.3 coefficients used before #371. | T → 1, monotonic. | existing + added |
| CAMB | Regression against stored data. Neutrino-species tests are physical: P_tot < P_cb, and they agree for massless neutrinos. CLASS agrees with it to 1% in P(k) (see CLASS). The single-run and copy-safety tests of #363 (`test_default_massfunction_runs_camb_once`, `test_copied_camb_framework_recomputes_correctly`, …, in `test_transfer.py`) are non-physical: they check caching and state, not physics. | EH agreement to 6% in P(k). Each massive species in `m_nu` reaches CAMB (#370, `test_transfer_models.py`): 3 x 0.1 eV matches CAMB run with three massive species (and differs from one 0.3 eV species by >1%), split masses match CAMB's normal hierarchy, and Neff and the mass sum are conserved. | existing + added |
| CLASS (#369) | `test_transfer_class.py` (needs `classy`): CLASS's H(z) and Neff match astropy's (1e-4) and its neutrino density is astropy's; k^n T² has the shape of CLASS's own P(k); agrees with CAMB, an independent code, to 1% in P(k) for ΛCDM, wCDM and 3 massive ν, and in dn/dm; P_tot = P_cb on large scales and (1 − f_ν)² P_cb inside free-streaming; tot ≡ cb for massless ν; T → 1 at low k. | — | existing |
| FromFile / FromArray | — | **None possible:** these are user-supplied tables. FromArray is used with T = 1 to build the self-similar tests. | none possible |
| Transfer: σ8 normalisation | `test_sigma8z`: σ(8) = σ8. `test_sigma_8_species_*`. | σ(8) = σ8 for a power law (`test_physical_sigma.py`). | existing + added |
| Transfer: P(k) | — | P ∝ k^{n_s} on large scales. P(z) = D(z)² P(0). | added |
| Halofit | `test_halofit.py` already has strong physical tests: the low-k linear limit, σ_G(1/k_nl) = 1, k_nl rising with z, the boost falling with z, and agreement with CAMB's halofit. | For a power-law Δ², n_eff = n and C = 0 exactly, and k_nl is analytic. P_nl → P_lin at z = 10 for k < 0.1. | existing + added |

## Growth (`cosmology/growth_factor.py`)

New tests are in `test_physical_growth.py`.

| Model | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| ODEGrowthFactor | Agrees with the integral method and with Heath (independent methods). D ∝ a² in radiation domination, f → 1 in matter domination, monotonic. | D = a and f = 1 exactly in EdS. Linder f = Ω_m^0.55 (1%). CPT92 g₀ (1%). f = dlnD/dlna. Peebles f₀ ≈ Ω_m^0.6 in open universes. | added |
| ODEGrowthFactor for wCDM | `test_growth.py` (#364): `test_growth_matches_independent_ode` (D to 1e-5 and f to 1e-4 against an independent solve with astropy's E(z), for w = −0.8, −1.2 and w0wa, through `ODEGrowthFactor` and `GrowthFactor`); `test_dark_energy_growth_rate_matches_linder` (γ = 0.55 + 0.05(1+w), 1%); `test_dark_energy_growth_matches_camb` (D to 1e-3 against CAMB). | — | fixed (B4, #364) |
| ODEGrowthFactor with massive neutrinos | `test_physical_growth.py` (#368): `test_dlne_dlna_is_log_derivative_of_efunc_with_massive_neutrinos`, `test_massive_neutrino_ode_growth_matches_camb_cb_on_small_scales`, `test_massive_neutrino_growth_matches_momentum_form_ode`. | dlnE/dlna equals a finite difference of astropy's `efunc`. D matches CAMB `delta_nonu` at k/h = 5, below the free-streaming scale, to 2e-4. D and f match an independent momentum-form ODE that needs only E(z), to 2e-5. All failed before the fix: the time dependence of `nu_relative_density` was dropped from dlnE/dlna, so D was 0.2–0.9% low at z = 10. | fixed (#368) |
| GrowthFactor (selector) | Selector tests. Tinker08 within 1% of ODE. Heath in open universes (D, and f since #364: see below). | Linder (with and without radiation). CPT92 g₀. Near-EdS limit (Ω_Λ = 1e-6). | added |
| GrowthFactor in exact EdS | `test_exact_einstein_de_sitter` (`test_growth.py`, #364): D = a and f = 1 to 1e-9 for `GrowthFactor` and `Eisenstein97GrowthFactor` with Ω_m = 1, Ω_Λ = 0. It raised ZeroDivisionError before. | — | fixed (B5, #364) |
| GrowthFactor / Heath77 growth rate in open universes | `test_open_universe_growth_rate` (`test_growth.py`, #364): for Ω_m = 0.1, 0.3, 0.5 with Λ = 0, 0 < f₀ < 1, f₀ = Ω_m^0.6 to 5% (Peebles), and f agrees with the ODE (1e-4) and integral (1e-6) solutions, for `Heath77GrowthFactor`, `GrowthFactor` and `GenMFGrowth`. `test_closed_lambda0_heath_growth_rate`: f > 1 for Ω_m = 1.5. | — | fixed (B7, #364) |
| IntegralGrowthFactor | Agrees with ODE. | EdS, Linder, f = dlnD/dlna (flat and non-flat), Peebles. | added |
| Eisenstein97GrowthFactor | Agrees with ODE. Exact EdS, D = a and f = 1 (#364). | Linder, f = dlnD/dlna, near-EdS limit. | added (EdS exact fixed: B5, #364) |
| Heath77GrowthFactor | Agrees with ODE (D). `test_heath_growth_factor_einstein_de_sitter` (`test_growth.py`) now runs (#364 added the `test_` prefix). Open and closed Λ = 0 growth rate (#364, above). | EdS (D = a, f = 1). | added (rate fixed: B7, #364) |
| GenMFGrowth | Within 5% of ODE (an approximation). Open Λ = 0 growth rate matches Peebles and the ODE (#364, above). | EdS, Linder, f = dlnD/dlna in ΛCDM. | added (rate fixed: B8, #364) |
| Carroll1992 | Within 5% of ODE. | EdS exact. The Lahav growth rate agrees with ODE to 1%. | added |
| CambGrowth | Linder (ΛCDM). Neutrino-species bounds. 3 x 0.1 eV growth matches CAMB with three massive species (#370). `test_dark_energy_growth_matches_camb` (#364): for wCDM and w0wa (w_a now reaches CAMB), its growth rate, which goes through the fixed ODE, equals dlnD/dlna of CAMB's own D to 2e-3. | — | existing |
| ClassGrowth | — | `test_growth_class.py`: D agrees with CAMB (ΛCDM, w = −0.8, and 0.3 eV neutrinos for both species) and with an independent ODE solve to 1e-3; f agrees with the ODE to 1e-3, with Linder's Ω_m(z)^0.55 to 1% and with finite differences of D. D_cb = D_tot for massless neutrinos; with massive ones 1 < D_cb/D_tot ≤ 1/(1 − f_ν), rising with z. | added |
| FromFile / FromArray | — | **None possible:** user-supplied tables. | none possible |
| D(z=0) = 1 | — | **None needed:** this holds by construction (D⁺/D⁺(0)) for every model, so a test would only repeat the code. | none possible |

## Filters, σ(M), ν, n_eff, dn/dm (`density_field/filters.py`, `mass_function/hmf.py`)

New tests are in `test_physical_sigma.py`.

| Quantity | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| TopHat | Analytic σ² and σ₁² for P = k² (k ≤ 1). Analytic dW/dlnx and dlnσ²/dlnR. | White-noise variance = P₀/V (Poisson). Self-similar σ(M) via MassFunction. | existing + added |
| Gaussian | Analytic σ², σ₁² and dlnσ²/dlnR for a power law. | — | existing |
| SharpK | Analytic σ² for P = k². | Analytic σ² = R^{−(n+3)}/[2π²(n+3)] for n = −2.5, −2, −1. | existing + added |
| SmoothK | Analytic power-law σ², σ₁². dlnσ²/dlnR = −(n+3). | — | existing |
| SharpKEllipsoid | Analytic γ and a₃ for a power law. `TestSharpKEllipsoidInMassFunction` (`test_filters.py`, #361), with the physical ν = δ_c²/σ²(R, z): a₃/R = 0.6–0.85 at ν ≈ 1 (Schneider+13 eqs A6–A7, 33); high peaks are spherical (a₃ → R and dn/dm → SharpK at ν > 25); patches are more spherical at z = 4.4 than z = 0. `test_sigma_scale_is_amplitude`: a₃ depends only on the physical amplitude of P. | — | existing (ν fixed, #361) |
| σ(M) | `test_genmf` (regression). | σ = σ8 (R/8)^{−(n+3)/2} for n = −2.5, −2, −1 (1e-3). σ falls with M. | added |
| ν, M_* (`mass_nonlinear`) | `test_nu` (coverage). | M_* = M8(σ8/δc)^{6/(n+3)}. M_* grows with time. | added |
| n_eff | `test_neff_at_collapse` (coverage). | n_eff = n for a power law (1e-3 at n = −2, 2e-2 at n = −1 because of the documented wiggles). | added |
| dn/dm | — | PS closed form for P ∝ k^{−2} (1e-3). | added |
| dn/dm with mass conversion (`disable_mass_conversion=False`) | `test_hmf_mass_definition_consistency`: 200c ≡ 200m/Ω_m(z), now to 1e-8 (#362). `test_mdef.py` (#362): `test_mass_conversion_conserves_cumulative_counts` (n_new(>M) = n_meas(>M_meas(M)) to 5e-3, for 200m and 1600m); `test_mass_conversion_to_same_definition_is_identity`. | — | fixed (B2, #362) |

## Cumulative integrals (`mass_function/integrate_hmf.py`, `ngtm`, `rho_gtm`)

New tests are in `test_physical_sigma.py`.

| Quantity | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| `hmf_integral_gtm` | Analytic integral of a gamma-function-shaped dn/dm (3%). | Analytic integral of a power-law dn/dm, for number and mass (1e-3). | existing + added |
| ngtm | — | PS closed form n(>M) ∝ E₂(ν²/2)/ν² for n = −1 (5e-3, ν < 3). Monotonic. | added |
| rho_gtm | `test_fcoll.py` (5%). | erfc(ν/√2) for power-law spectra (5e-3). Monotonic. 0 < ρ(>M) < ρ̄ for normalised fits. ρ(>M→0) → ρ̄. | added |
| rho_ltm | Indirect (it is ρ̄ − rho_gtm). | — | existing |

## Mass definitions (`halos/mass_definitions.py`)

New tests are in `test_physical_mdef.py`.

| Quantity | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| SOVirial (Bryan & Norman) | None. | Δ = 18π² in EdS. → 18π² at high z in ΛCDM. Δ_c ≈ 100 and Δ_m ≈ 330 for Ω_m = 0.3. Monotonic approach to 18π² from both sides. | added |
| SOMean / SOCritical | — | SOCritical(Δ) ≡ SOMean(Δ/Ω_m(z)). The m ↔ r sphere encloses the halo density. | added |
| FOF | — | **None possible:** ρ_FoF = 9/(2πb³)ρ̄ is an order-of-magnitude definition (White+01). Any check would rewrite the formula. | none possible |
| `change_definition` (NFW) | Agrees with COLOSSUS to 1% (independent code). | Agrees to 1e-6 with a direct NFW enclosed-mass solve written in the test. Agrees with Hu & Kravtsov (2003) to 1%. M500c < M200c < M200m. Round trip. | added |
| `critical_density`, `mean_density` | — | ρ_c,0 = 2.77537e11 h² M☉/Mpc³ (PDG / fundamental constants). ρ̄ = Ω_m ρ_c. ρ̄(z) = ρ̄₀(1+z)³ (`test_physical_transfer.py`). | added |

## WDM (`alternatives/wdm.py`)

New tests are in `test_physical_wdm.py`.

| Quantity | Existing physical coverage | New / proposed physical tests | Status |
|---|---|---|---|
| Viel05 / Bode01 transfer | T(k→0) = 1. | T(2π/λ_hm) = 1/2 (the definition of the half-mode scale). T is monotonic and → 0. | added |
| λ_fs, λ_hm, M_fs, M_hm | Positivity and ordering only. | M_fs and M_hm match Schneider+12 Table 1 (10%). λ_hm/λ_fs = 13.93 (their Eq. 8). All fall with m_x. | added |
| Bode01 (issue #366) | None (it was an alias of Viel05). | λ_hm/2 matches Bode+01's quoted R_s for 175 eV, 350 eV and 1.5 keV (6%). M_hm matches their quoted 4e11 / 4e12 h⁻¹M☉ (20%). λ_hm ∝ m_x^−1.15 (eq. A9). Differs from Viel05: λ_hm ratio 1.088, set by ν = 1.2 vs 1.12. All failed before #372. | fixed (#372) |
| Viel05 break scale | Schneider+12 Table 1. | Eq. 7 second line agrees with its first (m_x/T_x) form, combined with eq. 2 (2%). λ_hm ∝ m_x^−1.11. | added |
| Both: limits | — | T → (αk)^−10 for k ≫ 1/α, for any ν. T → 1 as m_x → ∞ at fixed k. | added |
| `mu` → `nu` rename | — | `mu` still works, with a DeprecationWarning, including through `wdm_params`. Passing both raises. | added |
| M_hm vs redshift | `TestHalfModeMassComoving` in `test_wdm.py` (added with the fix in #358). | M_hm must not depend on z (the transfer function is z-independent and masses are comoving). This failed by a factor (1+z)³ before #358. | fixed (B3, #358) |
| MassFunctionWDM / TransferWDM | dndm agrees with CDM at high M and is suppressed at low M (1e-3). | σ_WDM ≤ σ_CDM, converging at high M. dn/dm suppressed below M_hm and equal to CDM above 1000 M_hm. CDM limit as m_x → ∞. | added |
| Schneider12_vCDM, Schneider12, Lovell14 | `test_high_m` (agreement at high M). | The factor lies in (0, 1], rises monotonically with M, and → 1. | added |

## Bugs found

All of these are now fixed. Each fix PR adds a physical regression test, and each
test cited below was checked to fail with its fix reverted.

| ID | Model | Physical test | Measured vs expected (before the fix) | Fixed by | Regression test |
|---|---|---|---|---|---|
| B1 | `Bhattacharya(normed=True)` | ∫f dlnν = 1. With q = 1, A = 0.3222. | 27.9 vs 1. A = 3.104 = 1/0.3222: `_norm` returned the reciprocal. | #359 | `test_fitting_functions_extra.py::test_bhattacharya_normed_conserves_mass`, `::test_bhattacharya_normed_q1_reduces_to_smt` |
| B2 | `MassFunction.dndm` mass conversion | n_new(>M) = n_meas(>M_meas) | Ratio 0.69–1.02 (→200m) and 0.95–3.08 (→1600m). It evaluated dn/dM at M_new and used M/M_meas instead of the Jacobian. | #362 | `test_mdef.py::test_mass_conversion_conserves_cumulative_counts` |
| B3 | `WDM.m_hm` | M_hm independent of z | ×27 at z = 2, from a (1+z)³ factor in `rho_mean`. | #358 | `test_wdm.py::TestHalfModeMassComoving` |
| B4 | `ODEGrowthFactor` (w ≠ −1) | Linder γ, plus an independent ODE | f was 7% low at w = −0.8 and D 2.4% high. The DE term was missing from `dlne_dlna`. (#368 fixed the analogous massive-neutrino term.) | #364 | `test_growth.py::test_growth_matches_independent_ode`, `::test_dark_energy_growth_rate_matches_linder`, `::test_dark_energy_growth_matches_camb` |
| B5 | `GrowthFactor` / `Eisenstein97GrowthFactor` | EdS: D = a | ZeroDivisionError from (Ω_m/Ω_Λ)^{1/3}. | #364 | `test_growth.py::test_exact_einstein_de_sitter` |
| B6 | `Peacock.cutmask` | Non-empty mask inside the calibrated range | Always False (`m < 1e10 and m > 1e15`). | #359 | `test_fitting_functions_extra.py::test_peacock_cutmask_calibrated_range` |
| B7 | `Heath77GrowthFactor.growth_rate` (and `GrowthFactor` for open Λ = 0) | f₀ ≈ Ω_m^0.6, 0 < f < 1 | f = −0.69, 0.61, 3.55 vs 0.25, 0.49, 0.66 for Ω_m = 0.1, 0.3, 0.5. It inherited the Integral class's rate formula, which assumes a different normalisation. | #364 | `test_growth.py::test_open_universe_growth_rate`, `::test_closed_lambda0_heath_growth_rate` |
| B8 | `GenMFGrowth.growth_rate`, open Λ = 0 | Peebles f₀ ≈ Ω_m^0.6 | NaN, from catastrophic cancellation of the closed form at a ~ 1e-8. | #364 | `test_growth.py::test_open_universe_growth_rate` |
| B9 | Watson SO z > 0 α/β coefficients | Eqs 14–16 at z → 0⁺ give Table 2 (published v4) | α and β were swapped (the arXiv v1 labelling): off by ~0.75. | #367 | `test_published_fits.py::test_watson_redshift_fit_reproduces_table2_at_z0`, `::test_watson_redshift_fit_continuous_with_z0_fit` |
| B10 | Bocquet16 defaults, ln M units | Table 2 of the published (v3) paper; the mass ratio depends on M in M☉ only | The defaults were the arXiv v1 table, and ln M used M☉/h instead of M☉ (0.5–1% in the ratio). | #367 | `test_published_fits.py::test_bocquet16_defaults_match_published_table`; `test_physical_fits.py::test_bocquet_mass_conversion_depends_on_mass_in_msun` |
| B11 | BBKS Liddle baryons | At h = 0.5 (√(2h) = 1) it equals the preprint Sugiyama form | √(Ω_b h) instead of √(2h). | #367 | `test_bbks.py::test_liddle_equals_sugiyama_at_h_half` |

B9–B11 started as review findings without a test; #367 added tests for them.

Other bugs fixed since this inventory was written, with physical tests cited in the
tables above:

- Inside `MassFunction`, SharpKEllipsoid computed its ellipticity from the unnormalised power, not the physical ν = δ_c²/σ²(R, z) (#361).
- Behroozi's dθ/dM was in M☉ while dn/dM is in M☉/h (#359).
- BondEfs used the BE84 Ω = 0.3 coefficients instead of EBW92's Γ = Ω_m h form (#371).
- Bode01 was an alias of Viel05 (issue #366, fixed by #372).
- Massive-neutrino term in `dlne_dlna` (#368); every massive species reaches CAMB (#370).
- CAMB ran more than once per framework, and copies shared CAMB state (#363). Its
  tests check caching and copy-safety, so they are not physical.

Also noted:

- `tests/test_growth.py::test_heath_growth_factor_einstein_de_sitter` now has the
  `test_` prefix and runs (#364).
- Tinker08, Tinker10 and Watson can now be instantiated directly without a
  `mass_definition`: they default to their preferred definition (#359,
  `test_fitting_functions_extra.py::test_generic_so_fits_default_to_preferred_mdef`).
