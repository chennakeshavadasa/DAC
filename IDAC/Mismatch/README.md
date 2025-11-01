# Mismatches in I-DAC

This section illustrates the statistical behavior of **Differential Non-Linearity (DNL)** and **Integral Non-Linearity (INL)** across multiple Monte Carlo runs of a current-steering DAC (I-DAC).  
The DAC is modeled using a binary-weighted current array composed of unit current sources modeled as:

Iₓ = I₀ + I₀′

where **I₀′**sm represents a **random Gaussian error** term with zero mean and standard deviation **σ**.  
The DAC output currents are then generated using scaled versions of these unit cells — Iₓ, 2Iₓ, 4Iₓ, 8Iₓ .....

<p align="center">
  <img width="80%" alt="IDAC_Mismatch_DNL_INL"
       src="https://github.com/user-attachments/assets/23e48d1f-814b-4e3f-af5b-e91bbfb83065" />
</p>

---

## Differential Non-Linearity (DNL)

In the DNL plot, we observe that the **maximum standard deviation (σ)** occurs approximately in the **mid-code region**.  
This corresponds to transitions involving the **largest current steps**, where multiple current sources switch simultaneously, leading to higher cumulative mismatch. The **DNL &sigma;** is largest around the **mid-code region** because those transitions correspond to large current steps (many unit cells switching simultaneously).  
Typically, the scaling follows:

σ_next ≈ σ_max / √2


That is, the next-largest transition shows roughly **1/√2** of the maximum &sigma;, and this scaling continues for smaller transitions.

This happens because the **DNL errors of adjacent digital codes are correlated**.  
If there are three devices with standard deviations &sigma;₁, &sigma;₂, &sigma;₃ such that &sigma;₁ and &sigma;₂ are correlated, and &sigma;₂ and &sigma;₃ are correlated, then &sigma;₃ is indirectly correlated to &sigma;₁.  

The combined relationship can be approximated as:

σ₃ ≈ σ₁ · σ₂ / √2

Hence, correlation between bit-cell mismatches causes structured variations in DNL &sigma; across codes rather than purely random noise.

---

## Integral Non-Linearity (INL)

**INL** is essentially the cumulative sum of **DNL** errors.  
Even small correlated DNL deviations accumulate, producing a steadily increasing INL deviation with code.  
Thus, **INL &sigma;** often grows with increasing digital code, up to the mid-scale region, after which symmetry may reduce it.

---

## Summary

- Each unit current: `Iₓ = I₀ + I₀′` with Gaussian error (σ = mismatch)
- Mid-code DNL &sigma; is highest due to large current transitions  
- Correlations between bit-cell errors lead to structured, code-dependent DNL patterns  
- INL &sigma; grows with code due to accumulation of correlated DNL errors
