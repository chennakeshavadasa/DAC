# Mismatches in I-DAC

This section illustrates the statistical behavior of **Differential Non-Linearity (DNL)** and **Integral Non-Linearity (INL)** across multiple Monte Carlo runs of a current-steering DAC (I-DAC).  
The DAC is modeled using a binary-weighted current array composed of unit current sources modeled as:

\[
I_x = I_0 + I_0'
\]

where \( I_0' \) represents a **random Gaussian error** term with zero mean and standard deviation \( \sigma \).  
The DAC output currents are then generated using scaled versions of these unit cells — \( I_x, 2I_x, 4I_x, 8I_x, \dots \).

<p align="center">
  <img width="80%" alt="IDAC_Mismatch_DNL_INL"
       src="https://github.com/user-attachments/assets/23e48d1f-814b-4e3f-af5b-e91bbfb83065" />
</p>

---

## Differential Non-Linearity (DNL)

In the DNL plot, we observe that the **maximum standard deviation (σ)** occurs approximately in the **mid-code region**.  
This corresponds to transitions involving the **largest current steps**, where multiple current sources switch simultaneously, leading to higher cumulative mismatch.

The next largest DNL standard deviation is roughly:

\[
\sigma_{\text{next}} = \frac{\sigma_{\text{max}}}{\sqrt{2}}
\]

and this scaling trend continues for subsequent transitions.  
This behavior arises because the **mismatch errors of different current sources are correlated** — transitions involving shared or related devices contribute correlated errors.

To visualize this conceptually, suppose three current sources have mismatch sigmas \( \sigma_1, \sigma_2, \sigma_3 \), and they are pairwise correlated:

\[
\sigma_{12} \propto \sigma_1 \sigma_2, \quad
\sigma_{23} \propto \sigma_2 \sigma_3
\]

Then, due to transitive correlation among them:

\[
\sigma_3 \propto \sigma_1 \sigma_2 \sqrt{2}
\]

Thus, the DNL variation across codes is not independent — it reflects the **underlying correlation structure** of the unit current sources.

---

## Integral Non-Linearity (INL)

INL, being the cumulative sum of DNL errors, inherits these correlation effects.  
Hence, even though each DNL step has small variations, **their accumulated effect manifests as larger deviations in INL**.  
The INL σ tends to increase gradually with code until it saturates, reflecting how random but correlated DNL errors accumulate over successive transitions.
