# Supervisor Feedback on Proposal v1

Date: 2026-06

## Context

This feedback was provided after reviewing:

- `docs/proposal/v1/Proposal_XAI_CXR_Trustworthiness_ArianAbbasi_v1.docx`
- `docs/proposal/v1/Proposal_XAI_CXR_Trustworthiness_ArianAbbasi_v1.pdf`

## Key Feedback

1. The proposed three criteria (Localization, Faithfulness, Stability) are relevant, but the proposal should move beyond evaluating them independently.

2. The thesis should be reframed around a multidimensional reliability assessment framework for XAI explanations in chest X-ray diagnosis.

3. The framework should include a Reliability Profile for each explanation, covering:
   - Localization
   - Faithfulness
   - Stability
   - Anatomical Plausibility
   - Radiologist Agreement

4. A Composite Explanation Reliability Index (CERI) should be considered for quantitative comparison and ranking of XAI methods.

5. The proposal should include annotation-based evaluation using datasets with lesion-level annotations, such as:
   - VinDr-CXR
   - RSNA Pneumonia Detection
   - SIIM-ACR Pneumothorax

6. The evaluation should include at least two radiologists on a limited subset of images, if possible.

7. The literature review should be reorganized around XAI method families:
   - CAM-based methods
   - Gradient-based methods
   - Perturbation-based methods
   - Attention-based methods

8. The proposed stability improvement method should be positioned as:
   - Stability-Enhanced Explanation Aggregation (SEEA)

9. The revised proposal should focus on:
   - reliability-aware evaluation
   - robust ranking of XAI methods
   - anatomical plausibility
   - improving explanation stability

## Action Items for Proposal v2

- Update thesis title.
- Rewrite problem statement around reliability-aware XAI.
- Add Anatomical Plausibility Score (APS).
- Define Reliability Profile and CERI.
- Add radiologist agreement as validation layer.
- Revise research questions and hypotheses.
- Revise novelty section.
- Revise methodology section.
- Update literature review and references.
