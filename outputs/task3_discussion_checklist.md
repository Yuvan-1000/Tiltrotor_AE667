Task 3.4 discussion notes -- computed from this run
================================================================================
Error metrics:
  CT: RMSE=2.704e-03  MAE=2.038e-03  MAPE=47.72%
  CQ: RMSE=2.142e-04  MAE=1.712e-04  MAPE=46.40%

Observed pattern: BEMT UNDER-predicts CT at every theta0 tested (never
over-predicts). The ratio CT_exp / CT_bemt by collective setting:
  1deg:1.67x, 2deg:1.81x, 4deg:1.94x, 6deg:2.02x, 8deg:2.06x, 10deg:2.04x, 12deg:1.90x
This ratio is smooth and one-directional (roughly 1.7-2.0x across the whole
range, not scattered) -- that shape points to a genuine, systematic modeling
gap rather than a numerical bug or noisy data.

Stall check: across the whole sweep, sectional stall was
NEVER flagged
against the adopted alpha_stall criterion -- so stall is
NOT the explanation
for this particular discrepancy; the maximum blade angle of attack reached
in this run stayed under the adopted stall angle at every collective tested.

--------------------------------------------------------------------------------
Open questions for your team to resolve (use your own judgement / the source
paper -- do not treat the bullets below as established facts to copy in):

1. The handout's Cl=5.75*alpha, Cd=0.0113+1.25*alpha^2 model has no camber
   term (Cl=0 at alpha=0) and no realistic CLmax/stall break. Check what
   airfoil section was actually used on the Knight & Hefner model rotor
   blades (see the source PDF) -- if it had camber or a higher effective
   lift-curve slope than 5.75/rad, that alone could explain a roughly
   constant multiplicative under-prediction like the one seen here.
2. Classical (even tip-loss-corrected) momentum theory neglects wake
   contraction, which is a known source of thrust under-prediction; the
   source paper discusses this limitation of its own theory -- read the
   relevant section and summarize it in your own words rather than quoting
   it directly.
3. No rotational augmentation of inboard lift is modeled (2-D sectional
   Cl(alpha) assumed to hold locally on a rotating blade) -- a known
   under-prediction mechanism in both helicopter and wind-turbine BEMT.
4. Over what theta0 / CT range would you consider this specific model
   (linear Cl, no camber, no Re correction) trustworthy for preliminary
   design, given the MAPE above and the fact the gap is systematic rather
   than random?

Primary source (public domain, NASA/NACA TN 626):
  https://archive.org/download/nasa_techdoc_19930081433/19930081433.pdf
Full-text OCR (useful for finding the tested airfoil section, RPM, Re):
  https://archive.org/stream/nasa_techdoc_19930081433/19930081433_djvu.txt
