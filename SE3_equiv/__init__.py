"""SE(3) equivariance evaluation of a landmark model.

    se3.py            rigid transforms: sampling, applying to a volume and to points
    predictor.py      nrrd -> landmarks (wraps inference/tools/infer_landmarks)
    metrics.py        per-transform equivariance errors and summaries
    run_se3_equiv.py  CLI: augment -> predict -> compare -> CSVs / plot
"""
