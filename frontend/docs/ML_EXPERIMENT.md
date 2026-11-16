# Fixed experiment: model v2

Source: UCI Bike Sharing, DOI 10.24432/C5W894, historical 2011–2012 hourly records. Credit: Hadi Fanaee-T (2013), CC BY 4.0 per UCI. Related paper: Fanaee-T & Gama (2013), DOI 10.1007/s13748-013-0040-3.

Features, in order: hour, weekday (Sunday=0), month, weather category, holiday, working day, year (2011=0, 2012=1). Working day is derived from weekday and holiday in the UI. No temperature conversion or missing weather forecast is assumed. `casual`, `registered`, and `instant` are excluded.

Whole-date temporal split:

| Split | Range | Rows |
| --- | --- | ---: |
| Training | 2011-01-01 to 2012-05-25 | 12,139 |
| Validation | 2012-05-26 to 2012-09-12 | 2,640 |
| Final test | 2012-09-13 to 2012-12-31 | 2,600 |

Four ML candidates were declared before fitting. The baseline is the training mean for the hour of week, with the global training mean as fallback. The winner was selected by validation MAE among ML candidates, with the baseline reported as a reference. The winner was not refit on validation. Negative predictions are clipped to zero consistently in scoring and browser inference.

| Candidate | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| Hour-of-week baseline | 127.5 | 169.1 |
| Ridge, one-hot, alpha 10 | 114.7 | 143.1 |
| Gradient Boosting, 150 trees, depth 3 | 74.3 | 105.6 |
| Gradient Boosting, 200 trees, depth 4 | 56.1 | 81.7 |
| Histogram Boosting, 150 iterations, 15 leaves | **48.1** | **74.4** |

Selected model test MAE **62.59**, RMSE **97.50** rentals/hour. Baseline test MAE **103.19**, RMSE **149.71**. Relative test MAE reduction **39.35%**; this is not an accuracy percentage or a confidence interval.

The exported histogram trees use version-specific sklearn internals (`1.9.1`); dependencies are locked in the main desktop project. Export is verified against sklearn on all 17,379 original feature rows. The actual JavaScript runtime is also verified on all 32,256 combinations of hour × weekday × month × weather × holiday × year. Maximum observed absolute prediction difference: **0**. Repeat parity verification after any dependency or export change.

The UI uses browser-side model inference, not a Python API. No validation/test labels are used to train the model. The 72 displayed test examples are uniformly selected by index across the entire test period, without error-based cherry picking; they are not contiguous hourly records. This single temporal holdout has seasonal/year distribution shift, only three training observations for severe weather, and no evidence of accuracy for today's demand. Further model selection should use training-period temporal cross-validation, not repeated tuning against the published final test.

Exact metrics, split bounds, dataset hash, seed, and dependency versions are included in the downloadable experiment JSON.
