# Bike Demand — Frontend

A lightweight, buildless, responsive demand explorer with a trained ML model. Open `dist/index.html` directly or serve `dist` with a static server. No npm packages, inference API, or backend required.

The Russian interface shows an ML scenario prediction and a separate historical mean. The model was fitted on training data only (12,139 observed hours, 2011-01-01 through 2012-05-25), selected on validation, and evaluated on a later final test. This is a historical scenario estimator, not a validated forecast for today's demand.

Features: day/weather/hour/month/year/holiday scenarios, keyboard-accessible two-series SVG chart, sparse-weather and extrapolation warnings, CSV export, quality dashboard, downloadable experiment metrics, dataset and methodology pages, responsive mobile layout.

`dist/data.js` contains training aggregates `[weekday, hour, weather, observation_count, rental_sum]` for the historical reference. For groups with fewer than five weather-matched observations, that reference falls back to the same weekday/hour without weather filtering. The reference does not filter month/year/holiday. `dist/model.js` contains the trained histogram-boosting trees and measured evaluation results, including 72 uniformly sampled test examples. No validation/test rows are used to fit the model.

Final test MAE: **62.59** rentals/hour, compared with **103.19** for the hour-of-week baseline (39.35% MAE reduction). These are not accuracy percentages. See [the experiment record](docs/ML_EXPERIMENT.md).

Browser inference is implemented in `dist/model-runtime.js`. The desktop project includes training and export scripts, locked Python dependencies, and a parity fixture. Verify JavaScript parity with:

```text
node check-model.cjs <absolute-path-to-desktop-project>/artifacts/parity-fixture.json
```

32,256 calendar/weather scenarios were checked against sklearn, with zero observed prediction difference. The exported sklearn internals are version-specific; repeat verification after dependency changes.

Dataset credit: Fanaee-T, H. (2013), Bike Sharing, UCI Machine Learning Repository, DOI: 10.24432/C5W894, CC BY 4.0 per the UCI dataset page. Related paper: Fanaee-T & Gama (2013), DOI: 10.1007/s13748-013-0040-3.

Source dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Next: further temporal evaluation, newer data, and an optional FastAPI service. Source remains compatible with static hosting and local offline inference (the optional web font falls back to system fonts).
