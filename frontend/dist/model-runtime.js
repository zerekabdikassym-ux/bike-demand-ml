/* Exact inference for the exported, version-locked sklearn estimator. */
((root) => {
  'use strict';
  function predict(model, features) {
    if (!Array.isArray(features) || features.length !== 7 || features.some(v => !Number.isFinite(v))) throw new Error('Expected seven finite features');
    if (model.kind === 'ridge') {
      let result = model.intercept, offset = 0;
      model.categories.forEach((categories, i) => { const index = categories.indexOf(features[i]); if (index >= 0) result += model.coefficients[offset + index]; offset += categories.length; });
      return Math.max(0, result);
    }
    let result = model.base;
    for (const tree of model.trees) {
      let node = tree[0];
      while (!(model.kind === 'hist_gradient_boosting' ? node[5] : node[0] === -1)) node = tree[features[node[2]] <= node[3] ? node[0] : node[1]];
      result += model.rate * node[4];
    }
    return Math.max(0, result);
  }
  root.BikeModelRuntime = {predict};
})(globalThis);
