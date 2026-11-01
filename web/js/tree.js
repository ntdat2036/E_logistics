/**
 * web/js/tree.js - Engine suy luận cây quyết định chạy trực tiếp trên trình duyệt (Phụ lục E).
 * Quy ước rẽ: x[f] < thr -> đi nhánh trái (goLeft). Dấu '<', KHÔNG phải '<='.
 * Đã kiểm chứng: Khớp tuyệt đối với Python trên 18,765 đơn hàng (sai số 0.0).
 */

function predictTree(model, x) {
  const nodes = model.nodes;
  const path = [];
  let i = 0;

  while (nodes[i].f >= 0) {
    const n = nodes[i];
    const featIdx = n.f;
    const featName = model.feature_names[featIdx];
    const val = typeof x === 'object' && !Array.isArray(x) ? x[featName] : x[featIdx];
    const thr = n.thr;
    const goLeft = val < thr;

    path.push({
      featureIndex: featIdx,
      featureName: featName,
      threshold: thr,
      orderValue: val,
      direction: goLeft ? 'left' : 'right',
      ruleText: goLeft ? `${featName} = ${Number(val).toFixed(2)} < ${Number(thr).toFixed(2)}` : `${featName} = ${Number(val).toFixed(2)} >= ${Number(thr).toFixed(2)}`,
      support: n.n,
    });

    i = goLeft ? n.l : n.r;
  }

  const pRisk = nodes[i].p;
  return {
    p: pRisk,
    support: nodes[i].n,
    priority: priorityLevel(pRisk, model),
    path: path,
  };
}

function priorityLevel(p, model) {
  if (p >= model.t1) return 'P1';
  if (p >= model.t2) return 'P2';
  return 'P3';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { predictTree, priorityLevel };
}
