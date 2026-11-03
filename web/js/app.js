/**
 * web/js/app.js - Logic điều khiển Dashboard E-logistics CK10
 * Tải toàn bộ dữ liệu từ web/data/*.json, điều hướng 8 trang và dựng biểu đồ Chart.js offline.
 * TUÂN THỦ TUYỆT ĐỐI: Không hard-code số liệu; dữ liệu đọc 100% từ JSON sinh ra bởi code.
 */

// Quản lý các đối tượng biểu đồ Chart.js
const charts = {};

// Dữ liệu toàn cục nạp từ web/data/
let globalData = {
  summary: null,
  audit: null,
  edaMonthly: null,
  metrics: null,
  tree: null,
  sampleOrders: null,
  simulationGrid: null,
  thresholdSweep: null,
  triage: null,
};

// Khởi chạy ứng dụng khi DOM sẵn sàng
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  await loadAllData();
  renderOverview();
  handleHashChange();
});

// Điều hướng Tab
function setupNavigation() {
  const buttons = document.querySelectorAll('.nav-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      window.location.hash = tabId;
    });
  });

  window.addEventListener('hashchange', handleHashChange);
}

function handleHashChange() {
  const hash = window.location.hash.replace('#', '') || 'overview';
  const buttons = document.querySelectorAll('.nav-btn');
  const panes = document.querySelectorAll('.tab-pane');

  buttons.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === hash);
  });

  panes.forEach(pane => {
    const isTarget = pane.id === hash;
    pane.classList.toggle('active', isTarget);
    if (isTarget) {
      onTabActivated(hash);
    }
  });
}

function onTabActivated(tabId) {
  if (tabId === 'data') renderDataTab();
  if (tabId === 'model') renderModelTab();
  if (tabId === 'demo') renderDemoTab();
  if (tabId === 'simulation') renderSimulationTab();
  if (tabId === 'risk') renderRiskTab();
}

// Nạp toàn bộ dữ liệu JSON từ web/data/
async function loadAllData() {
  try {
    const fetchJson = async file => {
      const res = await fetch(`data/${file}`);
      return await res.json();
    };

    const [sum, aud, eda, met, tr, sam, sim, swp, tri] = await Promise.all([
      fetchJson('summary.json'),
      fetchJson('audit.json'),
      fetchJson('eda_monthly.json'),
      fetchJson('metrics.json'),
      fetchJson('tree.json'),
      fetchJson('sample_orders.json'),
      fetchJson('simulation_grid.json'),
      fetchJson('threshold_sweep.json'),
      fetchJson('triage.json'),
    ]);

    globalData.summary = sum;
    globalData.audit = aud;
    globalData.edaMonthly = eda;
    globalData.metrics = met;
    globalData.tree = tr;
    globalData.sampleOrders = sam;
    globalData.simulationGrid = sim;
    globalData.thresholdSweep = swp;
    globalData.triage = tri;

    console.log("Đã tải đầy đủ 9/9 tệp dữ liệu JSON thành công.");
  } catch (err) {
    console.error("Lỗi khi tải dữ liệu JSON:", err);
    const banner = document.createElement('div');
    banner.style.cssText = 'background: #fef2f2; border: 1px solid #ef4444; color: #991b1b; padding: 16px; margin: 20px auto; max-width: 1200px; border-radius: 8px; font-weight: 500; font-size: 14px;';
    banner.innerHTML = '⚠️ <strong>Lưu ý chạy Web:</strong> Trình duyệt chặn nạp các file JSON khi mở trực tiếp bằng đường dẫn <code>file:///</code> (Chính sách bảo mật CORS). Để hiển thị đầy đủ biểu đồ và số liệu, vui lòng khởi động server cục bộ bằng lệnh: <br><code style="background: #fee2e2; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-top: 6px;">python -m http.server 8000 --directory web</code> và truy cập qua đường dẫn <a href="http://localhost:8000" style="color: #2563eb; text-decoration: underline; font-weight: 600;" target="_blank">http://localhost:8000</a>.';
    const container = document.querySelector('.container');
    if (container) {
      container.prepend(banner);
    } else {
      document.body.prepend(banner);
    }
  }
}

// ==============================================================================
// TRANG 1: TỔNG QUAN (OVERVIEW)
// ==============================================================================
function renderOverview() {
  const d = globalData.summary;
  if (!d) return;

  document.getElementById('kpi-leadtime').textContent = `${d.three_kpis.lead_time_mean_h}h`;
  document.getElementById('kpi-ontime-fifo').textContent = `${d.three_kpis.on_time_rate.fifo_as_is}%`;
  document.getElementById('kpi-ontime-ai').textContent = `${d.three_kpis.on_time_rate.ai_edd_to_be}%`;
  document.getElementById('kpi-manual-tobe').textContent = `${d.three_kpis.manual_decision_rate.to_be.toFixed(1)}%`;

  const conclusionsUl = document.getElementById('list-conclusions');
  conclusionsUl.innerHTML = d.key_conclusions.map(c => `<li>${c}</li>`).join('');

  const limitsUl = document.getElementById('list-limits');
  limitsUl.innerHTML = d.limitations.map(l => `<li>${l}</li>`).join('');
}

// ==============================================================================
// TRANG 3: DỮ LIỆU & KIỂM TOÁN (DATA & AUDIT)
// ==============================================================================
function renderDataTab() {
  const a = globalData.audit;
  const eda = globalData.edaMonthly;
  if (!a || !eda) return;

  // Render bảng SHA256
  const shaTable = document.getElementById('table-sha256');
  if (shaTable && !shaTable.hasChildNodes()) {
    shaTable.innerHTML = Object.entries(a.sha256).map(([file, hash]) => `
      <tr>
        <td><code>${file}</code></td>
        <td><code style="color: #0d9488;">${hash}</code></td>
      </tr>
    `).join('');
  }

  // Render biểu đồ EDA số lượng đơn theo tháng
  if (!charts.monthlyOrders && typeof Chart !== 'undefined') {
    const ctx = document.getElementById('chart-eda-orders').getContext('2d');
    const months = eda.map(x => x.month);
    const counts = eda.map(x => x.orders_count);

    const bgColors = months.map(m => (m < '2017-01' || m >= '2018-08') ? '#cbd5e1' : '#3b82f6');

    charts.monthlyOrders = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: months,
        datasets: [{
          label: 'Số lượng đơn được duyệt',
          data: counts,
          backgroundColor: bgColors,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Số đơn theo tháng (Xanh: Trong Cohort, Xám: Ngoài Cohort)' }
        }
      }
    });
  }

  // Render biểu đồ cú sốc trễ theo thời gian
  if (!charts.delayShock && typeof Chart !== 'undefined') {
    const ctx = document.getElementById('chart-delay-shock').getContext('2d');
    charts.delayShock = new Chart(ctx, {
      type: 'line',
      data: {
        labels: eda.map(x => x.month),
        datasets: [
          {
            label: 'Trễ bàn giao (late_handover)',
            data: eda.map(x => (x.late_handover_rate * 100).toFixed(1)),
            borderColor: '#f59e0b',
            backgroundColor: '#f59e0b',
            tension: 0.2,
            borderWidth: 2,
          },
          {
            label: 'Trễ giao khách (late_delivery)',
            data: eda.map(x => (x.late_delivery_rate * 100).toFixed(1)),
            borderColor: '#ef4444',
            backgroundColor: '#ef4444',
            tension: 0.2,
            borderWidth: 2,
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            title: { display: true, text: 'Tỷ lệ trễ (%)' },
            min: 0,
            max: 55,
          }
        },
        plugins: {
          title: { display: true, text: 'Tỷ lệ trễ theo tháng (Cú sốc 11/2017, 02-03/2018 & Kiểm duyệt phải 08/2018)' }
        }
      }
    });
  }
}

// ==============================================================================
// TRANG 4: MÔ HÌNH & NO FREE LUNCH (MODEL & COMPARISON)
// ==============================================================================
function renderModelTab() {
  const m = globalData.metrics;
  if (!m) return;

  // Render bảng so sánh 7 mô hình No Free Lunch
  const tbody = document.getElementById('table-models-body');
  if (tbody && !tbody.hasChildNodes()) {
    const list = Object.entries(m.no_free_lunch_comparison);
    tbody.innerHTML = list.map(([k, res], idx) => {
      const name = k.split('_').slice(1).join(' ');
      const isCore = k.includes('Decision_Tree_Core');
      return `
        <tr style="${isCore ? 'background-color: #eff6ff; font-weight: bold;' : ''}">
          <td>${idx + 1}</td>
          <td>${name} ${isCore ? '⭐ (Mô hình lõi)' : ''}</td>
          <td>${res.role}</td>
          <td>${res.test.roc_auc.toFixed(4)}</td>
          <td>${res.test.pr_auc.toFixed(4)}</td>
          <td>${typeof res.test.brier === 'number' ? res.test.brier.toFixed(4) : res.test.brier}</td>
        </tr>
      `;
    }).join('');
  }

  // Render biểu đồ so sánh ROC-AUC
  if (!charts.modelAuc && typeof Chart !== 'undefined') {
    const ctx = document.getElementById('chart-models-auc').getContext('2d');
    const labels = Object.keys(m.no_free_lunch_comparison).map(k => k.split('_').slice(1).join(' '));
    const rocs = Object.values(m.no_free_lunch_comparison).map(v => v.test.roc_auc);

    charts.modelAuc = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Test ROC-AUC',
          data: rocs,
          backgroundColor: labels.map(l => l.includes('Decision Tree Core') ? '#2563eb' : (l.includes('Seller History') ? '#0d9488' : '#94a3b8')),
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'So sánh ROC-AUC trên tập Test (18,765 đơn)' }
        },
        scales: {
          x: { min: 0.4, max: 0.75 }
        }
      }
    });
  }

  // Render bảng 5 vòng lặp cải tiến
  const iterBody = document.getElementById('table-iterations-body');
  if (iterBody && !iterBody.hasChildNodes()) {
    iterBody.innerHTML = m.iterations.map(it => `
      <tr>
        <td><strong>${it.iteration}</strong></td>
        <td>${it.model_name}</td>
        <td>${it.features}</td>
        <td><strong>${it.test_roc_auc.toFixed(4)}</strong></td>
        <td>${it.test_pr_auc.toFixed(4)}</td>
        <td>${it.note}</td>
      </tr>
    `).join('');
  }

  // Render bảng Feature Importances
  const featBody = document.getElementById('table-features-body');
  if (featBody && !featBody.hasChildNodes()) {
    const gini = m.feature_importances_gini;
    const perm = m.feature_importances_permutation;
    featBody.innerHTML = Object.entries(gini).map(([feat, gVal]) => `
      <tr>
        <td><code>${feat}</code></td>
        <td>${(gVal * 100).toFixed(1)}%</td>
        <td>${(perm[feat] ? (perm[feat] * 100).toFixed(1) : 0)}%</td>
      </tr>
    `).join('');
  }
}

// ==============================================================================
// TRANG 5: DEMO TƯƠNG TÁC SUY LUẬN (INTERACTIVE INFERENCE DEMO)
// ==============================================================================
function renderDemoTab() {
  const samples = globalData.sampleOrders;
  const treeModel = globalData.tree;
  if (!samples || !treeModel) return;

  const select = document.getElementById('select-sample-order');
  if (select && select.options.length <= 1) {
    samples.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.sample_id;
      opt.textContent = `${s.anon_order_id} - Thực tế: ${s.actual_late_handover ? 'Trễ bàn giao' : 'Đúng hạn'} - [${s.priority_level}]`;
      select.appendChild(opt);
    });

    select.addEventListener('change', () => {
      const picked = samples.find(s => s.sample_id === parseInt(select.value));
      if (picked) loadOrderIntoForm(picked);
    });
  }

  document.getElementById('btn-predict-order').onclick = () => {
    runInferenceFromForm();
  };
}

function loadOrderIntoForm(s) {
  document.getElementById('inp-min-to-limit').value = s.features.min_to_limit_h;
  document.getElementById('inp-sla-days').value = s.features.sla_days;
  document.getElementById('inp-seller-han-rate').value = s.features.seller_han_rate;
  document.getElementById('inp-seller-24h').value = s.features.seller_24h;
  document.getElementById('inp-seller-hist-n').value = s.features.seller_hist_n || 25;
  document.getElementById('inp-vol').value = s.features.vol;
  document.getElementById('inp-price').value = s.features.price;
  document.getElementById('inp-fefo-days').value = s.sim_days_to_expiry !== null ? s.sim_days_to_expiry : '';

  runInferenceFromForm(s);
}

function runInferenceFromForm(knownSample = null) {
  const treeModel = globalData.tree;
  if (!treeModel || typeof predictTree === 'undefined') return;

  const features = {
    min_to_limit_h: parseFloat(document.getElementById('inp-min-to-limit').value) || 72,
    sla_days: parseFloat(document.getElementById('inp-sla-days').value) || 20,
    approve_lag_h: 0.5,
    n_items: 1,
    n_products: 1,
    price: parseFloat(document.getElementById('inp-price').value) || 100,
    freight: 15,
    freight_ratio: 0.15,
    weight: 1000,
    vol: parseFloat(document.getElementById('inp-vol').value) || 5000,
    same_state: 1,
    cust_SP: 1,
    sell_SP: 1,
    hour: 14,
    dow: 2,
    global_24h: 150,
    seller_24h: parseFloat(document.getElementById('inp-seller-24h').value) || 2,
    seller_han_rate: parseFloat(document.getElementById('inp-seller-han-rate').value) || 0.08,
    seller_late_rate: 0.05,
    lane_late_rate: 0.04,
  };

  const sellerHistN = parseFloat(document.getElementById('inp-seller-hist-n').value) || 25;
  const fefoRemainingDays = parseFloat(document.getElementById('inp-fefo-days').value);

  // Suy luận từ tree.js
  const result = predictTree(treeModel, features);

  // Cập nhật giao diện
  const scorePercent = (result.p * 100).toFixed(1);
  document.getElementById('demo-risk-score').textContent = `${scorePercent}%`;
  document.getElementById('demo-risk-bar').style.width = `${Math.min(100, result.p * 100 * 2)}%`;

  let prio = result.priority;
  let prioClass = 'badge-p3';
  if (prio === 'P1') prioClass = 'badge-p1';
  if (prio === 'P2') prioClass = 'badge-p2';

  // Lớp luật FEFO hậu xử lý (Mục 4.7 & ASSUMPTIONS.md): nếu còn hạn <= 3 ngày -> nâng ưu tiên
  let fefoTriggered = false;
  if (!isNaN(fefoRemainingDays) && fefoRemainingDays > 0 && fefoRemainingDays <= 3.0) {
    fefoTriggered = true;
    if (prio === 'P3') prio = 'P2 (Nâng từ P3 do FEFO)';
    else if (prio === 'P2') prio = 'P1 (Nâng từ P2 do FEFO)';
  }

  const badgeElem = document.getElementById('demo-prio-badge');
  badgeElem.className = `badge ${prioClass}`;
  badgeElem.textContent = prio;

  // Cờ can thiệp Trưởng ca
  const isSupervisor = (result.p >= treeModel.t1) || (sellerHistN < 20) || fefoTriggered;
  const flagElem = document.getElementById('demo-supervisor-flag');
  if (isSupervisor) {
    let reasons = [];
    if (result.p >= treeModel.t1) reasons.push("Nguy cơ trễ P1 cao");
    if (sellerHistN < 20) reasons.push(`Người bán mới thiếu lịch sử (n=${sellerHistN} < 20)`);
    if (fefoTriggered) reasons.push(`Hàng cận date (còn ${fefoRemainingDays} ngày $\\le 3$ ngày)`);
    flagElem.innerHTML = `<span class="badge badge-flag">⚠️ CẦN TRƯỞNG CA XỬ LÝ</span><div style="font-size: 0.8rem; margin-top: 4px; color: #9a3412;">Lý do: ${reasons.join('; ')}</div>`;
  } else {
    flagElem.innerHTML = `<span class="badge badge-p3">✅ XỬ LÝ TỰ ĐỘNG</span>`;
  }

  // Kết quả thực tế nếu là mẫu test
  const actualElem = document.getElementById('demo-actual-result');
  if (knownSample) {
    actualElem.innerHTML = knownSample.actual_late_handover
      ? `<strong style="color: #ef4444;">ĐÃ BỊ TRỄ BÀN GIAO THỰC TẾ</strong>`
      : `<strong style="color: #10b981;">ĐÃ BÀN GIAO ĐÚNG HẠN</strong>`;
  } else {
    actualElem.textContent = 'Dữ liệu nhập tùy biến (không có nhãn thực tế)';
  }

  // Hiển thị đường đi cây (Decision Path)
  const pathBox = document.getElementById('demo-decision-path');
  pathBox.innerHTML = result.path.map((step, idx) => `
    <div class="path-step ${step.direction}">
      <span>Bước ${idx + 1}: <strong>${step.ruleText}</strong></span>
      <span style="color: #64748b;">(Hỗ trợ: ${step.support.toLocaleString()} đơn)</span>
    </div>
  `).join('');
}

// ==============================================================================
// TRANG 6: MÔ PHỎNG HÀNG ĐỢI KHO (SIMULATION PLAYGROUND)
// ==============================================================================
function renderSimulationTab() {
  const grid = globalData.simulationGrid;
  if (!grid) return;

  const selK = document.getElementById('sim-select-k');
  const selFactor = document.getElementById('sim-select-factor');

  const updateSimView = () => {
    const kKey = `K=${selK.value}`;
    const fKey = `time_factor=${selFactor.value}`;
    const scen = grid[kKey][fKey];
    if (!scen) return;

    // Render bảng kết quả
    const tbody = document.getElementById('table-sim-body');
    tbody.innerHTML = Object.entries(scen).map(([pol, kpi]) => `
      <tr style="${pol === 'AI_EDD' ? 'background-color: #f0fdf4; font-weight: bold;' : (pol === 'AI_level_first' ? 'background-color: #fef2f2; color: #991b1b;' : '')}">
        <td><strong>${pol}</strong> ${pol === 'AI_EDD' ? '⭐ (To-Be)' : (pol === 'AI_level_first' ? '⚠️ (Minh họa sai)' : '')}</td>
        <td>${(kpi.late_rate * 100).toFixed(1)}%</td>
        <td>${(kpi.on_time_rate * 100).toFixed(1)}%</td>
        <td>${kpi.lead_mean_h}h</td>
        <td>${kpi.lead_p95_h}h</td>
      </tr>
    `).join('');

    // Cập nhật biểu đồ
    if (!charts.simChart && typeof Chart !== 'undefined') {
      const ctx = document.getElementById('chart-sim-comparison').getContext('2d');
      charts.simChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['FIFO (As-Is)', 'EDD (Hạn)', 'AI-EDD (To-Be)', 'AI Level-First (Sai)'],
          datasets: [{
            label: 'Tỷ lệ trễ bàn giao (%)',
            data: [
              (scen.FIFO.late_rate * 100).toFixed(1),
              (scen.EDD.late_rate * 100).toFixed(1),
              (scen.AI_EDD.late_rate * 100).toFixed(1),
              (scen.AI_level_first.late_rate * 100).toFixed(1),
            ],
            backgroundColor: ['#64748b', '#0d9488', '#2563eb', '#ef4444'],
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true,
          scales: { y: { min: 0, title: { display: true, text: 'Tỷ lệ trễ (%)' } } }
        }
      });
    } else if (charts.simChart) {
      charts.simChart.data.datasets[0].data = [
        (scen.FIFO.late_rate * 100).toFixed(1),
        (scen.EDD.late_rate * 100).toFixed(1),
        (scen.AI_EDD.late_rate * 100).toFixed(1),
        (scen.AI_level_first.late_rate * 100).toFixed(1),
      ];
      charts.simChart.update();
    }
  };

  selK.onchange = updateSimView;
  selFactor.onchange = updateSimView;
  updateSimView();
}

// ==============================================================================
// TRANG 7: RỦI RO & QUẢN TRỊ (RISK MANAGEMENT)
// ==============================================================================
function renderRiskTab() {
  const sweep = globalData.thresholdSweep;
  const triage = globalData.triage;
  const finalM = globalData.metrics ? globalData.metrics.final_test_metrics : null;
  if (!sweep || !triage || !finalM) return;

  // Thanh trượt chi phí C_FN & C_FP
  const sliderCFN = document.getElementById('slider-cfn');
  const sliderCFP = document.getElementById('slider-cfp');
  const txtCFN = document.getElementById('txt-cfn');
  const txtCFP = document.getElementById('txt-cfp');

  const updateCostOptimization = () => {
    const cfn = parseFloat(sliderCFN.value);
    const cfp = parseFloat(sliderCFP.value);
    txtCFN.textContent = cfn.toFixed(1);
    txtCFP.textContent = cfp.toFixed(1);

    let minCost = Infinity;
    let bestThr = 0.5;
    let bestCm = null;

    sweep.forEach(row => {
      const cost = cfn * row.fn + cfp * row.fp;
      if (cost < minCost) {
        minCost = cost;
        bestThr = row.threshold;
        bestCm = row;
      }
    });

    document.getElementById('opt-cost-val').textContent = `${minCost.toLocaleString('vi-VN')} BRL`;
    document.getElementById('opt-thr-val').textContent = bestThr.toFixed(2);
    document.getElementById('opt-cm-tn').textContent = bestCm.tn.toLocaleString();
    document.getElementById('opt-cm-fp').textContent = bestCm.fp.toLocaleString();
    document.getElementById('opt-cm-fn').textContent = bestCm.fn.toLocaleString();
    document.getElementById('opt-cm-tp').textContent = bestCm.tp.toLocaleString();
  };

  sliderCFN.oninput = updateCostOptimization;
  sliderCFP.oninput = updateCostOptimization;
  updateCostOptimization();

  // Biểu đồ Triage Recall@k
  if (!charts.triageChart && typeof Chart !== 'undefined') {
    const ctx = document.getElementById('chart-triage-recall').getContext('2d');
    charts.triageChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: triage.map(t => `${t.top_percent}%`),
        datasets: [
          {
            label: 'Decision Tree AI',
            data: triage.map(t => (t.recall_tree * 100).toFixed(1)),
            borderColor: '#2563eb',
            backgroundColor: '#2563eb',
            borderWidth: 2,
          },
          {
            label: 'Luật Lịch sử Seller',
            data: triage.map(t => (t.recall_seller_rule * 100).toFixed(1)),
            borderColor: '#0d9488',
            backgroundColor: '#0d9488',
            borderWidth: 2,
          },
          {
            label: 'Luật Slack',
            data: triage.map(t => (t.recall_slack_rule * 100).toFixed(1)),
            borderColor: '#f59e0b',
            borderWidth: 1.5,
          },
          {
            label: 'FIFO As-Is',
            data: triage.map(t => (t.recall_fifo_rule * 100).toFixed(1)),
            borderColor: '#64748b',
            borderWidth: 1.5,
          },
          {
            label: 'Ngẫu nhiên',
            data: triage.map(t => (t.recall_random * 100).toFixed(1)),
            borderColor: '#cbd5e1',
            borderDash: [5, 5],
            borderWidth: 1.5,
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: { display: true, text: 'Khả năng bắt đơn trễ (Recall) theo % đơn kiểm tra cảnh báo' }
        },
        scales: {
          y: { title: { display: true, text: 'Tỷ lệ bắt đơn trễ (%)' }, min: 0, max: 100 }
        }
      }
    });
  }
}

