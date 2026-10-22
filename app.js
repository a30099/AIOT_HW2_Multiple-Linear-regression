const $ = (selector) => document.querySelector(selector);
const problemSelect = $('#problemSelect');
const datasetUrl = $('#datasetUrl');
const loadButton = $('#loadButton');
const fileInput = $('#fileInput');
const targetSelect = $('#targetSelect');
const modelSelect = $('#modelSelect');
const featureCount = $('#featureCount');
const runButton = $('#runButton');
const statusBox = $('#status');
let currentData = null;
let currentName = '';

problemSelect.addEventListener('change', () => {
  datasetUrl.value = problemSelect.value === 'student'
    ? 'data/StudentPerformanceFactors.csv'
    : 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';
  currentData = null;
  runButton.disabled = true;
  targetSelect.disabled = true;
  statusBox.textContent = '請載入資料';
});

loadButton.addEventListener('click', async () => {
  const url = datasetUrl.value.trim();
  if (!url) return setStatus('請輸入 CSV 網址', true);
  setStatus('正在載入 CSV…');
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    useCsv(await response.text(), url.split('/').pop() || 'remote.csv');
  } catch (error) {
    setStatus(`載入失敗：${error.message}。來源可能未開放 CORS，可改用檔案上傳。`, true);
  }
});

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  setStatus('正在讀取檔案…');
  try { useCsv(await file.text(), file.name); }
  catch (error) { setStatus(`讀取失敗：${error.message}`, true); }
});

runButton.addEventListener('click', runRegression);
modelSelect.addEventListener('change', () => { featureCount.disabled = modelSelect.value === 'simple'; });

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? '#a43427' : '';
}

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = [];
  let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { value += '"'; i++; }
      else quoted = !quoted;
    } else if (char === ',' && !quoted) { row.push(value.trim()); value = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i++;
      row.push(value.trim());
      if (row.some(cell => cell !== '')) rows.push(row);
      row = []; value = '';
    } else value += char;
  }
  if (value || row.length) { row.push(value.trim()); rows.push(row); }
  if (rows.length < 3) throw new Error('CSV 至少需要標題與兩筆資料');
  const width = rows[0].length;
  return { headers: rows[0].map((h, i) => h || `Column_${i + 1}`), rows: rows.slice(1).filter(r => r.length === width) };
}

function useCsv(text, name) {
  const parsed = parseCsv(text);
  const numeric = parsed.headers.filter((_, col) => {
    const values = parsed.rows.map(r => r[col]).filter(v => v !== '');
    return values.length >= 3 && values.filter(v => Number.isFinite(Number(v))).length / values.length > .9;
  });
  if (numeric.length < 2) throw new Error('至少需要兩個數值欄位');
  currentData = { ...parsed, numeric };
  currentName = name;
  targetSelect.innerHTML = numeric.map(h => `<option value="${escapeHtml(h)}">${escapeHtml(h)}</option>`).join('');
  const preferred = numeric.includes('Exam_Score') ? 'Exam_Score' : numeric[numeric.length - 1];
  targetSelect.value = preferred;
  targetSelect.disabled = false;
  runButton.disabled = false;
  featureCount.max = Math.max(1, numeric.length - 1);
  featureCount.value = Math.min(6, numeric.length - 1);
  setStatus(`已載入 ${name} · ${parsed.rows.length.toLocaleString()} 筆 · ${parsed.headers.length} 欄`);
  renderPreview(parsed);
}

function runRegression() {
  if (!currentData) return;
  try {
    setStatus('正在計算模型…');
    const target = targetSelect.value;
    const targetIndex = currentData.headers.indexOf(target);
    const candidates = currentData.numeric.filter(h => h !== target);
    const clean = currentData.rows.map(row => ({ row, y: Number(row[targetIndex]) }))
      .filter(item => Number.isFinite(item.y) && candidates.every(h => Number.isFinite(Number(item.row[currentData.headers.indexOf(h)]))));
    if (clean.length < Math.max(12, candidates.length + 3)) throw new Error('完整數值資料不足，無法穩定建立模型');
    const correlations = candidates.map(name => {
      const col = currentData.headers.indexOf(name);
      return { name, score: Math.abs(correlation(clean.map(v => Number(v.row[col])), clean.map(v => v.y))) || 0 };
    }).sort((a, b) => b.score - a.score);
    const count = modelSelect.value === 'simple' ? 1 : Math.max(1, Math.min(Number(featureCount.value) || 1, correlations.length, 12));
    const selected = correlations.slice(0, count).map(v => v.name);
    const selectedIndices = selected.map(h => currentData.headers.indexOf(h));
    const train = [], test = [];
    clean.forEach((item, i) => (i % 5 === 0 ? test : train).push(item));
    const means = selectedIndices.map(idx => mean(train.map(v => Number(v.row[idx]))));
    const stds = selectedIndices.map((idx, j) => std(train.map(v => Number(v.row[idx])), means[j]) || 1);
    const matrix = train.map(v => [1, ...selectedIndices.map((idx, j) => (Number(v.row[idx]) - means[j]) / stds[j])]);
    const yTrain = train.map(v => v.y);
    const coefficients = ols(matrix, yTrain);
    const actual = test.map(v => v.y);
    const predicted = test.map(v => dot([1, ...selectedIndices.map((idx, j) => (Number(v.row[idx]) - means[j]) / stds[j])], coefficients));
    const errors = actual.map((v, i) => v - predicted[i]);
    const mae = mean(errors.map(Math.abs));
    const rmse = Math.sqrt(mean(errors.map(v => v * v)));
    const yMean = mean(actual);
    const ssRes = errors.reduce((s, v) => s + v * v, 0);
    const ssTot = actual.reduce((s, v) => s + (v - yMean) ** 2, 0);
    const r2 = ssTot ? 1 - ssRes / ssTot : 0;
    $('#metricMae').textContent = formatMetric(mae);
    $('#metricRmse').textContent = formatMetric(rmse);
    $('#metricR2').textContent = formatMetric(r2);
    $('#dataSummary').textContent = `${currentName} 共 ${clean.length.toLocaleString()} 筆完整數值資料；依固定規則切分為 ${train.length.toLocaleString()} 筆訓練資料與 ${test.length.toLocaleString()} 筆測試資料。特徵選擇依與目標欄位的絕對 Pearson 相關係數排序。`;
    $('#featureTags').innerHTML = selected.map((name, i) => `<span>${i + 1}. ${escapeHtml(name)}</span>`).join('');
    $('#labOutput').hidden = false;
    drawChart(actual, predicted, target);
    setStatus(`分析完成 · ${modelSelect.options[modelSelect.selectedIndex].text}`);
  } catch (error) { setStatus(`分析失敗：${error.message}`, true); }
}

function ols(x, y) {
  const xt = transpose(x);
  const xtx = multiply(xt, x);
  for (let i = 1; i < xtx.length; i++) xtx[i][i] += 1e-8;
  return multiplyMatrixVector(invert(xtx), multiplyMatrixVector(xt, y));
}

function invert(matrix) {
  const n = matrix.length;
  const a = matrix.map((row, i) => [...row, ...Array.from({ length:n }, (_, j) => i === j ? 1 : 0)]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(a[r][i]) > Math.abs(a[pivot][i])) pivot = r;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    if (Math.abs(a[i][i]) < 1e-12) throw new Error('特徵矩陣無法反矩陣，請減少特徵數');
    const div = a[i][i];
    for (let j = 0; j < 2 * n; j++) a[i][j] /= div;
    for (let r = 0; r < n; r++) if (r !== i) {
      const factor = a[r][i];
      for (let j = 0; j < 2 * n; j++) a[r][j] -= factor * a[i][j];
    }
  }
  return a.map(row => row.slice(n));
}

function transpose(a) { return a[0].map((_, i) => a.map(row => row[i])); }
function multiply(a, b) { const bt = transpose(b); return a.map(row => bt.map(col => dot(row, col))); }
function multiplyMatrixVector(a, v) { return a.map(row => dot(row, v)); }
function dot(a, b) { return a.reduce((sum, v, i) => sum + v * b[i], 0); }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function std(a, m = mean(a)) { return Math.sqrt(mean(a.map(v => (v - m) ** 2))); }
function correlation(a, b) { const am = mean(a), bm = mean(b); const top = a.reduce((s, v, i) => s + (v-am)*(b[i]-bm), 0); const bottom = Math.sqrt(a.reduce((s,v)=>s+(v-am)**2,0)*b.reduce((s,v)=>s+(v-bm)**2,0)); return bottom ? top/bottom : 0; }
function formatMetric(value) { return Number.isFinite(value) ? value.toFixed(4) : '—'; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }

function renderPreview(data) {
  const columns = data.headers.slice(0, 8);
  const indices = columns.map(h => data.headers.indexOf(h));
  $('#previewTable').innerHTML = `<thead><tr>${columns.map(h => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead><tbody>${data.rows.slice(0,5).map(row => `<tr>${indices.map(i => `<td>${escapeHtml(row[i])}</td>`).join('')}</tr>`).join('')}</tbody>`;
}

function drawChart(actual, predicted, label) {
  const canvas = $('#predictionChart');
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.parentElement.clientWidth - 30;
  const cssHeight = Math.max(260, cssWidth * .47);
  canvas.width = cssWidth * dpr; canvas.height = cssHeight * dpr;
  canvas.style.width = `${cssWidth}px`; canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const pad = { l:52, r:18, t:28, b:45 }, w=cssWidth-pad.l-pad.r, h=cssHeight-pad.t-pad.b;
  let min = Math.min(...actual, ...predicted), max = Math.max(...actual, ...predicted);
  const margin = (max-min)*.07 || 1; min-=margin; max+=margin;
  const sx = v => pad.l + (v-min)/(max-min)*w, sy = v => pad.t+h-(v-min)/(max-min)*h;
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,cssWidth,cssHeight);
  ctx.strokeStyle='#dfe3df'; ctx.lineWidth=1; ctx.font='10px DM Mono, monospace'; ctx.fillStyle='#6c746f';
  for(let i=0;i<=4;i++){const v=min+(max-min)*i/4; const x=sx(v), y=sy(v); ctx.beginPath(); ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+h);ctx.stroke(); ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+w,y);ctx.stroke(); ctx.fillText(v.toFixed(1),x-10,pad.t+h+18);ctx.fillText(v.toFixed(1),5,y+3);}
  ctx.strokeStyle='#d04d35';ctx.lineWidth=1.5;ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(sx(min),sy(min));ctx.lineTo(sx(max),sy(max));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='rgba(23,63,50,.36)'; const step=Math.max(1,Math.floor(actual.length/600)); for(let i=0;i<actual.length;i+=step){ctx.beginPath();ctx.arc(sx(actual[i]),sy(predicted[i]),2.4,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle='#17211c';ctx.font='600 11px Noto Sans TC, sans-serif';ctx.fillText(`實際 ${label}`,pad.l+w/2-30,cssHeight-8);ctx.save();ctx.translate(13,pad.t+h/2+30);ctx.rotate(-Math.PI/2);ctx.fillText(`預測 ${label}`,0,0);ctx.restore();
}

window.addEventListener('DOMContentLoaded', () => loadButton.click());
