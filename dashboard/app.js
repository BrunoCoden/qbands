const CSV_URL = '../tablaQ.csv';
const DEFAULT_REFRESH_MS = 5000;

const elements = {
  autoRefresh: document.getElementById('auto-refresh'),
  refreshLabel: document.getElementById('refresh-interval'),
  reloadBtn: document.getElementById('reload-btn'),
  filterUpper: document.getElementById('filter-upper'),
  filterLower: document.getElementById('filter-lower'),
  rowsLimit: document.getElementById('rows-limit'),
  tableHead: document.querySelector('#data-table thead'),
  tableBody: document.querySelector('#data-table tbody'),
  lastUpdated: document.getElementById('last-updated'),
  summary: document.getElementById('summary-content'),
  summaryTemplate: document.getElementById('summary-template'),
};

const state = {
  headers: [],
  rows: [],
  timerId: null,
};

function setLastUpdated(text, isError = false) {
  if (!elements.lastUpdated) return;
  elements.lastUpdated.textContent = text || '';
  elements.lastUpdated.classList.toggle('error', isError);
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length);
  if (!lines.length) return { headers: [], rows: [] };

  const headers = lines[0].split(',').map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const cells = line.split(',');
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = (cells[idx] ?? '').trim();
    });
    return row;
  });
  return { headers, rows };
}

function formatValue(key, value) {
  if (value === undefined || value === null || value === '') return '';
  if ([
    'Open',
    'High',
    'Low',
    'Close',
    'UpperMid',
    'ValueUpper',
    'LowerMid',
    'ValueLower',
  ].includes(key)) {
    const num = Number(value);
    if (Number.isFinite(num)) {
      return num.toFixed(3);
    }
  }
  return value;
}

function applyFilters(rows) {
  let filtered = [...rows];
  if (elements.filterUpper.checked) {
    filtered = filtered.filter((row) => row.TouchUpperQ === '1');
  }
  if (elements.filterLower.checked) {
    filtered = filtered.filter((row) => row.TouchLowerQ === '1');
  }
  const limit = Math.max(10, Math.min(Number(elements.rowsLimit.value) || 200, 1000));
  return filtered.slice(-limit);
}

function renderSummary(rows) {
  if (!elements.summaryTemplate) return;
  elements.summary.innerHTML = '';
  if (!rows.length) {
    elements.summary.innerHTML = '<p class="empty">No hay datos.</p>';
    return;
  }
  const latest = rows[rows.length - 1];
  const fragment = elements.summaryTemplate.content.cloneNode(true);
  fragment.querySelectorAll('[data-field]').forEach((node) => {
    const key = node.dataset.field;
    node.textContent = formatValue(key, latest[key]);
    if (key === 'TouchUpperQ' || key === 'TouchLowerQ') {
      node.classList.toggle('active', latest[key] === '1');
    }
  });
  elements.summary.appendChild(fragment);
}

function renderTable(headers, rows) {
  if (!headers.length) {
    elements.tableHead.innerHTML = '';
    elements.tableBody.innerHTML = '<tr><td class="empty">No hay datos.</td></tr>';
    return;
  }

  elements.tableHead.innerHTML = `<tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr>`;

  if (!rows.length) {
    elements.tableBody.innerHTML = '<tr><td class="empty" colspan="12">Sin coincidencias.</td></tr>';
    return;
  }

  const rowsHtml = rows
    .map((row) => {
      const highlightUpper = row.TouchUpperQ === '1';
      const highlightLower = row.TouchLowerQ === '1';
      const classes = [];
      if (highlightUpper) classes.push('highlight-upper');
      if (highlightLower) classes.push('highlight-lower');

      const cells = headers
        .map((header) => {
          let value = row[header] ?? '';
          if (header === 'TouchUpperQ' || header === 'TouchLowerQ') {
            const on = value === '1';
            const extra = header === 'TouchLowerQ' ? ' lower' : '';
            value = `<span class="chip${on ? ' on' : ''}${extra}">${on ? '1' : '0'}</span>`;
          } else {
            value = formatValue(header, value);
          }
          return `<td>${value}</td>`;
        })
        .join('');
      return `<tr class="${classes.join(' ')}">${cells}</tr>`;
    })
    .join('');

  elements.tableBody.innerHTML = rowsHtml;
}

async function loadCsv() {
  try {
    setLastUpdated('Actualizando…');
    const response = await fetch(`${CSV_URL}?_=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const text = await response.text();
    const { headers, rows } = parseCsv(text);
    state.headers = headers;
    state.rows = rows;
    const filtered = applyFilters(rows);
    renderTable(headers, filtered);
    renderSummary(rows);
    const now = new Date();
    setLastUpdated(`Última actualización: ${now.toLocaleString()}`);
  } catch (err) {
    console.error('Error cargando tablaQ.csv', err);
    setLastUpdated(`Error al cargar datos: ${err.message}`, true);
  }
}

function scheduleRefresh() {
  if (state.timerId) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
  if (!elements.autoRefresh.checked) return;
  state.timerId = setInterval(loadCsv, DEFAULT_REFRESH_MS);
}

function init() {
  elements.refreshLabel.textContent = `(${DEFAULT_REFRESH_MS / 1000}s)`;
  elements.reloadBtn.addEventListener('click', loadCsv);
  elements.autoRefresh.addEventListener('change', () => {
    scheduleRefresh();
    if (elements.autoRefresh.checked) {
      loadCsv();
    }
  });

  [elements.filterUpper, elements.filterLower, elements.rowsLimit].forEach((el) => {
    el.addEventListener('change', () => {
      const filtered = applyFilters(state.rows);
      renderTable(state.headers, filtered);
    });
  });

  loadCsv();
  scheduleRefresh();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
