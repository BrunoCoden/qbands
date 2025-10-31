(() => {
  if (window.Chart) {
    const zoomPlugin = window["chartjs-plugin-zoom"];
    if (zoomPlugin) {
      Chart.register(zoomPlugin);
    }
    Chart.defaults.color = "#9fb0c2";
    Chart.defaults.font.family = "'Segoe UI', 'Tahoma', sans-serif";
  }
  let originalTrades = [];
  let filteredTrades = [];
  let headers = [];
  let currentSort = { key: null, direction: 1 };
  let chartCumulative = null;
  let chartDistribution = null;
  const AUTO_REFRESH_INTERVAL_MS = 30_000;
  let autoRefreshTimer = null;

  const els = {
    loadBtn: document.getElementById("load-btn"),
    pathInput: document.getElementById("csv-path"),
    fileInput: document.getElementById("file-input"),
    summaryPath: document.getElementById("summary-path"),
    autoRefresh: document.getElementById("auto-refresh"),
    statTrades: document.getElementById("stat-trades"),
    statWinrate: document.getElementById("stat-winrate"),
    statTotalPnL: document.getElementById("stat-total-pnl"),
    statAvgPnL: document.getElementById("stat-avg-pnl"),
    statMaxPnL: document.getElementById("stat-max-pnl"),
    statMinPnL: document.getElementById("stat-min-pnl"),
    filterSide: document.getElementById("filter-side"),
    filterContext: document.getElementById("filter-context"),
    filterExit: document.getElementById("filter-exit"),
    filterStart: document.getElementById("filter-start"),
    filterEnd: document.getElementById("filter-end"),
    resetFilters: document.getElementById("reset-filters"),
    tableHead: document.querySelector("#trades-table thead"),
    tableBody: document.querySelector("#trades-table tbody"),
    tableSearch: document.getElementById("table-search"),
    resetCum: document.querySelector('[data-reset="cumulative"]'),
    resetDist: document.querySelector('[data-reset="distribution"]'),
    pageSize: document.getElementById("page-size"),
    prevPage: document.getElementById("prev-page"),
    nextPage: document.getElementById("next-page"),
    paginationInfo: document.getElementById("pagination-info"),
  };

  const pagination = {
    page: 1,
    pageSize: Number(els.pageSize?.value) || 25,
  };

  const summaryEls = {
    loggedAt: document.getElementById("live-logged-at"),
    candles: document.getElementById("live-candles"),
    trades: document.getElementById("live-trades"),
    wins: document.getElementById("live-wins"),
    losses: document.getElementById("live-losses"),
    winRate: document.getElementById("live-win-rate"),
    totalReturn: document.getElementById("live-total-return"),
    avgReturn: document.getElementById("live-avg-return"),
    openPositions: document.getElementById("live-open-positions"),
    pendingOrders: document.getElementById("live-pending-orders"),
  };

  function parseLine(line) {
    const result = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"') {
        const nextChar = line[i + 1];
        if (inQuotes && nextChar === '"') {
          current += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === "," && !inQuotes) {
        result.push(current);
        current = "";
      } else {
        current += char;
      }
    }
    result.push(current);
    return result.map((v) => v.trim());
  }

  function parseCSV(text) {
    const rows = text.trim().split(/\r?\n/);
    if (!rows.length) {
      return { headers: [], rows: [] };
    }
    const parsedHeaders = parseLine(rows.shift());
    const parsedRows = rows
      .filter((line) => line.trim().length > 0)
      .map((line) => {
        const values = parseLine(line);
        const row = {};
        parsedHeaders.forEach((header, idx) => {
          row[header] = values[idx] ?? "";
        });
        return row;
      });
    return { headers: parsedHeaders, rows: parsedRows };
  }

  function toNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : 0;
  }

  function toDate(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? null : date;
  }

  function enrichTrades(rows) {
    return rows.map((row, idx) => {
      const entryDate = toDate(row.entry_time);
      const exitDate = toDate(row.exit_time);
      const pnl = toNumber(row.pnl_pct);
      return {
        ...row,
        _index: idx + 1,
        _entryDate: entryDate,
        _exitDate: exitDate,
        _pnl: pnl,
        _profitTarget: toNumber(row.profit_target_pct),
        _stopPct: toNumber(row.stop_pct),
        _barsHeld: toNumber(row.bars_held),
      };
    });
  }

  function updateStats(trades) {
    if (!trades.length) {
      els.statTrades.textContent = "0";
      els.statWinrate.textContent = "0%";
      els.statTotalPnL.textContent = "0%";
      els.statAvgPnL.textContent = "0%";
      els.statMaxPnL.textContent = "0%";
      els.statMinPnL.textContent = "0%";
      return;
    }

    const wins = trades.filter((t) => t.exit_reason === "target_hit");
    const losses = trades.filter((t) => t.exit_reason === "stop_hit");
    const totalPnL = trades.reduce((acc, t) => acc + t._pnl, 0);
    const avgPnL = totalPnL / trades.length;
    const maxPnL = Math.max(...trades.map((t) => t._pnl));
    const minPnL = Math.min(...trades.map((t) => t._pnl));
    const winRate = trades.length ? (wins.length / trades.length) * 100 : 0;

    const fmt = (value, digits = 2) => `${value.toFixed(digits)}%`;

    els.statTrades.textContent = `${trades.length}`;
    els.statWinrate.textContent = fmt(winRate);
    els.statTotalPnL.textContent = fmt(totalPnL);
    els.statAvgPnL.textContent = fmt(avgPnL);
    els.statMaxPnL.textContent = fmt(maxPnL);
    els.statMinPnL.textContent = fmt(minPnL);
  }

  function buildTableHead(headersList) {
    els.tableHead.innerHTML = "";
    if (!headersList.length) return;

    const tr = document.createElement("tr");
    headersList.forEach((header) => {
      const th = document.createElement("th");
      th.textContent = header;
      const indicator = document.createElement("span");
      indicator.className = "sort-indicator";
      indicator.textContent = "";
      th.appendChild(indicator);
      th.addEventListener("click", () => onSort(header));
      tr.appendChild(th);
    });
    els.tableHead.appendChild(tr);
  }

  function updateSortIndicators(activeHeader, direction) {
    const indicators = els.tableHead.querySelectorAll(".sort-indicator");
    indicators.forEach((indicator) => {
      const parentText = indicator.parentElement.textContent.replace(indicator.textContent, "").trim();
      if (activeHeader && parentText === activeHeader) {
        indicator.textContent = direction > 0 ? "▲" : "▼";
      } else {
        indicator.textContent = "";
      }
    });
  }

  function getTotalPages() {
    if (!filteredTrades.length) {
      return 1;
    }
    return Math.ceil(filteredTrades.length / pagination.pageSize);
  }

  function ensurePageBounds() {
    const totalPages = getTotalPages();
    if (pagination.page > totalPages) {
      pagination.page = totalPages;
    }
    if (pagination.page < 1) {
      pagination.page = 1;
    }
    if (!filteredTrades.length) {
      pagination.page = 1;
    }
    return totalPages;
  }

  function getPagedTrades() {
    if (!filteredTrades.length) {
      return [];
    }
    const startIdx = (pagination.page - 1) * pagination.pageSize;
    return filteredTrades.slice(startIdx, startIdx + pagination.pageSize);
  }

  function updatePaginationControls() {
    const totalTrades = filteredTrades.length;
    const totalPages = ensurePageBounds();
    const currentPage = totalTrades ? pagination.page : 0;
    if (els.paginationInfo) {
      const totalText = totalTrades ? totalPages : 0;
      els.paginationInfo.textContent = `Página ${currentPage} de ${totalText} (${totalTrades} trades)`;
    }
    if (els.prevPage) {
      els.prevPage.disabled = currentPage <= 1;
    }
    if (els.nextPage) {
      els.nextPage.disabled = !totalTrades || currentPage >= totalPages;
    }
    if (els.pageSize && !Number.isNaN(pagination.pageSize)) {
      els.pageSize.value = String(pagination.pageSize);
    }
  }

  function renderTable() {
    els.tableBody.innerHTML = "";
    if (!headers.length) {
      updatePaginationControls();
      return;
    }

    const pageRows = getPagedTrades();
    const fragment = document.createDocumentFragment();
    pageRows.forEach((row) => {
      const tr = document.createElement("tr");
      headers.forEach((header) => {
        const td = document.createElement("td");
        const value = row[header];
        const numericHeaders = ["pnl_pct", "profit_target_pct", "stop_pct"];
        if (numericHeaders.includes(header)) {
          const num = Number(value);
          if (header === "pnl_pct") {
            td.classList.add(num >= 0 ? "pn-positive" : "pn-negative");
          }
          td.textContent = Number.isFinite(num) ? num.toFixed(2) : value ?? "";
        } else {
          td.textContent = value ?? "";
        }
        tr.appendChild(td);
      });
      fragment.appendChild(tr);
    });
    els.tableBody.appendChild(fragment);
    updatePaginationControls();
  }

  function buildCumulativeData(trades) {
    let cumulative = 0;
    const labels = [];
    const data = [];
    trades.forEach((trade, idx) => {
      cumulative += trade._pnl;
      labels.push(`#${idx + 1}`);
      data.push(Number(cumulative.toFixed(4)));
    });
    return { labels, data };
  }

  function buildDistributionData(trades) {
    const labels = trades.map((trade, idx) => `#${idx + 1}`);
    const data = trades.map((trade) => Number(trade._pnl.toFixed(4)));
    return { labels, data };
  }

  function renderCharts(trades) {
    const ctxCum = document.getElementById("chart-cumulative");
    const ctxDist = document.getElementById("chart-distribution");

    const cum = buildCumulativeData(trades);
    const dist = buildDistributionData(trades);

    if (chartCumulative) {
      if (chartCumulative.resetZoom) {
        chartCumulative.resetZoom();
      }
      chartCumulative.data.labels = cum.labels;
      chartCumulative.data.datasets[0].data = cum.data;
      chartCumulative.update();
    } else {
      chartCumulative = new Chart(ctxCum, {
        type: "line",
        data: {
          labels: cum.labels,
          datasets: [
            {
              label: "Cumulative PnL %",
              data: cum.data,
              borderColor: "#4fe1a3",
              backgroundColor: "rgba(79, 225, 163, 0.1)",
              tension: 0.25,
              fill: true,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: { color: "#9fb0c2" },
              grid: { color: "rgba(255,255,255,0.05)" },
            },
            y: {
              ticks: { color: "#9fb0c2" },
              grid: { color: "rgba(255,255,255,0.05)" },
            },
          },
          plugins: {
            legend: {
              labels: { color: "#e1eaf3" },
            },
            zoom: {
              limits: {
                x: { min: 0 },
              },
              pan: {
                enabled: true,
                mode: "xy",
                modifierKey: "shift",
              },
              zoom: {
                wheel: { enabled: true },
                pinch: { enabled: true },
                mode: "xy",
              },
            },
          },
        },
      });
    }

    if (chartDistribution) {
      if (chartDistribution.resetZoom) {
        chartDistribution.resetZoom();
      }
      chartDistribution.data.labels = dist.labels;
      chartDistribution.data.datasets[0].data = dist.data;
      chartDistribution.data.datasets[0].backgroundColor = dist.data.map((value) =>
        value >= 0 ? "rgba(79,225,163,0.7)" : "rgba(255,106,106,0.7)"
      );
      chartDistribution.update();
    } else {
      chartDistribution = new Chart(ctxDist, {
        type: "bar",
        data: {
          labels: dist.labels,
          datasets: [
            {
              label: "PnL por trade (%)",
              data: dist.data,
              backgroundColor: dist.data.map((value) => (value >= 0 ? "rgba(79,225,163,0.7)" : "rgba(255,106,106,0.7)")),
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: { color: "#9fb0c2" },
              grid: { display: false },
            },
            y: {
              ticks: { color: "#9fb0c2" },
              grid: { color: "rgba(255,255,255,0.05)" },
            },
          },
          plugins: {
            legend: {
              labels: { color: "#e1eaf3" },
            },
            zoom: {
              limits: {
                x: { min: 0 },
              },
              pan: {
                enabled: true,
                mode: "xy",
                modifierKey: "shift",
              },
              zoom: {
                wheel: { enabled: true },
                pinch: { enabled: true },
                mode: "y",
              },
            },
          },
        },
      });
    }
  }

  function applyFilters() {
    const side = els.filterSide.value;
    const context = els.filterContext.value;
    const exit = els.filterExit.value;
    const start = els.filterStart.value ? new Date(els.filterStart.value) : null;
    const end = els.filterEnd.value ? new Date(els.filterEnd.value) : null;
    const search = els.tableSearch.value.trim().toLowerCase();

    filteredTrades = originalTrades.filter((trade) => {
      if (side && trade.side !== side) return false;
      if (context && trade.context !== context) return false;
      if (exit && trade.exit_reason !== exit) return false;
      if (start && trade._entryDate && trade._entryDate < start) return false;
      if (end && trade._entryDate && trade._entryDate > end) return false;
      if (search) {
        const haystack = headers.map((h) => `${trade[h] ?? ""}`.toLowerCase()).join(" ");
        if (!haystack.includes(search)) return false;
      }
      return true;
    });

    pagination.page = 1;
    applySorting();
    updateSortIndicators(currentSort.key, currentSort.direction);
    updateStats(filteredTrades);
    renderTable();
    renderCharts(filteredTrades);
  }

  function applySorting() {
    if (!currentSort.key) {
      filteredTrades.sort((a, b) => (a._index || 0) - (b._index || 0));
      return;
    }
    const { key, direction } = currentSort;
    filteredTrades.sort((a, b) => {
      const valA = a[key];
      const valB = b[key];
      const numA = Number(valA);
      const numB = Number(valB);
      if (Number.isFinite(numA) && Number.isFinite(numB)) {
        return (numA - numB) * direction;
      }
      const strA = (valA ?? "").toString().toLowerCase();
      const strB = (valB ?? "").toString().toLowerCase();
      if (strA < strB) return -1 * direction;
      if (strA > strB) return 1 * direction;
      return 0;
    });
  }

  function onSort(header) {
    if (currentSort.key === header) {
      currentSort.direction *= -1;
    } else {
      currentSort = { key: header, direction: 1 };
    }
    updateSortIndicators(header, currentSort.direction);
    pagination.page = 1;
    applySorting();
    renderTable();
    renderCharts(filteredTrades);
  }

  function resetFilters() {
    els.filterSide.value = "";
    els.filterContext.value = "";
    els.filterExit.value = "";
    els.filterStart.value = "";
    els.filterEnd.value = "";
    els.tableSearch.value = "";
    currentSort = { key: null, direction: 1 };
    updateSortIndicators("", 1);
    applyFilters();
  }

  async function fetchWithCache(path) {
    if (!path) return "";
    const isHttp = /^https?:/i.test(path);
    let url = path;
    if (isHttp) {
      const u = new URL(path);
      u.searchParams.set("_", Date.now().toString());
      url = u.toString();
    } else {
      const hasQuery = path.includes("?");
      const cachePart = `${hasQuery ? "&" : "?"}_=${Date.now()}`;
      url = `${path}${cachePart}`;
    }
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`No se pudo cargar ${path} (${response.status})`);
    }
    return response.text();
  }

  async function loadTradesFromPath(path, { silent = false } = {}) {
    if (!path) return;
    try {
      const text = await fetchWithCache(path);
      processCSV(text);
    } catch (error) {
      if (!silent) {
        alert(error.message);
      }
      console.error(error);
    }
  }

  async function loadSummary(path, { silent = false } = {}) {
    if (!path) return;
    try {
      const text = await fetchWithCache(path);
      const trimmed = text.trim();
      if (!trimmed) return;
      const lines = trimmed.split(/\r?\n/);
      if (lines.length < 2) return;
      const headersList = parseLine(lines[0]);
      const values = parseLine(lines[lines.length - 1]);
      const row = {};
      headersList.forEach((header, idx) => {
        row[header] = values[idx] ?? "";
      });
      updateLiveSummary(row);
    } catch (error) {
      if (!silent) {
        console.error(error);
      }
    }
  }

  async function loadFromPath() {
    const path = els.pathInput.value.trim();
    if (!path) return;
    await loadTradesFromPath(path);
    const summaryPath = els.summaryPath?.value.trim();
    if (summaryPath) {
      await loadSummary(summaryPath, { silent: true });
    }
  }

  function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      processCSV(event.target.result);
    };
    reader.onerror = () => {
      alert("No se pudo leer el archivo seleccionado.");
    };
    reader.readAsText(file);
  }

  function processCSV(text) {
    const parsed = parseCSV(text);
    headers = parsed.headers;
    originalTrades = enrichTrades(parsed.rows);
    filteredTrades = [...originalTrades];
    currentSort = { key: null, direction: 1 };
    pagination.page = 1;
    if (els.pageSize) {
      const size = Number(els.pageSize.value);
      if (Number.isFinite(size) && size > 0) {
        pagination.pageSize = size;
      }
    }

    buildTableHead(headers);
    updateSortIndicators("", 1);
    applyFilters();
  }

  function updateLiveSummary(row) {
    if (!row || !summaryEls.loggedAt) {
      return;
    }
    const fmtPerc = (value) => {
      const num = Number(value);
      if (!Number.isFinite(num)) return "-";
      return `${num.toFixed(2)}%`;
    };
    const set = (el, value) => {
      if (!el) return;
      el.textContent = value ?? "-";
    };
    set(summaryEls.loggedAt, row.logged_at || row.loggedAt || "-");
    set(summaryEls.candles, row.candles);
    set(summaryEls.trades, row.trades);
    set(summaryEls.wins, row.wins);
    set(summaryEls.losses, row.losses);
    set(summaryEls.winRate, fmtPerc(row.win_rate));
    set(summaryEls.totalReturn, fmtPerc(row.total_return_pct));
    set(summaryEls.avgReturn, fmtPerc(row.avg_return_pct));
    set(summaryEls.openPositions, row.open_positions);
    set(summaryEls.pendingOrders, row.pending_orders);
  }

  async function runAutoRefresh({ immediate = false } = {}) {
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    if (!els.autoRefresh || !els.autoRefresh.checked) {
      return;
    }
    const tick = async () => {
      const tradesPath = els.pathInput?.value.trim();
      if (tradesPath) {
        await loadTradesFromPath(tradesPath, { silent: true });
      }
      const summaryPath = els.summaryPath?.value.trim();
      if (summaryPath) {
        await loadSummary(summaryPath, { silent: true });
      }
    };
    if (immediate) {
      await tick();
    }
    autoRefreshTimer = setInterval(tick, AUTO_REFRESH_INTERVAL_MS);
  }

  function attachEventListeners() {
    if (els.loadBtn) {
      els.loadBtn.addEventListener("click", async () => {
        await loadFromPath();
        runAutoRefresh();
      });
    }
    if (els.fileInput) {
      els.fileInput.addEventListener("change", (event) => {
        const [file] = event.target.files;
        if (file) {
          loadFromFile(file);
        }
        event.target.value = "";
      });
    }
    if (els.autoRefresh) {
      els.autoRefresh.addEventListener("change", () => {
        runAutoRefresh({ immediate: true });
      });
    }
    [
      els.filterSide,
      els.filterContext,
      els.filterExit,
      els.filterStart,
      els.filterEnd,
    ].forEach((element) => {
      if (element) {
        element.addEventListener("change", applyFilters);
      }
    });
    if (els.tableSearch) {
      els.tableSearch.addEventListener("input", () => {
        applyFilters();
      });
    }
    if (els.resetFilters) {
      els.resetFilters.addEventListener("click", resetFilters);
    }
    if (els.pageSize) {
      els.pageSize.addEventListener("change", (event) => {
        const value = Number(event.target.value);
        if (Number.isFinite(value) && value > 0) {
          pagination.pageSize = value;
          pagination.page = 1;
          renderTable();
        }
      });
    }
    if (els.prevPage) {
      els.prevPage.addEventListener("click", () => {
        if (pagination.page > 1) {
          pagination.page -= 1;
          renderTable();
        }
      });
    }
    if (els.nextPage) {
      els.nextPage.addEventListener("click", () => {
        const totalPages = getTotalPages();
        if (pagination.page < totalPages) {
          pagination.page += 1;
          renderTable();
        }
      });
    }
    if (els.resetCum) {
      els.resetCum.addEventListener("click", () => {
        if (chartCumulative?.resetZoom) {
          chartCumulative.resetZoom();
        }
      });
    }
    if (els.resetDist) {
      els.resetDist.addEventListener("click", () => {
        if (chartDistribution?.resetZoom) {
          chartDistribution.resetZoom();
        }
      });
    }
  }

  function init() {
    attachEventListeners();
    applyFilters();
    const tradesPath = els.pathInput?.value.trim();
    const summaryPath = els.summaryPath?.value.trim();
    if (tradesPath) {
      loadTradesFromPath(tradesPath, { silent: true }).finally(() => {
        if (summaryPath) {
          loadSummary(summaryPath, { silent: true });
        }
        runAutoRefresh({ immediate: false });
      });
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
