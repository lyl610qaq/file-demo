(() => {
  "use strict";

  const TREE_PAGE_LIMIT = 200;
  const TREE_AUTO_PAGES = 20;
  const FILE_PAGE_BYTES = 65536;
  const FILE_PREVIEW_CHARS = 262144;
  const REQUEST_TIMEOUT_MS = 15000;
  const SOCKET_CONNECT_TIMEOUT_MS = 10000;
  const API_TREE = "/api/tree";
  const API_FILE = "/api/file";
  const TERMINAL_EVENTS = new Set(["run_completed", "run_failed"]);
  const EVENT_TYPES = new Set([
    "run_started",
    "model_call_started",
    "usage_updated",
    "tool_started",
    "tool_finished",
    "assistant_message",
    "run_completed",
    "run_failed",
  ]);

  const state = {
    socket: null,
    runId: null,
    steps: 0,
    running: false,
    resetting: false,
    modelConfigured: false,
    terminalSeen: false,
    treeEntries: [],
    treeCursor: null,
    collapsedPaths: new Set(),
    selectedPath: null,
  };

  const byId = (id) => document.getElementById(id);

  function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function safeString(value, fallback = "") {
    return typeof value === "string" ? value : fallback;
  }

  function safeCount(value) {
    return Number.isSafeInteger(value) && value >= 0 ? String(value) : "不可用";
  }

  function setText(id, value) {
    const node = byId(id);
    if (node) {
      node.textContent = value;
    }
  }

  function setStatus(text, kind) {
    const node = byId("run-status");
    node.textContent = text;
    node.dataset.state = kind;
  }

  function setModelStatus(text, kind) {
    const node = byId("model-status");
    node.textContent = text;
    node.dataset.state = kind;
  }

  function setControlsBusy() {
    const busy = state.running || state.resetting;
    byId("run-button").disabled = busy || !state.modelConfigured;
    byId("task-input").disabled = busy;
    byId("refresh-button").disabled = busy;
    byId("reset-button").disabled = busy;
    byId("confirm-reset").disabled = busy;
  }

  function activatePane(name) {
    document.querySelectorAll("[data-pane]").forEach((node) => {
      node.classList.toggle("active", node.dataset.pane === name);
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      const active = button.dataset.view === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-current", active ? "page" : "false");
    });
  }

  function refreshIcons() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
      document.documentElement.classList.add("icons-ready");
    }
  }

  function createIcon(name) {
    const icon = document.createElement("i");
    icon.dataset.lucide = name;
    icon.setAttribute("aria-hidden", "true");
    return icon;
  }

  function apiMessage(status, errorCode, context) {
    if (status === 409 || errorCode === "WORKSPACE_BUSY") {
      return "工作区正在使用，请稍后重试";
    }
    if (status === 429 || errorCode === "RATE_LIMITED") {
      return "操作过于频繁，请稍后重试";
    }
    if (status === 404) {
      return context === "file" ? "文件不存在或已被移动" : "请求的内容不存在";
    }
    if (status === 415) {
      return "该文件不是可预览的文本";
    }
    if (status === 403) {
      return "当前页面无权执行此操作";
    }
    if (status >= 500) {
      return "服务暂时不可用，请稍后重试";
    }
    return "请求未完成，请重试";
  }

  async function fetchJson(path, options = {}, context = "request") {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(path, {
        credentials: "same-origin",
        ...options,
        headers: {
          Accept: "application/json",
          ...(options.headers || {}),
        },
        signal: controller.signal,
      });
      let payload = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }
      if (!response.ok) {
        const detail = isRecord(payload) && isRecord(payload.detail) ? payload.detail : {};
        const error = new Error("request failed");
        error.userMessage = apiMessage(response.status, safeString(detail.error_code), context);
        error.status = response.status;
        throw error;
      }
      if (!isRecord(payload)) {
        const error = new Error("invalid response");
        error.userMessage = "服务返回了无法识别的数据";
        throw error;
      }
      return payload;
    } catch (error) {
      if (error && error.name === "AbortError") {
        const timeout = new Error("request timed out");
        timeout.userMessage = "请求超时，请稍后重试";
        throw timeout;
      }
      if (error && typeof error.userMessage === "string") {
        throw error;
      }
      const network = new Error("network failed");
      network.userMessage = "无法连接服务，请检查网络后重试";
      throw network;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function validateTreePayload(payload) {
    if (
      !Array.isArray(payload.entries) ||
      typeof payload.has_more !== "boolean" ||
      !(payload.next_cursor === null || typeof payload.next_cursor === "string")
    ) {
      return null;
    }
    const entries = [];
    for (const entry of payload.entries) {
      if (
        !isRecord(entry) ||
        typeof entry.path !== "string" ||
        !entry.path ||
        !["file", "directory"].includes(entry.type)
      ) {
        return null;
      }
      entries.push({
        path: entry.path,
        type: entry.type,
        size: Number.isSafeInteger(entry.size) && entry.size >= 0 ? entry.size : null,
      });
    }
    if (payload.has_more && !payload.next_cursor) {
      return null;
    }
    return {
      entries,
      has_more: payload.has_more,
      next_cursor: payload.next_cursor,
    };
  }

  function treeUrl(cursor) {
    const params = new URLSearchParams({
      path: ".",
      recursive: "true",
      limit: String(TREE_PAGE_LIMIT),
    });
    if (cursor) {
      params.set("cursor", cursor);
    }
    return `${API_TREE}?${params.toString()}`;
  }

  async function loadTree(options = {}) {
    const append = options.append === true;
    if (!append) {
      state.treeEntries = [];
      state.treeCursor = null;
      setText("tree-status", "正在读取文件");
      byId("tree-load-more").hidden = true;
    }

    let cursor = append ? state.treeCursor : null;
    let pages = 0;
    const incoming = [];
    try {
      do {
        const payload = validateTreePayload(await fetchJson(treeUrl(cursor), {}, "tree"));
        if (!payload) {
          throw Object.assign(new Error("invalid tree"), {
            userMessage: "文件列表格式无法识别",
          });
        }
        incoming.push(...payload.entries);
        cursor = payload.next_cursor;
        pages += 1;
        if (!payload.has_more) {
          cursor = null;
          break;
        }
      } while (pages < TREE_AUTO_PAGES);

      const paths = new Set(state.treeEntries.map((entry) => entry.path));
      for (const entry of incoming) {
        if (!paths.has(entry.path)) {
          state.treeEntries.push(entry);
          paths.add(entry.path);
        }
      }
      state.treeCursor = cursor;
      state.treeEntries.sort((left, right) => left.path.localeCompare(right.path, "zh-CN"));
      renderTree();
      setText("tree-summary", `${state.treeEntries.length} 项`);
      setText("tree-status", state.treeEntries.length ? "" : "工作区为空");
      byId("tree-load-more").hidden = state.treeCursor === null;
    } catch (error) {
      setText("tree-status", error.userMessage || "文件列表加载失败");
      if (!append) {
        byId("file-tree").replaceChildren();
        setText("tree-summary", "读取失败");
      }
    }
  }

  function buildTree(entries) {
    const root = { name: "", path: "", type: "directory", children: new Map() };
    for (const entry of entries) {
      const segments = entry.path.split("/").filter(Boolean);
      let parent = root;
      segments.forEach((segment, index) => {
        const path = segments.slice(0, index + 1).join("/");
        const final = index === segments.length - 1;
        if (!parent.children.has(segment)) {
          parent.children.set(segment, {
            name: segment,
            path,
            type: final ? entry.type : "directory",
            size: final ? entry.size : null,
            children: new Map(),
          });
        }
        const node = parent.children.get(segment);
        if (final) {
          node.type = entry.type;
          node.size = entry.size;
        }
        parent = node;
      });
    }
    return root;
  }

  function sortedChildren(node) {
    return Array.from(node.children.values()).sort((left, right) => {
      if (left.type !== right.type) {
        return left.type === "directory" ? -1 : 1;
      }
      return left.name.localeCompare(right.name, "zh-CN");
    });
  }

  function renderTreeGroup(node) {
    const group = document.createElement("ul");
    group.className = "tree-group";
    group.setAttribute("role", "group");

    for (const child of sortedChildren(node)) {
      const item = document.createElement("li");
      item.className = "tree-item";
      item.setAttribute("role", "treeitem");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "file-row";
      button.title = child.path;

      const chevron = document.createElement("span");
      chevron.className = "tree-chevron";
      chevron.textContent = child.type === "directory" ? "▾" : "";
      chevron.setAttribute("aria-hidden", "true");
      button.append(chevron);
      button.append(createIcon(child.type === "directory" ? "folder" : fileIcon(child.path)));

      const label = document.createElement("span");
      label.className = "file-label";
      label.textContent = child.name;
      button.append(label);

      if (child.type === "directory") {
        const collapsed = state.collapsedPaths.has(child.path);
        item.setAttribute("aria-expanded", collapsed ? "false" : "true");
        chevron.textContent = collapsed ? "›" : "▾";
        button.setAttribute("aria-label", `${collapsed ? "展开" : "折叠"}目录 ${child.name}`);
        button.addEventListener("click", () => {
          if (state.collapsedPaths.has(child.path)) {
            state.collapsedPaths.delete(child.path);
          } else {
            state.collapsedPaths.add(child.path);
          }
          renderTree();
        });
      } else {
        button.setAttribute("aria-label", `预览文件 ${child.name}`);
        if (state.selectedPath === child.path) {
          button.classList.add("active");
        }
        button.addEventListener("click", () => openFile(child.path));
      }

      item.append(button);
      if (child.type === "directory" && child.children.size) {
        item.append(renderTreeGroup(child));
      }
      group.append(item);
    }
    return group;
  }

  function fileIcon(path) {
    const lower = path.toLowerCase();
    if (lower.endsWith(".md") || lower.endsWith(".txt")) {
      return "file-text";
    }
    if (lower.endsWith(".csv") || lower.endsWith(".tsv")) {
      return "table";
    }
    if (lower.endsWith(".json") || lower.endsWith(".js") || lower.endsWith(".py")) {
      return "file-code";
    }
    return "file";
  }

  function renderTree() {
    const container = byId("file-tree");
    container.replaceChildren();
    const tree = buildTree(state.treeEntries);
    container.append(renderTreeGroup(tree));
    refreshIcons();
  }

  function validateFilePayload(payload) {
    if (
      typeof payload.path !== "string" ||
      typeof payload.content !== "string" ||
      typeof payload.has_more !== "boolean" ||
      !(payload.next_cursor === null || typeof payload.next_cursor === "string")
    ) {
      return null;
    }
    if (payload.has_more && !payload.next_cursor) {
      return null;
    }
    return {
      path: payload.path,
      content: payload.content,
      has_more: payload.has_more,
      next_cursor: payload.next_cursor,
      encoding: safeString(payload.encoding, "文本"),
    };
  }

  async function openFile(path) {
    state.selectedPath = path;
    renderTree();
    setText("viewer-path", path);
    setText("viewer-meta", "正在读取");
    setText("file-content", "");
    activatePane("task");

    let cursor = null;
    let content = "";
    let hasMore = false;
    let encoding = "文本";
    try {
      do {
        const params = new URLSearchParams({
          path,
          limit: String(FILE_PAGE_BYTES),
        });
        if (cursor) {
          params.set("cursor", cursor);
        }
        const page = validateFilePayload(
          await fetchJson(`${API_FILE}?${params.toString()}`, {}, "file"),
        );
        if (!page) {
          throw Object.assign(new Error("invalid file"), {
            userMessage: "文件内容格式无法识别",
          });
        }
        content += page.content;
        cursor = page.next_cursor;
        hasMore = page.has_more;
        encoding = page.encoding;
      } while (hasMore && cursor && content.length < FILE_PREVIEW_CHARS);

      if (!content) {
        setText("file-content", "空文件");
        setText("viewer-meta", encoding);
      } else {
        setText("file-content", content);
        setText(
          "viewer-meta",
          hasMore ? `${encoding} · 内容较长，已显示前 ${content.length} 个字符` : `${encoding} · ${content.length} 个字符`,
        );
      }
    } catch (error) {
      setText("file-content", error.userMessage || "文件预览失败");
      setText("viewer-meta", "读取失败");
    }
  }

  function resetTraceView() {
    state.steps = 0;
    state.terminalSeen = false;
    byId("trace-list").replaceChildren();
    setText("step-count", "0");
    byId("trace-empty").hidden = false;
  }

  function eventIsValid(event) {
    if (!isRecord(event) || !EVENT_TYPES.has(event.type)) {
      return false;
    }
    if (
      event.run_id !== undefined &&
      (typeof event.run_id !== "string" || !/^[A-Za-z0-9-]{1,128}$/.test(event.run_id))
    ) {
      return false;
    }
    if (event.type === "usage_updated") {
      return Number.isSafeInteger(event.model_calls) && isRecord(event.usage);
    }
    if (event.type === "assistant_message") {
      return typeof event.content === "string";
    }
    if (event.type === "tool_started") {
      return Number.isSafeInteger(event.step) && typeof event.tool === "string" && isRecord(event.args);
    }
    if (event.type === "tool_finished") {
      return (
        Number.isSafeInteger(event.step) &&
        typeof event.tool === "string" &&
        typeof event.ok === "boolean" &&
        typeof event.result_summary === "string"
      );
    }
    if (TERMINAL_EVENTS.has(event.type)) {
      return typeof event.message === "string";
    }
    return true;
  }

  function traceLabel(event) {
    const labels = {
      run_started: "任务已开始",
      model_call_started: "正在调用模型",
      tool_started: "开始执行工具",
      tool_finished: event.ok ? "工具执行完成" : "工具执行失败",
      run_completed: "任务已完成",
      run_failed: "任务未完成",
    };
    return labels[event.type] || "运行事件";
  }

  function toolDetail(event) {
    if (event.type === "model_call_started") {
      return `第 ${safeCount(event.call)} 次调用`;
    }
    if (event.type === "tool_started") {
      let args = "";
      try {
        args = JSON.stringify(event.args);
      } catch {
        args = "{}";
      }
      return `${event.tool} ${args}`;
    }
    if (event.type === "tool_finished") {
      return `${event.tool} · ${event.result_summary}`;
    }
    if (event.type === "run_failed") {
      return runFailureMessage(event.message);
    }
    if (event.type === "run_completed") {
      return "运行轨迹已保存";
    }
    return "";
  }

  function appendStep(event) {
    const item = document.createElement("li");
    item.className = "trace-item";
    item.dataset.kind = event.type === "run_failed"
      ? "error"
      : event.type.startsWith("tool_")
        ? "tool"
        : event.type === "model_call_started"
          ? "warning"
          : "model";

    const title = document.createElement("strong");
    title.textContent = traceLabel(event);
    const detail = document.createElement("code");
    detail.textContent = toolDetail(event);
    item.append(title, detail);
    byId("trace-list").append(item);
    byId("trace-empty").hidden = true;
    state.steps += 1;
    setText("step-count", String(state.steps));
  }

  function runFailureMessage(message) {
    const messages = {
      "Invalid run request": "任务请求无效",
      "Model is not configured": "模型尚未配置",
      "Server is busy": "服务繁忙，请稍后重试",
      "Run timed out": "任务运行超时",
      "Run failed": "任务运行失败，请重试",
      "Model call limit reached": "模型调用次数已达上限",
      "Response tool call limit reached": "单次工具调用过多，任务已停止",
      "Total tool call limit reached": "工具调用次数已达上限",
      "Repeated tool call limit reached": "检测到重复操作，任务已停止",
      "Tool result byte budget exceeded": "工具结果过大，任务已停止",
    };
    if (messages[message]) {
      return messages[message];
    }
    if (message.startsWith("Model call failed:")) {
      return "模型服务调用失败";
    }
    return "任务运行失败，请重试";
  }

  function enableTraceDownload(runId) {
    state.runId = runId;
    const link = byId("trace-download");
    link.href = `/api/runs/${encodeURIComponent(runId)}/trace`;
    link.classList.remove("is-disabled");
    link.setAttribute("aria-disabled", "false");
  }

  function disableTraceDownload() {
    state.runId = null;
    const link = byId("trace-download");
    link.href = "#";
    link.classList.add("is-disabled");
    link.setAttribute("aria-disabled", "true");
  }

  function finishRun(message, failed) {
    if (!state.running) {
      return;
    }
    state.running = false;
    setControlsBusy();
    setStatus(failed ? message : "完成", failed ? "failed" : "completed");
    setText("reply-state", failed ? "未完成" : "已完成");
    loadTree();
  }

  function handleEvent(event) {
    if (!eventIsValid(event) || state.terminalSeen) {
      return;
    }
    if (event.run_id) {
      enableTraceDownload(event.run_id);
    }

    if (event.type === "usage_updated") {
      setText("usage-calls", safeCount(event.model_calls));
      setText("usage-total", safeCount(event.usage.total_tokens));
      return;
    }
    if (event.type === "assistant_message") {
      setText("assistant-output", event.content || "模型未返回文字内容");
      return;
    }

    appendStep(event);
    if (event.type === "run_completed") {
      state.terminalSeen = true;
      if (!byId("assistant-output").textContent.trim()) {
        setText("assistant-output", event.message || "任务已完成");
      }
      finishRun("完成", false);
      state.socket?.close(1000, "completed");
    } else if (event.type === "run_failed") {
      state.terminalSeen = true;
      const message = runFailureMessage(event.message);
      setText("assistant-output", message);
      finishRun(message, true);
      state.socket?.close(1000, "failed");
    }
  }

  function socketFailureMessage(closeEvent) {
    if (closeEvent && closeEvent.code === 1013) {
      return "服务繁忙，请稍后重试";
    }
    if (closeEvent && closeEvent.code === 1008) {
      return "任务请求被拒绝";
    }
    return "连接已中断，请重试";
  }

  function startRun(task) {
    const trimmed = task.trim();
    if (!trimmed || state.running || state.resetting || !state.modelConfigured) {
      return;
    }

    state.running = true;
    resetTraceView();
    disableTraceDownload();
    setControlsBusy();
    setText("assistant-output", "正在运行");
    setText("reply-state", "处理中");
    setStatus("正在连接", "running");

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/ws/agent`);
    state.socket = socket;
    const connectTimer = window.setTimeout(() => {
      if (state.running && socket.readyState === WebSocket.CONNECTING) {
        socket.close();
        setText("assistant-output", "连接超时，请重试");
        finishRun("连接超时", true);
      }
    }, SOCKET_CONNECT_TIMEOUT_MS);

    socket.addEventListener("open", () => {
      window.clearTimeout(connectTimer);
      if (socket !== state.socket || !state.running) {
        socket.close();
        return;
      }
      setStatus("运行中", "running");
      socket.send(JSON.stringify({ type: "run", task: trimmed }));
    });

    socket.addEventListener("message", (message) => {
      if (socket !== state.socket || typeof message.data !== "string") {
        return;
      }
      try {
        handleEvent(JSON.parse(message.data));
      } catch {
        setText("assistant-output", "服务返回了无法识别的事件");
        finishRun("事件格式错误", true);
        socket.close(1000, "invalid-event");
      }
    });

    socket.addEventListener("error", () => {
      window.clearTimeout(connectTimer);
    });

    socket.addEventListener("close", (event) => {
      window.clearTimeout(connectTimer);
      if (socket !== state.socket) {
        return;
      }
      state.socket = null;
      if (state.running && !state.terminalSeen) {
        const message = socketFailureMessage(event);
        setText("assistant-output", message);
        appendStep({ type: "run_failed", message });
        finishRun(message, true);
      }
    });
  }

  async function loadMeta() {
    try {
      const meta = await fetchJson("/api/meta");
      if (typeof meta.model !== "string" || typeof meta.configured !== "boolean") {
        throw Object.assign(new Error("invalid meta"), {
          userMessage: "模型状态格式无法识别",
        });
      }
      state.modelConfigured = meta.configured;
      setModelStatus(
        meta.configured ? `${meta.model} · 已连接` : `${meta.model} · 未配置`,
        meta.configured ? "ready" : "warning",
      );
      if (!meta.configured) {
        setStatus("模型未配置", "failed");
        setText("assistant-output", "请先在服务端配置模型");
      }
    } catch (error) {
      state.modelConfigured = false;
      setModelStatus("模型状态不可用", "warning");
      setStatus(error.userMessage || "连接服务失败", "failed");
    } finally {
      setControlsBusy();
    }
  }

  async function resetWorkspace() {
    if (state.running || state.resetting) {
      return;
    }
    state.resetting = true;
    setControlsBusy();
    setStatus("正在重置工作区", "running");
    byId("reset-dialog").close();
    try {
      await fetchJson(
        "/api/reset",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        },
        "reset",
      );
      state.selectedPath = null;
      state.collapsedPaths.clear();
      setText("viewer-path", "文件预览");
      setText("viewer-meta", "");
      setText("file-content", "从左侧选择文件");
      setText("assistant-output", "工作区已重置");
      setStatus("工作区已重置", "completed");
      await loadTree();
    } catch (error) {
      const message = error.userMessage || "工作区重置失败";
      setStatus(message, "failed");
      setText("assistant-output", message);
    } finally {
      state.resetting = false;
      setControlsBusy();
    }
  }

  function bindInterface() {
    byId("task-form").addEventListener("submit", (event) => {
      event.preventDefault();
      startRun(byId("task-input").value);
    });
    byId("refresh-button").addEventListener("click", () => loadTree());
    byId("tree-load-more").addEventListener("click", () => loadTree({ append: true }));
    byId("reset-button").addEventListener("click", () => byId("reset-dialog").showModal());
    byId("cancel-reset").addEventListener("click", () => byId("reset-dialog").close());
    byId("confirm-reset").addEventListener("click", resetWorkspace);
    byId("trace-download").addEventListener("click", (event) => {
      if (!state.runId) {
        event.preventDefault();
      }
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => activatePane(button.dataset.view));
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    bindInterface();
    refreshIcons();
    disableTraceDownload();
    setControlsBusy();
    await Promise.all([loadMeta(), loadTree()]);
  });
})();
