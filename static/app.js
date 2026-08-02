(() => {
  "use strict";

  const TREE_PAGE_LIMIT = 200;
  const TREE_INITIAL_PAGES = 20;
  const TREE_MAX_PAGES = 25;
  const TREE_MAX_ENTRIES = 5000;
  const TREE_MAX_JSON_BYTES = 2 * 1024 * 1024;
  const FILE_PAGE_BYTES = 65536;
  const FILE_MAX_PAGE_REQUESTS = 1024;
  const FILE_MAX_PREVIEW_BYTES = 1024 * 1024;
  const MAX_JSON_RESPONSE_BYTES = 2 * 1024 * 1024;
  const MAX_SOCKET_EVENT_CHARS = 256 * 1024;
  const MAX_EVENT_STRING_CHARS = 64 * 1024;
  const MAX_TOOL_EVENT_CHARS = 64 * 1024;
  const REQUEST_TIMEOUT_MS = 15000;
  const SOCKET_CONNECT_TIMEOUT_MS = 10000;
  const RUN_WATCHDOG_BUFFER_MS = 5000;
  const RUN_WATCHDOG_MIN_MS = 6000;
  const RUN_WATCHDOG_MAX_MS = 3605000;
  const API_TREE = "/api/tree";
  const API_FILE = "/api/file";
  const UTF8_ENCODER = new TextEncoder();
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
    connectionTimer: null,
    runWatchdog: null,
    runId: null,
    activeRunId: null,
    steps: 0,
    running: false,
    resetting: false,
    modelConfigured: false,
    maxRunSeconds: null,
    maxReadBytes: 1024,
    terminalSeen: false,
    lastModelCall: -1,
    lastUsageModelCalls: -1,
    lastToolStep: -1,
    activeToolStep: null,
    activeToolName: null,
    treeEntries: [],
    treeCursor: null,
    treeSeenCursors: new Set(),
    treePages: 0,
    treeJsonBytes: 0,
    treeLoading: false,
    treeLimitReached: false,
    collapsedPaths: new Set(),
    selectedPath: null,
    fileController: null,
    fileRequestToken: 0,
    lastTreeFocusPath: null,
    activePane: "task",
    treeGroupSequence: 0,
    usage: {
      modelCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    },
  };

  const byId = (id) => document.getElementById(id);
  const mobileLayout = window.matchMedia("(max-width: 860px)");

  function isPlainObject(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return false;
    }
    const prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
  }

  function isNonnegativeInteger(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }

  function safeString(value, fallback = "") {
    return typeof value === "string" ? value : fallback;
  }

  function safeCount(value) {
    return isNonnegativeInteger(value) ? String(value) : "不可用";
  }

  function setText(id, value) {
    const node = byId(id);
    if (node) {
      node.textContent = value;
    }
  }

  function userError(code, message) {
    const error = new Error(code);
    error.code = code;
    error.userMessage = message;
    return error;
  }

  function setStatus(text, kind) {
    const node = byId("run-status");
    node.textContent = text;
    node.dataset.state = kind;
  }

  function setModelStatus(text, kind) {
    const node = byId("model-name");
    node.textContent = text;
    node.dataset.state = kind;
  }

  function setControlsBusy() {
    const busy = state.running || state.resetting;
    byId("run-button").disabled = busy || !state.modelConfigured;
    byId("task-input").disabled = busy;
    byId("refresh-button").disabled = busy || state.treeLoading;
    byId("tree-load-more").disabled = busy || state.treeLoading;
    byId("reset-button").disabled = busy;
    byId("confirm-reset").disabled = busy;
  }

  function activatePane(name) {
    state.activePane = name;
    const mobile = mobileLayout.matches;
    document.querySelectorAll("[data-pane]").forEach((node) => {
      const active = node.dataset.pane === name;
      node.classList.toggle("active", active);
      node.hidden = mobile && !active;
      node.setAttribute("aria-hidden", String(mobile && !active));
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      const active = button.dataset.view === name;
      button.classList.toggle("active", active);
      if (active) {
        button.setAttribute("aria-current", "page");
      } else {
        button.removeAttribute("aria-current");
      }
    });
  }

  function resetUsage() {
    state.usage = {
      modelCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    setText("usage-calls", "0");
    setText("usage-prompt", "0");
    setText("usage-completion", "0");
    setText("usage-total", "0");
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
      return context === "file"
        ? "文件不存在或已被移动"
        : "请求的内容不存在";
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

  function responseTooLargeError() {
    return userError("RESPONSE_TOO_LARGE", "响应内容过大");
  }

  function invalidResponseError() {
    return userError(
      "INVALID_RESPONSE",
      "服务返回了无法识别的数据",
    );
  }

  function streamUnavailableError() {
    return userError("STREAM_UNAVAILABLE", "响应流不可用");
  }

  function trustedContentLength(response) {
    const raw = response.headers?.get?.("content-length");
    if (typeof raw !== "string" || !/^\d+$/.test(raw.trim())) {
      return null;
    }
    const parsed = Number(raw.trim());
    return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null;
  }

  function decodeUtf8(bytes) {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw invalidResponseError();
    }
  }

  async function cancelReader(reader) {
    try {
      await reader.cancel();
    } catch {
      // Preserve the stable size error even if stream cleanup fails.
    }
  }

  async function cancelResponseBody(response) {
    const body = response.body;
    if (!body || body.locked || typeof body.cancel !== "function") {
      return;
    }
    try {
      await body.cancel();
    } catch {
      // Preserve the stable size error even if response cleanup fails.
    }
  }

  async function readResponseTextLimited(
    response,
    maxBytes,
    abortController = null,
  ) {
    if (!isNonnegativeInteger(maxBytes)) {
      throw invalidResponseError();
    }

    const contentLength = trustedContentLength(response);
    if (contentLength !== null && contentLength > maxBytes) {
      const cancellation = cancelResponseBody(response);
      abortController?.abort();
      await cancellation;
      throw responseTooLargeError();
    }

    if (!response.body || typeof response.body.getReader !== "function") {
      throw streamUnavailableError();
    }

    const reader = response.body.getReader();
    const chunks = [];
    let byteLength = 0;
    while (true) {
      const result = await reader.read();
      if (
        result === null ||
        typeof result !== "object" ||
        Array.isArray(result) ||
        typeof result.done !== "boolean"
      ) {
        await cancelReader(reader);
        throw invalidResponseError();
      }
      if (result.done) {
        break;
      }
      if (!(result.value instanceof Uint8Array)) {
        await cancelReader(reader);
        throw invalidResponseError();
      }
      byteLength += result.value.byteLength;
      if (!Number.isSafeInteger(byteLength) || byteLength > maxBytes) {
        await cancelReader(reader);
        throw responseTooLargeError();
      }
      chunks.push(result.value);
    }

    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return { text: decodeUtf8(bytes), byteLength };
  }

  async function fetchJson(path, options = {}, context = "request") {
    const controller = new AbortController();
    const externalSignal = options.signal;
    let timedOut = false;
    const forwardAbort = () => controller.abort();
    if (externalSignal?.aborted) {
      controller.abort();
    } else {
      externalSignal?.addEventListener("abort", forwardAbort, { once: true });
    }
    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, REQUEST_TIMEOUT_MS);
    const requestOptions = { ...options };
    delete requestOptions.signal;

    try {
      const response = await fetch(path, {
        credentials: "same-origin",
        ...requestOptions,
        headers: {
          Accept: "application/json",
          ...(requestOptions.headers || {}),
        },
        signal: controller.signal,
      });
      const limited = await readResponseTextLimited(
        response,
        MAX_JSON_RESPONSE_BYTES,
        controller,
      );
      const raw = limited.text;

      let payload = null;
      try {
        payload = JSON.parse(raw);
      } catch {
        payload = null;
      }
      if (!response.ok) {
        const detail =
          isPlainObject(payload) && isPlainObject(payload.detail)
            ? payload.detail
            : {};
        const error = userError(
          "REQUEST_FAILED",
          apiMessage(
            response.status,
            safeString(detail.error_code),
            context,
          ),
        );
        error.status = response.status;
        throw error;
      }
      if (!isPlainObject(payload)) {
        throw invalidResponseError();
      }
      return { data: payload, byteLength: limited.byteLength };
    } catch (error) {
      if (error && error.name === "AbortError") {
        if (!timedOut && externalSignal?.aborted) {
          throw error;
        }
        throw userError("REQUEST_TIMEOUT", "请求超时，请稍后重试");
      }
      if (error && typeof error.userMessage === "string") {
        throw error;
      }
      throw userError(
        "NETWORK_FAILED",
        "无法连接服务，请检查网络后重试",
      );
    } finally {
      window.clearTimeout(timer);
      externalSignal?.removeEventListener?.("abort", forwardAbort);
    }
  }

  function validateTreePayload(payload) {
    if (
      !isPlainObject(payload) ||
      !Array.isArray(payload.entries) ||
      typeof payload.has_more !== "boolean" ||
      !(
        payload.next_cursor === null ||
        typeof payload.next_cursor === "string"
      )
    ) {
      return null;
    }
    const entries = [];
    for (const entry of payload.entries) {
      if (
        !isPlainObject(entry) ||
        typeof entry.path !== "string" ||
        !entry.path ||
        !["file", "directory"].includes(entry.type)
      ) {
        return null;
      }
      entries.push({
        path: entry.path,
        type: entry.type,
        size:
          isNonnegativeInteger(entry.size)
            ? entry.size
            : null,
      });
    }
    return {
      entries,
      hasMore: payload.has_more,
      nextCursor: payload.next_cursor,
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

  function invalidateFileRequest(options = {}) {
    state.fileRequestToken += 1;
    if (state.fileController) {
      state.fileController.abort();
      state.fileController = null;
    }
    if (options.clearSelection === true) {
      state.selectedPath = null;
    }
  }

  function resetTreePagination() {
    state.treeEntries = [];
    state.treeCursor = null;
    state.treeSeenCursors = new Set();
    state.treePages = 0;
    state.treeJsonBytes = 0;
    state.treeLimitReached = false;
  }

  function treePaginationError() {
    return userError("TREE_PAGINATION", "文件列表分页异常");
  }

  async function loadTree(options = {}) {
    const append = options.append === true;
    if (state.treeLoading || (append && !state.treeCursor)) {
      return;
    }
    if (!append && options.keepFileRequest !== true) {
      invalidateFileRequest();
      if (state.selectedPath) {
        setText("viewer-meta", "预览已停止");
      }
    }

    state.treeLoading = true;
    setControlsBusy();
    if (!append) {
      resetTreePagination();
      setText("tree-status", "正在读取文件");
      byId("tree-load-more").hidden = true;
    }

    const pageBudget = append
      ? TREE_MAX_PAGES - state.treePages
      : Math.min(TREE_INITIAL_PAGES, TREE_MAX_PAGES);
    let pagesLoaded = 0;
    let limitMessage = "";

    try {
      while (
        pagesLoaded < pageBudget &&
        state.treePages < TREE_MAX_PAGES &&
        state.treeEntries.length < TREE_MAX_ENTRIES
      ) {
        const currentCursor = state.treeCursor;
        const response = await fetchJson(
          treeUrl(currentCursor),
          {},
          "tree",
        );
        if (
          state.treeJsonBytes + response.byteLength >
          TREE_MAX_JSON_BYTES
        ) {
          state.treeLimitReached = true;
          state.treeCursor = null;
          limitMessage = "文件列表响应过大，仅显示已加载内容";
          break;
        }
        state.treeJsonBytes += response.byteLength;

        const page = validateTreePayload(response.data);
        if (!page) {
          throw userError(
            "INVALID_TREE",
            "文件列表格式无法识别",
          );
        }
        if (page.hasMore) {
          if (
            page.entries.length === 0 ||
            typeof page.nextCursor !== "string" ||
            !page.nextCursor ||
            page.nextCursor === currentCursor ||
            state.treeSeenCursors.has(page.nextCursor)
          ) {
            throw treePaginationError();
          }
        } else if (page.nextCursor !== null) {
          throw treePaginationError();
        }

        const paths = new Set(
          state.treeEntries.map((entry) => entry.path),
        );
        const freshEntries = page.entries.filter(
          (entry) => !paths.has(entry.path),
        );
        if (page.entries.length > 0 && freshEntries.length === 0) {
          throw treePaginationError();
        }

        const remaining = TREE_MAX_ENTRIES - state.treeEntries.length;
        state.treeEntries.push(...freshEntries.slice(0, remaining));
        state.treePages += 1;
        pagesLoaded += 1;

        if (freshEntries.length > remaining) {
          state.treeLimitReached = true;
          state.treeCursor = null;
          limitMessage = `文件列表过大，仅显示前 ${TREE_MAX_ENTRIES} 项`;
          break;
        }
        if (!page.hasMore) {
          state.treeCursor = null;
          break;
        }

        state.treeSeenCursors.add(page.nextCursor);
        state.treeCursor = page.nextCursor;
        if (
          state.treeEntries.length >= TREE_MAX_ENTRIES ||
          state.treePages >= TREE_MAX_PAGES
        ) {
          state.treeLimitReached = true;
          state.treeCursor = null;
          limitMessage =
            state.treeEntries.length >= TREE_MAX_ENTRIES
              ? `文件列表过大，仅显示前 ${TREE_MAX_ENTRIES} 项`
              : `文件列表页数过多，仅显示前 ${state.treeEntries.length} 项`;
          break;
        }
      }

      state.treeEntries.sort((left, right) =>
        left.path.localeCompare(right.path, "zh-CN"),
      );
      renderTree();
      setText("tree-summary", `${state.treeEntries.length} 项`);
      if (limitMessage) {
        setText("tree-status", limitMessage);
      } else if (state.treeEntries.length === 0) {
        setText("tree-status", "工作区为空");
      } else if (state.treeCursor) {
        setText("tree-status", "可继续加载更多文件");
      } else {
        setText("tree-status", "");
      }
    } catch (error) {
      if (error && error.name === "AbortError") {
        return;
      }
      state.treeCursor = null;
      state.treeLimitReached = true;
      setText(
        "tree-status",
        error.userMessage || "文件列表加载失败",
      );
      if (!append && state.treeEntries.length === 0) {
        byId("file-tree").replaceChildren();
        setText("tree-summary", "读取失败");
      }
    } finally {
      state.treeLoading = false;
      byId("tree-load-more").hidden =
        !state.treeCursor || state.treeLimitReached;
      setControlsBusy();
    }
  }

  function buildTree(entries) {
    const root = {
      name: "",
      path: "",
      type: "directory",
      children: new Map(),
    };
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

  function renderTreeGroup(node, focusTargets) {
    const group = document.createElement("ul");
    group.className = "tree-group";

    for (const child of sortedChildren(node)) {
      const item = document.createElement("li");
      item.className = "tree-item";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "file-row";
      button.title = child.path;
      button.dataset.treePath = child.path;
      button.addEventListener("focus", () => {
        state.lastTreeFocusPath = child.path;
      });
      focusTargets.set(child.path, button);

      const chevron = document.createElement("span");
      chevron.className = "tree-chevron";
      chevron.textContent = child.type === "directory" ? "▾" : "";
      chevron.setAttribute("aria-hidden", "true");
      button.append(chevron);
      button.append(
        createIcon(
          child.type === "directory"
            ? "folder"
            : fileIcon(child.path),
        ),
      );

      const label = document.createElement("span");
      label.className = "file-label";
      label.textContent = child.name;
      button.append(label);

      if (child.type === "directory") {
        const collapsed = state.collapsedPaths.has(child.path);
        button.classList.add("directory-button");
        button.setAttribute("aria-expanded", collapsed ? "false" : "true");
        chevron.textContent = collapsed ? "›" : "▾";
        button.setAttribute(
          "aria-label",
          `${collapsed ? "展开" : "折叠"}目录 ${child.name}`,
        );
        button.addEventListener("click", () => {
          if (state.collapsedPaths.has(child.path)) {
            state.collapsedPaths.delete(child.path);
          } else {
            state.collapsedPaths.add(child.path);
          }
          renderTree();
        });
      } else {
        button.setAttribute(
          "aria-label",
          `预览文件 ${child.name}`,
        );
        if (state.selectedPath === child.path) {
          button.classList.add("active");
        }
        button.addEventListener("click", () => openFile(child.path));
      }

      item.append(button);
      if (child.type === "directory") {
        const childGroup = renderTreeGroup(child, focusTargets);
        state.treeGroupSequence += 1;
        childGroup.id = `tree-group-${state.treeGroupSequence}`;
        button.setAttribute("aria-controls", childGroup.id);
        item.append(childGroup);
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
    if (
      lower.endsWith(".json") ||
      lower.endsWith(".js") ||
      lower.endsWith(".py")
    ) {
      return "file-code";
    }
    return "file";
  }

  function nearestFocusTarget(path, targets) {
    let candidate = path;
    while (candidate) {
      if (targets.has(candidate)) {
        return targets.get(candidate);
      }
      const separator = candidate.lastIndexOf("/");
      candidate = separator >= 0 ? candidate.slice(0, separator) : "";
    }
    return null;
  }

  function renderTree() {
    const container = byId("file-tree");
    const focusedPath =
      document.activeElement?.dataset?.treePath ||
      state.lastTreeFocusPath;
    container.replaceChildren();
    state.treeGroupSequence = 0;
    const focusTargets = new Map();
    const tree = buildTree(state.treeEntries);
    container.append(renderTreeGroup(tree, focusTargets));
    refreshIcons();

    if (focusedPath) {
      const target = nearestFocusTarget(focusedPath, focusTargets);
      (target || container).focus();
    }
  }

  function validateFilePayload(payload) {
    if (
      !isPlainObject(payload) ||
      typeof payload.path !== "string" ||
      typeof payload.content !== "string" ||
      !isNonnegativeInteger(payload.offset) ||
      !isNonnegativeInteger(payload.next_offset) ||
      typeof payload.has_more !== "boolean" ||
      !(
        payload.next_cursor === null ||
        typeof payload.next_cursor === "string"
      ) ||
      !(
        payload.bytes_read === undefined ||
        isNonnegativeInteger(payload.bytes_read)
      )
    ) {
      return null;
    }
    return {
      path: payload.path,
      content: payload.content,
      offset: payload.offset,
      nextOffset: payload.next_offset,
      hasMore: payload.has_more,
      nextCursor: payload.next_cursor,
      bytesRead:
        payload.bytes_read === undefined ? null : payload.bytes_read,
      encoding: safeString(payload.encoding, "文本"),
    };
  }

  function measureFilePageBytes(page) {
    const offsetBytes = page.nextOffset - page.offset;
    const utf8Bytes = UTF8_ENCODER.encode(page.content).byteLength;
    return {
      utf8Bytes,
      budgetBytes: Math.max(
        offsetBytes,
        page.bytesRead ?? 0,
        utf8Bytes,
      ),
    };
  }

  function truncateUtf8(text, maxBytes) {
    const bytes = UTF8_ENCODER.encode(text);
    if (bytes.byteLength <= maxBytes) {
      return { text, byteLength: bytes.byteLength };
    }
    let end = Math.min(maxBytes, bytes.byteLength);
    while (end > 0) {
      try {
        return {
          text: new TextDecoder("utf-8", { fatal: true }).decode(
            bytes.subarray(0, end),
          ),
          byteLength: end,
        };
      } catch {
        end -= 1;
      }
    }
    return { text: "", byteLength: 0 };
  }

  function formatBytes(bytes) {
    if (bytes >= 1024 * 1024) {
      const megabytes = bytes / (1024 * 1024);
      const digits = Number.isInteger(megabytes) ? 0 : 2;
      return `${megabytes.toFixed(digits)} MB`;
    }
    return `${bytes} 字节`;
  }

  function fileRequestIsCurrent(token, path, pagePath = path) {
    return (
      token === state.fileRequestToken &&
      state.selectedPath === path &&
      pagePath === path
    );
  }

  function filePaginationError() {
    return userError("FILE_PAGINATION", "文件分页无进展");
  }

  async function openFile(path) {
    invalidateFileRequest();
    const token = state.fileRequestToken;
    const controller = new AbortController();
    state.fileController = controller;
    state.selectedPath = path;
    renderTree();
    setText("viewer-path", path);
    setText("viewer-meta", "正在读取");
    setText("file-content", "");
    activatePane("task");

    let cursor = null;
    let expectedOffset = 0;
    let content = "";
    let contentBytes = 0;
    let encoding = "文本";
    let truncated = false;
    const seenCursors = new Set();
    const pageLimit = Math.min(FILE_PAGE_BYTES, state.maxReadBytes);
    const pageRequestCap = Math.min(
      FILE_MAX_PAGE_REQUESTS,
      Math.ceil(FILE_MAX_PREVIEW_BYTES / pageLimit),
    );

    try {
      for (let pageNumber = 0; pageNumber < pageRequestCap; pageNumber += 1) {
        const params = new URLSearchParams({
          path,
          offset: String(expectedOffset),
          limit: String(pageLimit),
        });
        if (cursor) {
          params.set("cursor", cursor);
        }
        const response = await fetchJson(
          `${API_FILE}?${params.toString()}`,
          { signal: controller.signal },
          "file",
        );
        if (!fileRequestIsCurrent(token, path)) {
          return;
        }
        const page = validateFilePayload(response.data);
        if (
          !page ||
          !fileRequestIsCurrent(token, path, page?.path) ||
          page.offset !== expectedOffset ||
          page.nextOffset < page.offset
        ) {
          throw filePaginationError();
        }
        if (page.hasMore) {
          if (
            page.content.length === 0 ||
            page.nextOffset <= page.offset ||
            typeof page.nextCursor !== "string" ||
            !page.nextCursor ||
            page.nextCursor === cursor ||
            seenCursors.has(page.nextCursor)
          ) {
            throw filePaginationError();
          }
        } else if (page.nextCursor !== null) {
          throw filePaginationError();
        }

        encoding = page.encoding;
        const measured = measureFilePageBytes(page);
        const remaining = FILE_MAX_PREVIEW_BYTES - contentBytes;
        if (measured.budgetBytes > remaining) {
          const utf8Limit = measured.budgetBytes === 0
            ? 0
            : Math.floor(
                (remaining * measured.utf8Bytes) /
                measured.budgetBytes,
              );
          const clipped = truncateUtf8(page.content, utf8Limit);
          content += clipped.text;
          contentBytes += measured.utf8Bytes === 0
            ? 0
            : Math.min(
                remaining,
                Math.ceil(
                  (clipped.byteLength * measured.budgetBytes) /
                  measured.utf8Bytes,
                ),
              );
          truncated = true;
          break;
        }
        content += page.content;
        contentBytes += measured.budgetBytes;
        expectedOffset = page.nextOffset;

        if (!page.hasMore) {
          break;
        }
        seenCursors.add(page.nextCursor);
        cursor = page.nextCursor;
        if (
          contentBytes >= FILE_MAX_PREVIEW_BYTES ||
          pageNumber + 1 >= pageRequestCap
        ) {
          truncated = true;
          break;
        }
      }

      if (!fileRequestIsCurrent(token, path)) {
        return;
      }
      if (!content) {
        setText("file-content", "空文件");
        setText("viewer-meta", encoding);
      } else {
        setText("file-content", content);
        setText(
          "viewer-meta",
          truncated
            ? `${encoding} · 内容过大，仅显示前 ${formatBytes(contentBytes)}`
            : `${encoding} · ${formatBytes(contentBytes)}`,
        );
      }
    } catch (error) {
      if (error && error.name === "AbortError") {
        return;
      }
      if (!fileRequestIsCurrent(token, path)) {
        return;
      }
      setText(
        "file-content",
        error.userMessage || "文件预览失败",
      );
      setText("viewer-meta", "读取失败");
    } finally {
      if (token === state.fileRequestToken) {
        state.fileController = null;
      }
    }
  }

  function resetProtocolState() {
    state.activeRunId = null;
    state.lastModelCall = -1;
    state.lastUsageModelCalls = -1;
    state.lastToolStep = -1;
    state.activeToolStep = null;
    state.activeToolName = null;
  }

  function resetTraceView() {
    state.steps = 0;
    state.terminalSeen = false;
    resetProtocolState();
    byId("trace-list").replaceChildren();
    setText("step-count", "0");
    byId("trace-empty").hidden = false;
  }

  function hasExactKeys(value, keys) {
    if (!isPlainObject(value)) {
      return false;
    }
    const actual = Object.keys(value).sort();
    const expected = [...keys].sort();
    return (
      actual.length === expected.length &&
      actual.every((key, index) => key === expected[index])
    );
  }

  function validRunId(value) {
    return (
      typeof value === "string" &&
      /^[A-Za-z0-9-]{1,128}$/.test(value)
    );
  }

  function validBoundedString(value, maximum = MAX_EVENT_STRING_CHARS) {
    return typeof value === "string" && value.length <= maximum;
  }

  function validUsage(value) {
    if (
      !hasExactKeys(value, [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
      ])
    ) {
      return false;
    }
    return [
      value.prompt_tokens,
      value.completion_tokens,
      value.total_tokens,
    ].every((token) => token === null || isNonnegativeInteger(token));
  }

  function usageIsMonotonic(next) {
    const pairs = [
      [state.usage.promptTokens, next.prompt_tokens],
      [state.usage.completionTokens, next.completion_tokens],
      [state.usage.totalTokens, next.total_tokens],
    ];
    return pairs.every(([previous, current]) => {
      if (previous === null) {
        return current === null;
      }
      if (current === null) {
        return true;
      }
      return isNonnegativeInteger(previous) && current >= previous;
    });
  }

  function validJsonValue(value, depth = 0, budget = { nodes: 0 }) {
    budget.nodes += 1;
    if (budget.nodes > 256 || depth > 5) {
      return false;
    }
    if (value === null || typeof value === "boolean") {
      return true;
    }
    if (typeof value === "string") {
      return value.length <= 8192;
    }
    if (typeof value === "number") {
      return Number.isFinite(value);
    }
    if (Array.isArray(value)) {
      return (
        value.length <= 64 &&
        value.every((item) => validJsonValue(item, depth + 1, budget))
      );
    }
    if (!isPlainObject(value) || Object.keys(value).length > 32) {
      return false;
    }
    return Object.values(value).every((item) =>
      validJsonValue(item, depth + 1, budget),
    );
  }

  function validToolArguments(value) {
    if (!isPlainObject(value) || !validJsonValue(value)) {
      return false;
    }
    try {
      return JSON.stringify(value).length <= MAX_TOOL_EVENT_CHARS;
    } catch {
      return false;
    }
  }

  function eventBelongsToRun(event) {
    return (
      state.activeRunId !== null &&
      event.run_id === state.activeRunId
    );
  }

  function validateRunEvent(event) {
    if (
      !isPlainObject(event) ||
      !EVENT_TYPES.has(event.type)
    ) {
      return false;
    }

    if (event.type === "run_started") {
      if (
        !hasExactKeys(event, ["type", "run_id"]) ||
        !validRunId(event.run_id) ||
        state.activeRunId !== null
      ) {
        return false;
      }
      state.activeRunId = event.run_id;
      return true;
    }

    if (event.type === "run_failed" && event.run_id === undefined) {
      return (
        hasExactKeys(event, ["type", "message"]) &&
        validBoundedString(event.message)
      );
    }

    if (!validRunId(event.run_id) || !eventBelongsToRun(event)) {
      return false;
    }

    if (event.type === "model_call_started") {
      if (
        !hasExactKeys(event, ["type", "run_id", "call"]) ||
        !isNonnegativeInteger(event.call) ||
        event.call <= state.lastModelCall
      ) {
        return false;
      }
      state.lastModelCall = event.call;
      return true;
    }

    if (event.type === "usage_updated") {
      if (
        !hasExactKeys(event, [
          "type",
          "run_id",
          "model_calls",
          "usage",
        ]) ||
        !isNonnegativeInteger(event.model_calls) ||
        event.model_calls < state.lastModelCall ||
        event.model_calls <= state.lastUsageModelCalls ||
        !validUsage(event.usage) ||
        !usageIsMonotonic(event.usage)
      ) {
        return false;
      }
      state.lastUsageModelCalls = event.model_calls;
      return true;
    }

    if (event.type === "tool_started") {
      if (
        !hasExactKeys(event, [
          "type",
          "run_id",
          "step",
          "tool",
          "args",
        ]) ||
        !isNonnegativeInteger(event.step) ||
        event.step <= state.lastToolStep ||
        state.activeToolStep !== null ||
        !validBoundedString(event.tool, 128) ||
        !event.tool ||
        !validToolArguments(event.args)
      ) {
        return false;
      }
      state.activeToolStep = event.step;
      state.activeToolName = event.tool;
      return true;
    }

    if (event.type === "tool_finished") {
      if (
        !hasExactKeys(event, [
          "type",
          "run_id",
          "step",
          "tool",
          "ok",
          "result_summary",
        ]) ||
        event.step !== state.activeToolStep ||
        event.tool !== state.activeToolName ||
        typeof event.ok !== "boolean" ||
        !validBoundedString(event.result_summary)
      ) {
        return false;
      }
      state.lastToolStep = event.step;
      state.activeToolStep = null;
      state.activeToolName = null;
      return true;
    }

    if (event.type === "assistant_message") {
      return (
        hasExactKeys(event, ["type", "run_id", "content"]) &&
        validBoundedString(event.content)
      );
    }

    if (event.type === "run_completed") {
      return (
        hasExactKeys(event, [
          "type",
          "run_id",
          "status",
          "message",
          "model_calls",
          "usage",
        ]) &&
        event.status === "completed" &&
        validBoundedString(event.message) &&
        isNonnegativeInteger(event.model_calls) &&
        event.model_calls >= state.lastModelCall &&
        event.model_calls >= state.lastUsageModelCalls &&
        validUsage(event.usage) &&
        usageIsMonotonic(event.usage) &&
        state.activeToolStep === null
      );
    }

    return (
      hasExactKeys(event, [
        "type",
        "run_id",
        "status",
        "message",
        "model_calls",
        "usage",
      ]) &&
      event.status === "failed" &&
      validBoundedString(event.message) &&
      isNonnegativeInteger(event.model_calls) &&
      event.model_calls >= state.lastModelCall &&
      event.model_calls >= state.lastUsageModelCalls &&
      validUsage(event.usage) &&
      usageIsMonotonic(event.usage)
    );
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
      return `${event.tool} ${JSON.stringify(event.args)}`;
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
    item.dataset.kind =
      event.type === "run_failed"
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

  function clearConnectionTimer() {
    if (state.connectionTimer !== null) {
      window.clearTimeout(state.connectionTimer);
      state.connectionTimer = null;
    }
  }

  function clearRunWatchdog() {
    if (state.runWatchdog !== null) {
      window.clearTimeout(state.runWatchdog);
      state.runWatchdog = null;
    }
  }

  function clearRunTimers() {
    clearConnectionTimer();
    clearRunWatchdog();
  }

  function finishRun(message, failed) {
    if (!state.running) {
      return;
    }
    clearRunTimers();
    state.running = false;
    setControlsBusy();
    setStatus(
      failed ? message : "完成",
      failed ? "failed" : "completed",
    );
    setText("reply-state", failed ? "未完成" : "已完成");
    void loadTree();
  }

  function failProtocol(socket) {
    if (!state.running || state.terminalSeen) {
      return;
    }
    state.terminalSeen = true;
    clearRunTimers();
    setText("assistant-output", "运行协议错误");
    appendStep({ type: "run_failed", message: "Run failed" });
    finishRun("运行协议错误", true);
    socket.close(1002, "protocol-error");
  }

  function handleEvent(event) {
    if (state.terminalSeen || !validateRunEvent(event)) {
      return false;
    }
    if (event.run_id) {
      enableTraceDownload(event.run_id);
    }

    if (event.type === "usage_updated") {
      state.usage = {
        modelCalls: event.model_calls,
        promptTokens: event.usage.prompt_tokens,
        completionTokens: event.usage.completion_tokens,
        totalTokens: event.usage.total_tokens,
      };
      setText("usage-calls", safeCount(state.usage.modelCalls));
      setText("usage-prompt", safeCount(state.usage.promptTokens));
      setText(
        "usage-completion",
        safeCount(state.usage.completionTokens),
      );
      setText("usage-total", safeCount(state.usage.totalTokens));
      return true;
    }
    if (event.type === "assistant_message") {
      setText(
        "assistant-output",
        event.content || "模型未返回文字内容",
      );
      return true;
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
    return true;
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

  function watchdogDelay() {
    const configured = Number(state.maxRunSeconds) * 1000;
    return Math.min(
      Math.max(
        configured + RUN_WATCHDOG_BUFFER_MS,
        RUN_WATCHDOG_MIN_MS,
      ),
      RUN_WATCHDOG_MAX_MS,
    );
  }

  function startRun(task) {
    const trimmed = task.trim();
    if (
      !trimmed ||
      state.running ||
      state.resetting ||
      !state.modelConfigured
    ) {
      return;
    }

    clearRunTimers();
    resetUsage();
    state.running = true;
    resetTraceView();
    disableTraceDownload();
    setControlsBusy();
    setText("assistant-output", "正在运行");
    setText("reply-state", "处理中");
    setStatus("正在连接", "running");

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(
      `${protocol}//${location.host}/ws/agent`,
    );
    state.socket = socket;
    state.connectionTimer = window.setTimeout(() => {
      if (
        socket === state.socket &&
        state.running &&
        socket.readyState === WebSocket.CONNECTING
      ) {
        state.terminalSeen = true;
        setText("assistant-output", "连接超时，请重试");
        finishRun("连接超时", true);
        socket.close(1000, "connect-timeout");
      }
    }, SOCKET_CONNECT_TIMEOUT_MS);

    socket.addEventListener("open", () => {
      if (socket !== state.socket) {
        socket.close();
        return;
      }
      clearConnectionTimer();
      if (!state.running) {
        socket.close();
        return;
      }
      state.runWatchdog = window.setTimeout(() => {
        if (
          socket === state.socket &&
          state.running &&
          !state.terminalSeen
        ) {
          state.terminalSeen = true;
          setText("assistant-output", "运行响应超时");
          finishRun("运行响应超时", true);
          socket.close(1000, "run-watchdog");
        }
      }, watchdogDelay());
      setStatus("运行中", "running");
      socket.send(JSON.stringify({ type: "run", task: trimmed }));
    });

    socket.addEventListener("message", (message) => {
      if (socket !== state.socket || state.terminalSeen) {
        return;
      }
      if (
        typeof message.data !== "string" ||
        message.data.length > MAX_SOCKET_EVENT_CHARS
      ) {
        failProtocol(socket);
        return;
      }
      let event;
      try {
        event = JSON.parse(message.data);
      } catch {
        failProtocol(socket);
        return;
      }
      if (!handleEvent(event)) {
        failProtocol(socket);
      }
    });

    socket.addEventListener("error", () => {
      if (socket !== state.socket) {
        return;
      }
      clearRunTimers();
      if (
        state.running &&
        !state.terminalSeen
      ) {
        state.terminalSeen = true;
        setText("assistant-output", "连接发生错误，请重试");
        finishRun("连接发生错误", true);
        socket.close(1000, "socket-error");
      }
    });

    socket.addEventListener("close", (event) => {
      if (socket !== state.socket) {
        return;
      }
      clearRunTimers();
      state.socket = null;
      if (state.running && !state.terminalSeen) {
        state.terminalSeen = true;
        const message = socketFailureMessage(event);
        setText("assistant-output", message);
        appendStep({ type: "run_failed", message: "Run failed" });
        finishRun(message, true);
      }
    });
  }

  async function loadMeta() {
    try {
      const response = await fetchJson("/api/meta");
      const meta = response.data;
      if (
        !hasExactKeys(meta, [
          "model",
          "configured",
          "max_run_seconds",
          "max_read_bytes",
        ]) ||
        typeof meta.model !== "string" ||
        typeof meta.configured !== "boolean" ||
        typeof meta.max_run_seconds !== "number" ||
        !Number.isFinite(meta.max_run_seconds) ||
        meta.max_run_seconds <= 0 ||
        meta.max_run_seconds > 3600 ||
        !Number.isInteger(meta.max_read_bytes) ||
        meta.max_read_bytes < 1024 ||
        meta.max_read_bytes > FILE_PAGE_BYTES
      ) {
        throw userError(
          "INVALID_META",
          "模型状态格式无法识别",
        );
      }
      state.modelConfigured = meta.configured;
      state.maxRunSeconds = meta.max_run_seconds;
      state.maxReadBytes = meta.max_read_bytes;
      setModelStatus(
        meta.configured
          ? `${meta.model} · 已连接`
          : `${meta.model} · 未配置`,
        meta.configured ? "ready" : "warning",
      );
      if (!meta.configured) {
        setStatus("模型未配置", "failed");
        setText("assistant-output", "请先在服务端配置模型");
      }
    } catch (error) {
      state.modelConfigured = false;
      state.maxRunSeconds = null;
      setModelStatus("模型状态不可用", "warning");
      setStatus(
        error.userMessage || "连接服务失败",
        "failed",
      );
    } finally {
      setControlsBusy();
    }
  }

  async function resetWorkspace() {
    if (state.running || state.resetting) {
      return;
    }
    invalidateFileRequest({ clearSelection: true });
    state.resetting = true;
    setControlsBusy();
    setStatus("正在重置工作区", "running");
    byId("reset-dialog").close();
    setText("viewer-path", "文件预览");
    setText("viewer-meta", "");
    setText("file-content", "从左侧选择文件");
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
      state.collapsedPaths.clear();
      setText("assistant-output", "工作区已重置");
      setStatus("工作区已重置", "completed");
      await loadTree({ keepFileRequest: true });
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
    byId("refresh-button").addEventListener("click", () =>
      loadTree(),
    );
    byId("tree-load-more").addEventListener("click", () =>
      loadTree({ append: true }),
    );
    byId("reset-button").addEventListener("click", () =>
      byId("reset-dialog").showModal(),
    );
    byId("cancel-reset").addEventListener("click", () =>
      byId("reset-dialog").close(),
    );
    byId("confirm-reset").addEventListener(
      "click",
      resetWorkspace,
    );
    byId("trace-download").addEventListener("click", (event) => {
      if (!state.runId) {
        event.preventDefault();
      }
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () =>
        activatePane(button.dataset.view),
      );
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    bindInterface();
    activatePane("task");
    mobileLayout.addEventListener("change", () =>
      activatePane(state.activePane),
    );
    refreshIcons();
    disableTraceDownload();
    setControlsBusy();
    await loadMeta();
    await loadTree();
  });
})();
