import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const APP_SOURCE = await readFile(
  new URL("../static/app.js", import.meta.url),
  "utf8",
);

const flush = async () => {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve();
  }
};

const UTF8_ENCODER = new TextEncoder();
const JSON_RESPONSE_BYTE_LIMIT = 2 * 1024 * 1024;
const FILE_PREVIEW_BYTE_LIMIT = 1024 * 1024;

class FakeClassList {
  constructor(owner) {
    this.owner = owner;
    this.values = new Set();
  }

  set(value) {
    this.values = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  add(...names) {
    names.forEach((name) => this.values.add(name));
  }

  remove(...names) {
    names.forEach((name) => this.values.delete(name));
  }

  contains(name) {
    return this.values.has(name);
  }

  toggle(name, force) {
    const enabled = force === undefined ? !this.contains(name) : Boolean(force);
    if (enabled) {
      this.add(name);
    } else {
      this.remove(name);
    }
    return enabled;
  }

  toString() {
    return [...this.values].join(" ");
  }
}

class FakeElement {
  constructor(document, tagName, id = "") {
    this.ownerDocument = document;
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.classList = new FakeClassList(this);
    this.style = {};
    this.disabled = false;
    this.hidden = false;
    this.value = "";
    this.href = "";
    this.title = "";
    this.type = "";
    this._textContent = "";
    this.open = false;
  }

  get className() {
    return this.classList.toString();
  }

  set className(value) {
    this.classList.set(value);
  }

  get textContent() {
    return (
      this._textContent +
      this.children.map((child) => child.textContent).join("")
    );
  }

  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
  }

  append(...children) {
    for (const child of children) {
      child.parentNode = this;
      this.children.push(child);
    }
  }

  replaceChildren(...children) {
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = [];
    this._textContent = "";
    this.append(...children);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  async dispatch(type, detail = {}) {
    const event = {
      type,
      target: this,
      currentTarget: this,
      defaultPrevented: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      ...detail,
    };
    const pending = (this.listeners.get(type) ?? []).map((listener) =>
      listener(event),
    );
    await Promise.all(pending);
    return event;
  }

  async click() {
    if (this.disabled) {
      return;
    }
    this.focus();
    return this.dispatch("click");
  }

  focus() {
    this.ownerDocument.activeElement = this;
    for (const listener of this.listeners.get("focus") ?? []) {
      listener({
        type: "focus",
        target: this,
        currentTarget: this,
      });
    }
  }

  showModal() {
    this.open = true;
  }

  close() {
    this.open = false;
  }
}

const walk = function* (node) {
  yield node;
  for (const child of node.children) {
    yield* walk(child);
  }
};

class FakeDocument {
  constructor() {
    this.elements = new Map();
    this.domListeners = new Map();
    this.documentElement = new FakeElement(this, "html", "document-root");
    this.body = new FakeElement(this, "body", "body");
    this.documentElement.append(this.body);
    this.activeElement = this.body;
    this.panes = [];
    this.views = [];
    this.createStaticElements();
  }

  createStaticElements() {
    const ids = [
      "model-name",
      "usage-calls",
      "usage-prompt",
      "usage-completion",
      "usage-total",
      "trace-download",
      "file-tree",
      "refresh-button",
      "reset-button",
      "task-form",
      "task-input",
      "run-button",
      "run-status",
      "reply-state",
      "assistant-output",
      "viewer-path",
      "viewer-meta",
      "file-content",
      "trace-list",
      "trace-empty",
      "step-count",
      "tree-summary",
      "tree-status",
      "tree-load-more",
      "reset-dialog",
      "cancel-reset",
      "confirm-reset",
    ];
    for (const id of ids) {
      const tag = id === "task-form"
        ? "form"
        : id === "task-input"
          ? "textarea"
          : id === "trace-list"
            ? "ol"
            : id === "file-tree"
              ? "nav"
              : id.includes("button") || id.includes("reset") || id === "tree-load-more"
                ? "button"
                : id === "reset-dialog"
                  ? "dialog"
                  : "div";
      const element = new FakeElement(this, tag, id);
      this.elements.set(id, element);
      this.body.append(element);
    }
    this.getElementById("trace-download").className =
      "icon-button is-disabled";
    this.getElementById("tree-load-more").hidden = true;

    for (const name of ["files", "task", "trace"]) {
      const pane = new FakeElement(this, "section", `${name}-pane`);
      pane.dataset.pane = name;
      pane.className = name === "task" ? "pane active" : "pane";
      this.panes.push(pane);
      this.body.append(pane);

      const view = new FakeElement(this, "button", `${name}-view`);
      view.dataset.view = name;
      view.className = name === "task" ? "active" : "";
      this.views.push(view);
      this.body.append(view);
    }
  }

  getElementById(id) {
    return this.elements.get(id) ?? null;
  }

  createElement(tagName) {
    return new FakeElement(this, tagName);
  }

  addEventListener(type, listener) {
    const listeners = this.domListeners.get(type) ?? [];
    listeners.push(listener);
    this.domListeners.set(type, listeners);
  }

  async fireDOMContentLoaded() {
    const pending = (this.domListeners.get("DOMContentLoaded") ?? []).map(
      (listener) => listener({ type: "DOMContentLoaded" }),
    );
    await Promise.all(pending);
  }

  querySelectorAll(selector) {
    if (selector === "[data-pane]") {
      return this.panes;
    }
    if (selector === "[data-view]") {
      return this.views;
    }
    if (selector === ".file-row.active") {
      return [...walk(this.getElementById("file-tree"))].filter(
        (node) =>
          node.classList.contains("file-row") &&
          node.classList.contains("active"),
      );
    }
    return [];
  }

  findInTree(predicate) {
    return [...walk(this.getElementById("file-tree"))].find(predicate);
  }
}

class FakeClock {
  constructor() {
    this.now = 0;
    this.nextId = 1;
    this.tasks = new Map();
  }

  setTimeout(callback, delay) {
    const id = this.nextId;
    this.nextId += 1;
    this.tasks.set(id, {
      at: this.now + Number(delay),
      callback,
    });
    return id;
  }

  clearTimeout(id) {
    this.tasks.delete(id);
  }

  async tick(milliseconds) {
    const target = this.now + milliseconds;
    while (true) {
      const next = [...this.tasks.entries()]
        .filter(([, task]) => task.at <= target)
        .sort((left, right) => left[1].at - right[1].at)[0];
      if (!next) {
        break;
      }
      const [id, task] = next;
      this.tasks.delete(id);
      this.now = task.at;
      task.callback();
      await flush();
    }
    this.now = target;
    await flush();
  }
}

const responseBytes = (value) =>
  value instanceof Uint8Array ? value : UTF8_ENCODER.encode(String(value));

const byteResponse = (raw, options = {}) => {
  const status = options.status ?? 200;
  const bytes = responseBytes(raw);
  const chunks = (options.chunks ?? [bytes]).map(responseBytes);
  const stats = options.stats ?? {};
  stats.reads ??= 0;
  stats.cancels ??= 0;
  stats.textCalls ??= 0;
  stats.arrayBufferCalls ??= 0;
  stats.getReaderCalls ??= 0;
  let chunkIndex = 0;
  let cancelled = false;
  const contentLength = options.contentLength === undefined
    ? String(bytes.byteLength)
    : options.contentLength;

  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    headers: {
      get(name) {
        return String(name).toLowerCase() === "content-length"
          ? contentLength
          : null;
      },
    },
    body: options.body === false
      ? null
      : {
          getReader() {
            stats.getReaderCalls += 1;
            return {
              async read() {
                stats.reads += 1;
                if (cancelled || chunkIndex >= chunks.length) {
                  return { done: true, value: undefined };
                }
                const value = chunks[chunkIndex];
                chunkIndex += 1;
                return { done: false, value };
              },
              async cancel() {
                stats.cancels += 1;
                cancelled = true;
              },
            };
          },
        },
    async json() {
      return JSON.parse(raw);
    },
    async text() {
      stats.textCalls += 1;
      return raw;
    },
    async arrayBuffer() {
      stats.arrayBufferCalls += 1;
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      );
    },
  };
};

const jsonResponse = (value, options = {}) =>
  byteResponse(options.raw ?? JSON.stringify(value), options);

const abortError = () => {
  const error = new Error("aborted");
  error.name = "AbortError";
  return error;
};

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
};

const waitForDeferred = (pending, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const abort = () => reject(abortError());
    signal?.addEventListener("abort", abort, { once: true });
    pending.promise.then(resolve, reject);
  });

class FakeSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url, sockets) {
    this.url = url;
    this.readyState = FakeSocket.CONNECTING;
    this.listeners = new Map();
    this.sent = [];
    this.closeCalls = [];
    sockets.push(this);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  async emit(type, event = {}) {
    await Promise.all(
      (this.listeners.get(type) ?? []).map((listener) => listener(event)),
    );
  }

  async open() {
    this.readyState = FakeSocket.OPEN;
    await this.emit("open");
  }

  async message(value) {
    await this.emit("message", {
      data: typeof value === "string" ? value : JSON.stringify(value),
    });
  }

  close(code = 1000, reason = "") {
    if (this.readyState === FakeSocket.CLOSED) {
      return;
    }
    this.closeCalls.push({ code, reason });
    this.readyState = FakeSocket.CLOSED;
    void this.emit("close", { code, reason });
  }

  send(value) {
    this.sent.push(value);
  }
}

const defaultTree = {
  entries: [],
  warnings: [],
  has_more: false,
  next_cursor: null,
};

async function createHarness(fetchRoute, options = {}) {
  const document = new FakeDocument();
  const clock = new FakeClock();
  const sockets = [];
  const fetchCalls = [];
  const route = fetchRoute ?? (async (url) => {
    if (url === "/api/meta") {
      return jsonResponse({
        model: "test-model",
        configured: true,
        max_run_seconds: 1,
      });
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse(defaultTree);
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const fetch = async (url, request = {}) => {
    fetchCalls.push({ url: String(url), request });
    if (request.signal?.aborted) {
      throw abortError();
    }
    return route(String(url), request);
  };
  const window = {
    document,
    lucide: null,
    setTimeout: clock.setTimeout.bind(clock),
    clearTimeout: clock.clearTimeout.bind(clock),
    matchMedia() {
      return {
        matches: options.mobile === true,
        addEventListener() {},
      };
    },
  };
  window.window = window;
  const SocketConstructor = class extends FakeSocket {
    constructor(url) {
      super(url, sockets);
    }
  };
  Object.assign(SocketConstructor, {
    CONNECTING: FakeSocket.CONNECTING,
    OPEN: FakeSocket.OPEN,
    CLOSING: FakeSocket.CLOSING,
    CLOSED: FakeSocket.CLOSED,
  });
  const context = vm.createContext({
    AbortController,
    ArrayBuffer,
    TextDecoder,
    TextEncoder,
    Uint8Array,
    URLSearchParams,
    WebSocket: SocketConstructor,
    console,
    document,
    fetch,
    location: {
      protocol: "http:",
      host: "testserver",
    },
    window,
  });

  vm.runInContext(APP_SOURCE, context, { filename: "static/app.js" });
  await document.fireDOMContentLoaded();
  await flush();
  return {
    clock,
    document,
    fetchCalls,
    sockets,
  };
}

const fileButton = (harness, name) =>
  harness.document.findInTree(
    (node) => node.getAttribute("aria-label") === `预览文件 ${name}`,
  );

const directoryButton = (harness, name) =>
  harness.document.findInTree(
    (node) =>
      node.getAttribute("aria-label")?.endsWith(`目录 ${name}`) === true,
  );

const configuredMeta = (seconds = 1) =>
  jsonResponse({
    model: "test-model",
    configured: true,
    max_run_seconds: seconds,
  });

test("a slow file response cannot replace a newer file preview", async () => {
  const slowA = deferred();
  const route = async (url, request) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [
          { path: "A.md", type: "file", size: 1 },
          { path: "B.md", type: "file", size: 1 },
        ],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("path=A.md")) {
      return waitForDeferred(slowA, request.signal);
    }
    if (url.includes("path=B.md")) {
      return jsonResponse({
        path: "B.md",
        content: "B content",
        offset: 0,
        next_offset: 9,
        has_more: false,
        encoding: "utf-8",
        next_cursor: null,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  const first = fileButton(harness, "A.md").click();
  await flush();
  await fileButton(harness, "B.md").click();
  slowA.resolve(jsonResponse({
    path: "A.md",
    content: "A stale content",
    offset: 0,
    next_offset: 15,
    has_more: false,
    encoding: "utf-8",
    next_cursor: null,
  }));
  await first;
  await flush();

  assert.equal(
    harness.document.getElementById("file-content").textContent,
    "B content",
  );
  assert.equal(
    harness.document.getElementById("viewer-path").textContent,
    "B.md",
  );
});

test("reset invalidates an in-flight file preview", async () => {
  const slow = deferred();
  let treeLoads = 0;
  const route = async (url, request) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      treeLoads += 1;
      return jsonResponse({
        entries: treeLoads === 1
          ? [{ path: "A.md", type: "file", size: 1 }]
          : [{ path: "seed.md", type: "file", size: 4 }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      return waitForDeferred(slow, request.signal);
    }
    if (url === "/api/reset") {
      return jsonResponse({ status: "reset" });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  const preview = fileButton(harness, "A.md").click();
  await flush();
  await harness.document.getElementById("confirm-reset").click();
  slow.resolve(jsonResponse({
    path: "A.md",
    content: "stale after reset",
    offset: 0,
    next_offset: 17,
    has_more: false,
    encoding: "utf-8",
    next_cursor: null,
  }));
  await preview;
  await flush();

  assert.notEqual(
    harness.document.getElementById("file-content").textContent,
    "stale after reset",
  );
  assert.equal(
    harness.document.getElementById("viewer-path").textContent,
    "文件预览",
  );
});

test("refresh invalidates an in-flight file preview", async () => {
  const slow = deferred();
  let treeLoads = 0;
  const route = async (url, request) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      treeLoads += 1;
      return jsonResponse({
        entries: treeLoads === 1
          ? [{ path: "A.md", type: "file", size: 1 }]
          : [{ path: "B.md", type: "file", size: 1 }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      return waitForDeferred(slow, request.signal);
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  const preview = fileButton(harness, "A.md").click();
  await flush();
  await harness.document.getElementById("refresh-button").click();
  slow.resolve(jsonResponse({
    path: "A.md",
    content: "stale after refresh",
    offset: 0,
    next_offset: 19,
    has_more: false,
    encoding: "utf-8",
    next_cursor: null,
  }));
  await preview;
  await flush();

  assert.notEqual(
    harness.document.getElementById("file-content").textContent,
    "stale after refresh",
  );
  assert.equal(
    harness.document.getElementById("viewer-meta").textContent,
    "预览已停止",
  );
});

test("tree pagination rejects a repeated cursor", async () => {
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [{ path: "same.md", type: "file", size: 1 }],
        has_more: true,
        next_cursor: "same-cursor",
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);

  assert.match(
    harness.document.getElementById("tree-status").textContent,
    /分页异常/,
  );
  assert.equal(
    harness.document.getElementById("tree-load-more").hidden,
    true,
  );
});

test("file pagination rejects empty pages without progress", async () => {
  let filePage = 0;
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [{ path: "empty.md", type: "file", size: 2 }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      filePage += 1;
      return jsonResponse({
        path: "empty.md",
        content: "",
        offset: 0,
        next_offset: 0,
        has_more: filePage === 1,
        encoding: "utf-8",
        next_cursor: filePage === 1 ? "next" : null,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  await fileButton(harness, "empty.md").click();

  assert.match(
    harness.document.getElementById("file-content").textContent,
    /分页无进展/,
  );
});

test("file pagination rejects a cursor that does not advance", async () => {
  let page = 0;
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [{ path: "loop.md", type: "file", size: 2 }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      page += 1;
      return jsonResponse({
        path: "loop.md",
        content: "x",
        offset: page - 1,
        next_offset: page,
        has_more: true,
        encoding: "utf-8",
        next_cursor: "same",
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  await fileButton(harness, "loop.md").click();

  assert.match(
    harness.document.getElementById("file-content").textContent,
    /分页无进展/,
  );
});

test("file continuation sends the cursor's byte offset", async () => {
  const fileUrls = [];
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [{ path: "utf8.md", type: "file", size: 4 }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      fileUrls.push(url);
      if (fileUrls.length === 1) {
        return jsonResponse({
          path: "utf8.md",
          content: "你",
          offset: 0,
          next_offset: 3,
          has_more: true,
          encoding: "utf-8",
          next_cursor: "page-2",
        });
      }
      return jsonResponse({
        path: "utf8.md",
        content: "x",
        offset: 3,
        next_offset: 4,
        has_more: false,
        encoding: "utf-8",
        next_cursor: null,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);

  await fileButton(harness, "utf8.md").click();

  assert.equal(fileUrls.length, 2);
  const continuation = new URL(fileUrls[1], "http://testserver");
  assert.equal(continuation.searchParams.get("cursor"), "page-2");
  assert.equal(continuation.searchParams.get("offset"), "3");
});

test("stream reader accepts an exact ASCII byte boundary", async () => {
  const prefix =
    '{"entries":[],"has_more":false,"next_cursor":null,"pad":"';
  const suffix = '"}';
  const fixedBytes = UTF8_ENCODER.encode(prefix + suffix).byteLength;
  const raw = prefix + "x".repeat(JSON_RESPONSE_BYTE_LIMIT - fixedBytes) + suffix;
  const bytes = UTF8_ENCODER.encode(raw);
  const stats = {};
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return byteResponse(raw, {
        chunks: [bytes.subarray(0, 700_000), bytes.subarray(700_000)],
        stats,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.equal(bytes.byteLength, JSON_RESPONSE_BYTE_LIMIT);
  assert.equal(
    harness.document.getElementById("tree-status").textContent,
    "工作区为空",
  );
  assert.equal(stats.getReaderCalls, 1);
  assert.equal(stats.textCalls, 0);
  assert.equal(stats.cancels, 0);
});

test("stream reader decodes Chinese and emoji split across chunks", async () => {
  const raw = JSON.stringify({
    entries: [],
    has_more: false,
    next_cursor: null,
    note: "中文🙂",
  });
  const emojiIndex = raw.indexOf("🙂");
  const emojiByteStart = UTF8_ENCODER.encode(raw.slice(0, emojiIndex)).byteLength;
  const bytes = UTF8_ENCODER.encode(raw);
  const stats = {};
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return byteResponse(raw, {
        chunks: [
          bytes.subarray(0, emojiByteStart + 1),
          bytes.subarray(emojiByteStart + 1, emojiByteStart + 3),
          bytes.subarray(emojiByteStart + 3),
        ],
        stats,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.equal(
    harness.document.getElementById("tree-status").textContent,
    "工作区为空",
  );
  assert.equal(stats.getReaderCalls, 1);
  assert.equal(stats.textCalls, 0);
  assert.equal(stats.cancels, 0);
});

test("single oversized stream chunk is cancelled without UI injection", async () => {
  const marker = "UNTRUSTED_UI_MARKER";
  const raw = "x".repeat(JSON_RESPONSE_BYTE_LIMIT + 1) + marker;
  const stats = {};
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return byteResponse(raw, {
        contentLength: null,
        chunks: [UTF8_ENCODER.encode(raw)],
        stats,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.match(
    harness.document.getElementById("tree-status").textContent,
    /响应内容过大/,
  );
  assert.equal(stats.reads, 1);
  assert.equal(stats.cancels, 1);
  assert.equal(stats.textCalls, 0);
  assert.equal(harness.document.body.textContent.includes(marker), false);
});

test("multi-chunk overflow cancels before reading later chunks", async () => {
  const marker = "LATE_UNTRUSTED_MARKER";
  const first = UTF8_ENCODER.encode("x".repeat(JSON_RESPONSE_BYTE_LIMIT));
  const second = UTF8_ENCODER.encode("y");
  const third = UTF8_ENCODER.encode(marker);
  const stats = {};
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return byteResponse(
        "x".repeat(JSON_RESPONSE_BYTE_LIMIT) + "y" + marker,
        {
          contentLength: null,
          chunks: [first, second, third],
          stats,
        },
      );
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.match(
    harness.document.getElementById("tree-status").textContent,
    /响应内容过大/,
  );
  assert.equal(stats.reads, 2);
  assert.equal(stats.cancels, 1);
  assert.equal(stats.textCalls, 0);
  assert.equal(harness.document.body.textContent.includes(marker), false);
});

test("oversized Content-Length is rejected before body reading", async () => {
  const raw = JSON.stringify(defaultTree);
  const stats = {};
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return byteResponse(raw, {
        contentLength: String(JSON_RESPONSE_BYTE_LIMIT + 1),
        stats,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.match(
    harness.document.getElementById("tree-status").textContent,
    /响应内容过大/,
  );
  assert.equal(stats.getReaderCalls, 0);
  assert.equal(stats.reads, 0);
  assert.equal(stats.textCalls, 0);
  assert.equal(stats.arrayBufferCalls, 0);
});

test("response without a body uses bounded arrayBuffer fallback", async () => {
  const stats = {};
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse(defaultTree, { body: false, stats });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.equal(
    harness.document.getElementById("tree-status").textContent,
    "工作区为空",
  );
  assert.equal(stats.arrayBufferCalls, 1);
  assert.equal(stats.textCalls, 0);
});

test("tree aggregate budget counts UTF-8 wire bytes", async () => {
  let page = 0;
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      page += 1;
      return jsonResponse({
        entries: [{ path: `page-${page}.md`, type: "file", size: 1 }],
        has_more: page === 1,
        next_cursor: page === 1 ? "page-2" : null,
        pad: "中".repeat(400_000),
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });

  assert.equal(
    harness.document.getElementById("tree-summary").textContent,
    "1 项",
  );
  assert.match(
    harness.document.getElementById("tree-status").textContent,
    /响应过大.*仅显示已加载内容/,
  );
});

test("tree loading stops at 5000 entries and reports the cap", async () => {
  let page = 0;
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      const current = page;
      page += 1;
      return jsonResponse({
        entries: Array.from({ length: 200 }, (_, index) => ({
          path: `file-${String(current * 200 + index).padStart(5, "0")}.md`,
          type: "file",
          size: 1,
        })),
        has_more: true,
        next_cursor: `cursor-${page}`,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  await harness.document.getElementById("tree-load-more").click();

  assert.equal(
    harness.document.getElementById("tree-summary").textContent,
    "5000 项",
  );
  assert.match(
    harness.document.getElementById("tree-status").textContent,
    /仅显示前 5000 项/,
  );
  assert.equal(
    harness.document.getElementById("tree-load-more").hidden,
    true,
  );
});

test("repeated load-more clicks share one in-flight request", async () => {
  const pending = deferred();
  let page = 0;
  let appendedRequests = 0;
  const route = async (url, request) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      if (page < 20) {
        page += 1;
        return jsonResponse({
          entries: [{ path: `file-${page}.md`, type: "file", size: 1 }],
          has_more: true,
          next_cursor: `cursor-${page}`,
        });
      }
      appendedRequests += 1;
      return waitForDeferred(pending, request.signal);
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  const button = harness.document.getElementById("tree-load-more");
  const first = button.click();
  await flush();
  const second = button.click();
  await flush();

  assert.equal(appendedRequests, 1);
  pending.resolve(jsonResponse({
    entries: [{ path: "last.md", type: "file", size: 1 }],
    has_more: false,
    next_cursor: null,
  }));
  await Promise.all([first, second]);
});

test("file preview truncates Chinese content by UTF-8 bytes", async () => {
  const oversized = "中".repeat(400_000);
  const oversizedBytes = UTF8_ENCODER.encode(oversized).byteLength;
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [{ path: "large.md", type: "file", size: oversizedBytes }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      return jsonResponse({
        path: "large.md",
        content: oversized,
        offset: 0,
        next_offset: oversizedBytes,
        has_more: false,
        encoding: "utf-8",
        next_cursor: null,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);
  await fileButton(harness, "large.md").click();

  const visible = harness.document.getElementById("file-content").textContent;
  assert.ok(visible.length < oversized.length);
  assert.ok(UTF8_ENCODER.encode(visible).byteLength <= FILE_PREVIEW_BYTE_LIMIT);
  assert.match(
    harness.document.getElementById("viewer-meta").textContent,
    /仅显示前.*(字节|MB)/,
  );
  assert.doesNotMatch(
    harness.document.getElementById("viewer-meta").textContent,
    /字符/,
  );
});

test("one MiB of Chinese characters is rejected as a three MiB response", async () => {
  const content = "中".repeat(1024 * 1024);
  const stats = {};
  const route = async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse({
        entries: [{ path: "three-mib.md", type: "file", size: 3 * 1024 * 1024 }],
        has_more: false,
        next_cursor: null,
      });
    }
    if (url.includes("/api/file")) {
      const raw = JSON.stringify({
        path: "three-mib.md",
        content,
        offset: 0,
        next_offset: 3 * 1024 * 1024,
        has_more: false,
        encoding: "utf-8",
        next_cursor: null,
      });
      return byteResponse(raw, {
        contentLength: null,
        chunks: [UTF8_ENCODER.encode(raw)],
        stats,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  const harness = await createHarness(route);

  await fileButton(harness, "three-mib.md").click();

  const displayed = harness.document.getElementById("file-content").textContent;
  assert.ok(displayed.length < 100);
  assert.equal(displayed, "响应内容过大");
  assert.equal(stats.reads, 1);
  assert.equal(stats.cancels, 1);
  assert.equal(stats.textCalls, 0);
});

async function runningHarness(metaSeconds = 1) {
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta(metaSeconds);
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse(defaultTree);
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const input = harness.document.getElementById("task-input");
  input.value = "检查工作区";
  await harness.document.getElementById("task-form").dispatch("submit");
  const socket = harness.sockets[0];
  await socket.open();
  return { ...harness, socket };
}

test("malformed and cross-run events fail closed", async (context) => {
  const cases = [
    {
      name: "unknown event",
      events: [{ type: "unexpected" }],
    },
    {
      name: "run_started without id",
      events: [{ type: "run_started" }],
    },
    {
      name: "extra field",
      events: [{ type: "run_started", run_id: "run-1", extra: true }],
    },
    {
      name: "negative model call",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "model_call_started",
          run_id: "run-1",
          call: -1,
        },
      ],
    },
    {
      name: "tool args array",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "tool_started",
          run_id: "run-1",
          step: 1,
          tool: "read_file",
          args: [],
        },
      ],
    },
    {
      name: "cross run",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "model_call_started",
          run_id: "run-2",
          call: 1,
        },
      ],
    },
    {
      name: "decreasing usage",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "usage_updated",
          run_id: "run-1",
          model_calls: 2,
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6,
          },
        },
        {
          type: "usage_updated",
          run_id: "run-1",
          model_calls: 1,
          usage: {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
          },
        },
      ],
    },
    {
      name: "decreasing terminal model calls",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "usage_updated",
          run_id: "run-1",
          model_calls: 2,
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6,
          },
        },
        {
          type: "run_completed",
          run_id: "run-1",
          status: "completed",
          message: "done",
          model_calls: 1,
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6,
          },
        },
      ],
    },
    {
      name: "negative token",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "usage_updated",
          run_id: "run-1",
          model_calls: 1,
          usage: {
            prompt_tokens: -1,
            completion_tokens: 0,
            total_tokens: 0,
          },
        },
      ],
    },
    {
      name: "unknown tokens cannot become known again",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "usage_updated",
          run_id: "run-1",
          model_calls: 1,
          usage: {
            prompt_tokens: null,
            completion_tokens: null,
            total_tokens: null,
          },
        },
        {
          type: "usage_updated",
          run_id: "run-1",
          model_calls: 2,
          usage: {
            prompt_tokens: 2,
            completion_tokens: 1,
            total_tokens: 3,
          },
        },
      ],
    },
    {
      name: "tool_finished without a matching start",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "tool_finished",
          run_id: "run-1",
          step: 1,
          tool: "read_file",
          ok: true,
          result_summary: "done",
        },
      ],
    },
    {
      name: "assistant content type",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "assistant_message",
          run_id: "run-1",
          content: 42,
        },
      ],
    },
    {
      name: "completed status",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "run_completed",
          run_id: "run-1",
          status: "failed",
          message: "done",
          model_calls: 0,
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
        },
      ],
    },
    {
      name: "failed terminal extra field",
      events: [
        { type: "run_started", run_id: "run-1" },
        {
          type: "run_failed",
          run_id: "run-1",
          status: "failed",
          message: "failed",
          model_calls: 0,
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
          extra: true,
        },
      ],
    },
  ];

  for (const entry of cases) {
    await context.test(entry.name, async () => {
      const harness = await runningHarness();
      for (const event of entry.events) {
        await harness.socket.message(event);
      }
      assert.equal(
        harness.document.getElementById("assistant-output").textContent,
        "运行协议错误",
      );
      assert.equal(
        harness.document.getElementById("run-button").disabled,
        false,
      );
      assert.ok(harness.socket.closeCalls.length >= 1);
    });
  }
});

test("an early run_failed without run_id remains valid", async () => {
  const harness = await runningHarness();
  await harness.socket.message({
    type: "run_failed",
    message: "Server is busy",
  });

  assert.equal(
    harness.document.getElementById("assistant-output").textContent,
    "服务繁忙，请稍后重试",
  );
  assert.equal(
    harness.document.getElementById("run-button").disabled,
    false,
  );
});

test("a complete valid event sequence reaches one stable terminal", async () => {
  const harness = await runningHarness();
  const usage = {
    prompt_tokens: 3,
    completion_tokens: 2,
    total_tokens: 5,
  };
  const events = [
    { type: "run_started", run_id: "run-1" },
    {
      type: "model_call_started",
      run_id: "run-1",
      call: 1,
    },
    {
      type: "usage_updated",
      run_id: "run-1",
      model_calls: 1,
      usage,
    },
    {
      type: "tool_started",
      run_id: "run-1",
      step: 1,
      tool: "read_file",
      args: { path: "a.md" },
    },
    {
      type: "tool_finished",
      run_id: "run-1",
      step: 1,
      tool: "read_file",
      ok: true,
      result_summary: "read",
    },
    {
      type: "assistant_message",
      run_id: "run-1",
      content: "完成内容",
    },
    {
      type: "run_completed",
      run_id: "run-1",
      status: "completed",
      message: "完成内容",
      model_calls: 1,
      usage,
    },
  ];
  for (const event of events) {
    await harness.socket.message(event);
  }
  await harness.clock.tick(10000);

  assert.equal(
    harness.document.getElementById("assistant-output").textContent,
    "完成内容",
  );
  assert.equal(
    harness.document.getElementById("run-button").disabled,
    false,
  );
  assert.equal(
    harness.document.getElementById("run-status").textContent,
    "完成",
  );
});

test("server run deadline starts after socket open and restores controls", async () => {
  const harness = await runningHarness(1);

  await harness.clock.tick(5999);
  assert.equal(
    harness.document.getElementById("run-button").disabled,
    true,
  );
  await harness.clock.tick(1);

  assert.equal(
    harness.document.getElementById("assistant-output").textContent,
    "运行响应超时",
  );
  assert.equal(
    harness.document.getElementById("run-button").disabled,
    false,
  );
  assert.ok(harness.socket.closeCalls.length >= 1);
});

test("a stale socket close cannot clear the next run watchdog", async () => {
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta(1);
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse(defaultTree);
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const form = harness.document.getElementById("task-form");
  const input = harness.document.getElementById("task-input");
  input.value = "first";
  await form.dispatch("submit");
  const firstSocket = harness.sockets[0];
  firstSocket.close = function close(code = 1000, reason = "") {
    this.closeCalls.push({ code, reason });
    this.readyState = FakeSocket.CLOSING;
  };
  await firstSocket.open();
  await firstSocket.message({ type: "run_started", run_id: "run-1" });
  await firstSocket.message({
    type: "run_completed",
    run_id: "run-1",
    status: "completed",
    message: "done",
    model_calls: 0,
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  });

  input.value = "second";
  await form.dispatch("submit");
  const secondSocket = harness.sockets[1];
  await secondSocket.open();
  await firstSocket.emit("close", { code: 1000, reason: "completed" });
  await harness.clock.tick(6000);

  assert.equal(
    harness.document.getElementById("assistant-output").textContent,
    "运行响应超时",
  );
  assert.equal(
    harness.document.getElementById("run-button").disabled,
    false,
  );
});

test("a stale socket open cannot clear the next connection timeout", async () => {
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta(1);
    }
    if (url.startsWith("/api/tree")) {
      return jsonResponse(defaultTree);
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const form = harness.document.getElementById("task-form");
  const input = harness.document.getElementById("task-input");
  input.value = "first";
  await form.dispatch("submit");
  const firstSocket = harness.sockets[0];
  firstSocket.close = function close(code = 1000, reason = "") {
    this.closeCalls.push({ code, reason });
    this.readyState = FakeSocket.CLOSING;
  };
  await firstSocket.message({
    type: "run_failed",
    message: "Server is busy",
  });

  input.value = "second";
  await form.dispatch("submit");
  await firstSocket.open();
  await harness.clock.tick(10000);

  assert.equal(
    harness.document.getElementById("assistant-output").textContent,
    "连接超时，请重试",
  );
  assert.equal(
    harness.document.getElementById("run-button").disabled,
    false,
  );
});

test("refresh restores focus to the nearest surviving directory", async () => {
  let treeLoad = 0;
  const harness = await createHarness(async (url) => {
    if (url === "/api/meta") {
      return configuredMeta();
    }
    if (url.startsWith("/api/tree")) {
      treeLoad += 1;
      return jsonResponse({
        entries: treeLoad === 1
          ? [
              { path: "docs", type: "directory", size: 0 },
              { path: "docs/readme.md", type: "file", size: 1 },
            ]
          : [{ path: "docs", type: "directory", size: 0 }],
        has_more: false,
        next_cursor: null,
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  const removed = fileButton(harness, "readme.md");
  removed.focus();
  await harness.document.getElementById("refresh-button").click();

  assert.equal(
    harness.document.activeElement?.getAttribute("aria-label"),
    directoryButton(harness, "docs").getAttribute("aria-label"),
  );
});
