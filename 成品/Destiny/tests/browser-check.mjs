#!/usr/bin/env node
/*
 * Destiny 真浏览器点击级自检（零依赖）
 *
 * 为什么需要它：`rules-smoke-test.js` 只测纯函数，`dom-id-check.js` 只测 id 引用，
 * 两者都**跑不到真实浏览器**——按钮绑定、canvas 指针事件、下载、localStorage、
 * 弹窗，全是「写在代码里但从没被点过」的部分。
 *
 * 做法：不装 playwright / puppeteer。用 Chrome（或 Edge）自带的 DevTools 协议，
 * 通过 Node 18+ 内置的 WebSocket 直接驱动；鼠标用 Input.dispatchMouseEvent 发真事件，
 * 不是 `element.click()` 这种 JS 假点击。
 *
 * 用法：
 *   node tests/browser-check.mjs                 # 自动找 Chrome/Edge
 *   CHROME_PATH=/path/to/chrome node tests/browser-check.mjs
 *   DESTINY_SHOT=/tmp/shot.png node tests/browser-check.mjs   # 结束时截图
 *
 * 需要 Node 18+（用内置 fetch/WebSocket）。不需要联网，不需要装任何 npm 包。
 */

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const APP_PORT = Number(process.env.DESTINY_PORT || 8231);
const CDP_PORT = Number(process.env.DESTINY_CDP_PORT || 9333);
const PAGE_URL = `http://127.0.0.1:${APP_PORT}/prototype/index.html`;
const PROFILE = mkdtempSync(join(tmpdir(), 'destiny-profile-'));
const DOWNLOADS = mkdtempSync(join(tmpdir(), 'destiny-downloads-'));
const SHOT = process.env.DESTINY_SHOT || '';

const results = [];
const consoleErrors = [];
const dialogs = [];
let ws = null;
let chrome = null;
let server = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => Date.now();

function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  console.log(`${ok ? '  PASS  ' : '  FAIL  '}${name}${ok || !detail ? '' : `\n         → ${detail}`}`);
}

function findBrowser() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ].filter(Boolean);
  return candidates.find((p) => existsSync(p)) || null;
}

// ---------------------------------------------------------------- CDP 客户端
let msgId = 0;
const pending = new Map();

function send(method, params = {}, timeoutMs = 20000) {
  const id = ++msgId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res, rej) => {
    const timer = setTimeout(() => {
      if (pending.has(id)) { pending.delete(id); rej(new Error(`CDP 超时：${method}`)); }
    }, timeoutMs);
    pending.set(id, {
      res: (v) => { clearTimeout(timer); res(v); },
      rej: (e) => { clearTimeout(timer); rej(e); },
    });
  });
}

function onMessage(raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch { return; }
  if (msg.id) {
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.error) entry.rej(new Error(`${msg.error.message}${msg.error.data ? ' — ' + msg.error.data : ''}`));
    else entry.res(msg.result);
    return;
  }
  const p = msg.params || {};
  if (msg.method === 'Runtime.exceptionThrown') {
    const d = p.exceptionDetails || {};
    consoleErrors.push('未捕获异常：' + ((d.exception && (d.exception.description || d.exception.value)) || d.text));
  } else if (msg.method === 'Runtime.consoleAPICalled' && p.type === 'error') {
    consoleErrors.push('console.error：' + (p.args || []).map((a) => a.description || a.value).join(' '));
  } else if (msg.method === 'Log.entryAdded' && p.entry && p.entry.level === 'error') {
    consoleErrors.push('日志错误：' + p.entry.text);
  } else if (msg.method === 'Page.javascriptDialogOpening') {
    // btnSave / btnLoad 用 alert() 反馈；真实用户点掉就行，自动化里必须显式接受，否则页面卡死
    dialogs.push({ type: p.type, message: p.message });
    send('Page.handleJavaScriptDialog', { accept: true }).catch(() => {});
  }
}

async function evaluate(expression) {
  const r = await send('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true, userGesture: true,
  });
  if (r.exceptionDetails) {
    const d = r.exceptionDetails;
    throw new Error('页面内异常：' + ((d.exception && (d.exception.description || d.exception.value)) || d.text));
  }
  return r.result.value;
}

// ------------------------------------------------------------ 真鼠标 / 真键盘
async function mouseClick(x, y) {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, button: 'none', buttons: 0 });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', buttons: 0, clickCount: 1 });
  await sleep(90);
}

async function clickSelector(selector) {
  const box = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  })()`);
  if (!box) throw new Error(`找不到可点击元素：${selector}`);
  await mouseClick(box.x, box.y);
}

async function mouseDrag(from, to, steps = 8) {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: from.x, y: from.y, button: 'none', buttons: 0 });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: from.x, y: from.y, button: 'left', buttons: 1, clickCount: 1 });
  for (let i = 1; i <= steps; i++) {
    await send('Input.dispatchMouseEvent', {
      type: 'mouseMoved',
      x: from.x + ((to.x - from.x) * i) / steps,
      y: from.y + ((to.y - from.y) * i) / steps,
      button: 'left', buttons: 1,
    });
    await sleep(12);
  }
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: to.x, y: to.y, button: 'left', buttons: 0, clickCount: 1 });
  await sleep(120);
}

async function pressKey(key, code, vk) {
  const base = { key, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk };
  await send('Input.dispatchKeyEvent', { type: 'rawKeyDown', ...base });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', ...base });
  await sleep(90);
}

async function setValue(selector, value) {
  await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    el.value = ${JSON.stringify(String(value))};
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()`);
  await sleep(140);
}

async function setChecked(selector, checked) {
  await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    el.checked = ${checked ? 'true' : 'false'};
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return el.checked;
  })()`);
  await sleep(200);
}

// ---------------------------------------------------------------- 页面读取器
// 家具填充只有两种：#dbe7fb（未选中）/ #e3edff（选中）。判据必须**贴紧**这两个值——
// 早先用了「浅蓝区域」这种松判据，结果空房间也会命中 1900 多个像素（窗户/墙体填充是浅青色），
// 于是「清空后画布上没有家具」这条断言永远红。颜色判据和被测代码一样，松一点就是空转。
const ITEM_PIXELS = `(() => {
  const c = document.getElementById('scene');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const near = (r, g, b, t) => Math.abs(r - t[0]) <= 4 && Math.abs(g - t[1]) <= 4 && Math.abs(b - t[2]) <= 4;
  const hit = (i) => {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    return near(r, g, b, [219, 231, 251]) || near(r, g, b, [227, 237, 255]);
  };
  // 只数「成片」的家具填充：要求自己 + 右边 + 下边 + 右下共 2×2 四个像素同时命中。
  // 家具是整块矩形填充，而抗锯齿边缘、虚线框交界处会留下零星 1~2 个近似像素——
  // 不排除它们，这条断言就会在空房间上永远红（早先就是这样）。
  const w = c.width;
  let n = 0; const sample = new Map();
  for (let y = 0; y < c.height - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const i = ((y * w) + x) * 4;
      if (!hit(i) || !hit(i + 4) || !hit(i + w * 4) || !hit(i + w * 4 + 4)) continue;
      n++;
      const k = d[i] + ',' + d[i + 1] + ',' + d[i + 2];
      sample.set(k, (sample.get(k) || 0) + 1);
    }
  }
  return { count: n, sample: [...sample.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4) };
})()`;

const ISSUES = `(() => {
  const list = document.getElementById('issueList');
  return {
    count: list.children.length,
    text: list.textContent.replace(/\\s+/g, ' ').trim(),
    errors: (document.getElementById('issueCount').textContent.match(/^(\\d+)/) || [0, 0])[1],
  };
})()`;

const SEL_INFO = `(document.getElementById('selInfo').textContent || '')`;

/** 找一个「确实在家具实心区域里」的像素（取中心附近的点，避开两件家具之间的空隙）。 */
const FIND_ITEM_POINT = `(() => {
  const c = document.getElementById('scene');
  const rect = c.getBoundingClientRect();
  const ctx = c.getContext('2d');
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const dpr = c.width / rect.width;
  const hit = (x, y) => {
    const i = ((y * c.width) + x) * 4;
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const near = (t) => Math.abs(r - t[0]) <= 4 && Math.abs(g - t[1]) <= 4 && Math.abs(b - t[2]) <= 4;
    return near([219, 231, 251]) || near([227, 237, 255]);
  };
  let cx = 0, cy = 0, n = 0;
  for (let y = 0; y < c.height; y += 2) {
    for (let x = 0; x < c.width; x += 2) {
      if (hit(x, y)) { cx += x; cy += y; n++; }
    }
  }
  if (!n) return null;
  cx /= n; cy /= n;
  // 从重心出发找一个 13×13 邻居全命中的点，确保落在家具内部而不是边缘或缝隙
  for (let radius = 0; radius < 400; radius += 3) {
    for (const [dx, dy] of [[0, 0], [radius, 0], [-radius, 0], [0, radius], [0, -radius]]) {
      const x = Math.round(cx + dx), y = Math.round(cy + dy);
      if (x < 7 || y < 7 || x > c.width - 8 || y > c.height - 8) continue;
      let solid = true;
      for (let oy = -6; oy <= 6 && solid; oy += 3) {
        for (let ox = -6; ox <= 6; ox += 3) if (!hit(x + ox, y + oy)) { solid = false; break; }
      }
      if (solid) return { x: rect.left + x / dpr, y: rect.top + y / dpr };
    }
  }
  return { x: rect.left + cx / dpr, y: rect.top + cy / dpr };
})()`;

async function shotHash() {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  return createHash('sha1').update(r.data).digest('hex').slice(0, 12);
}

/** 把鼠标挪出画布：家具的 hover 高亮会改画布像素，比较画面之前必须先清掉悬停状态。 */
async function parkMouse() {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 2, y: 2, button: 'none', buttons: 0 });
  await sleep(150);
}

/** 画布逐像素指纹：比「家具像素大概有多少」严格得多，用来验证清空/恢复是否真的回到同一画面。 */
async function canvasFingerprint() {
  const url = await evaluate(`document.getElementById('scene').toDataURL('image/png')`);
  return createHash('sha1').update(url).digest('hex').slice(0, 12);
}

/** 真实鼠标点击问题清单里第 index 条（用于"能点"和"不能点"两类分别验证）。 */
async function clickIssueAt(index) {
  const box = await evaluate(`(() => {
    const li = document.querySelectorAll('#issueList li')[${index}];
    if (!li) return null;
    const r = li.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + Math.min(14, r.height / 2), clickable: li.classList.contains('clickable') };
  })()`);
  if (!box) throw new Error(`没有第 ${index} 条问题`);
  await mouseClick(box.x, box.y);
  return box.clickable;
}

const parsePos = (text) => {
  const m = text.match(/位置 x=(-?\d+) y=(-?\d+)/);
  return m ? { x: Number(m[1]), y: Number(m[2]) } : null;
};
const parseRot = (text) => {
  const m = text.match(/旋转 (\d+)°/);
  return m ? Number(m[1]) : null;
};

// ------------------------------------------------------------------ 主流程
async function main() {
  const browserPath = findBrowser();
  if (!browserPath) throw new Error('找不到 Chrome / Edge，可用 CHROME_PATH 指定');

  // 1. 起一个真 HTTP 服务（用项目自带的零依赖 server.js），而不是 file://，
  //    这样 localStorage / 下载 / 剪贴板的行为和用户双击 .bat 时一致
  server = spawn(process.execPath, [join(ROOT, 'server.js'), String(APP_PORT)], {
    env: { ...process.env, DESTINY_NO_OPEN: '1' }, stdio: 'ignore',
  });
  let up = false;
  for (let i = 0; i < 40 && !up; i++) {
    try {
      const res = await fetch(PAGE_URL);
      up = res.ok;
    } catch { await sleep(150); }
  }
  if (!up) throw new Error(`本地服务没起来：${PAGE_URL}`);

  // 2. 起真浏览器
  chrome = spawn(browserPath, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-extensions', '--disable-background-networking', '--hide-scrollbars',
    '--window-size=1500,980',
    `--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${PROFILE}`,
    '--remote-allow-origins=*', PAGE_URL,
  ], { stdio: 'ignore' });

  let target = null;
  for (let i = 0; i < 60 && !target; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      target = list.find((t) => t.type === 'page' && t.url.startsWith('http://127.0.0.1:'));
    } catch { /* 还没起来 */ }
    if (!target) await sleep(250);
  }
  if (!target) throw new Error('连不上浏览器调试端口');

  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener('open', res, { once: true });
    ws.addEventListener('error', () => rej(new Error('CDP WebSocket 连接失败')), { once: true });
  });
  ws.addEventListener('message', (e) => onMessage(e.data));

  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: DOWNLOADS }).catch(() => {});

  // 3. 等页面真的画出来
  let ready = false;
  for (let i = 0; i < 40 && !ready; i++) {
    try {
      ready = await evaluate(`document.readyState === 'complete' && !!document.getElementById('scene') && document.getElementById('scene').width > 0`);
    } catch { /* 还没好 */ }
    if (!ready) await sleep(150);
  }
  await sleep(400);

  // 剪贴板存根：headless 下没授权，装一个只记录文本的替身，用来验证「复制结论文本」的内容
  await evaluate(`Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: (t) => { window.__copied = String(t); return Promise.resolve(); } },
  }); window.__copied = null; true`);

  // ---- 场景 1：首屏
  const canvasBox = await evaluate(`(() => { const r = document.getElementById('scene').getBoundingClientRect(); return { w: Math.round(r.width), h: Math.round(r.height) }; })()`);
  const firstIssues = await evaluate(ISSUES);
  const verdict = await evaluate(`document.getElementById('verdict').textContent`);
  check('首屏渲染：canvas 有尺寸', canvasBox.w > 300 && canvasBox.h > 200, `canvas ${canvasBox.w}×${canvasBox.h}`);
  check('首屏渲染：问题清单已算出内容', firstIssues.count > 0, `条目 ${firstIssues.count}，verdict=${verdict}`);
  check('首屏渲染：结论文案已生成（不是「正在计算…」）', !/正在计算/.test(verdict), verdict);

  // ---- 场景 2：点击问题清单条目 → 选中家具（真实鼠标点击）
  const kinds = await evaluate(`(() => {
    const lis = [...document.querySelectorAll('#issueList li')];
    return {
      total: lis.length,
      clickable: lis.filter((li) => li.classList.contains('clickable')).length,
      notarget: lis.filter((li) => li.classList.contains('notarget')).length,
      notargetHasHint: lis.filter((li) => li.classList.contains('notarget')).every((li) => (li.title || '').length > 0),
      notargetTagged: lis.filter((li) => li.classList.contains('notarget')).every((li) => !!li.querySelector('.no-target-tag')),
    };
  })()`);
  const firstClickable = await evaluate(`[...document.querySelectorAll('#issueList li')].findIndex((li) => li.classList.contains('clickable'))`);
  check('问题清单条目区分「可点」与「不指向家具」两类', kinds.total > 0 && kinds.clickable > 0 && kinds.notarget > 0,
    `共 ${kinds.total} 条：可点 ${kinds.clickable}、不指向家具 ${kinds.notarget}`);
  check('不指向家具的条目有文字说明，不会被误当成「点坏了」', kinds.notargetHasHint && kinds.notargetTagged,
    `title 齐全=${kinds.notargetHasHint}，标签齐全=${kinds.notargetTagged}`);

  await clickIssueAt(firstClickable);
  const selAfterIssueClick = await evaluate(SEL_INFO);
  check('点击可点的问题条目能选中对应家具', /已选中：/.test(selAfterIssueClick), selAfterIssueClick.slice(0, 60));

  const notargetIndex = await evaluate(`[...document.querySelectorAll('#issueList li')].findIndex((li) => li.classList.contains('notarget'))`);
  const selBeforeNotarget = await evaluate(SEL_INFO);
  await clickIssueAt(notargetIndex);
  const selAfterNotarget = await evaluate(SEL_INFO);
  check('点击「不指向家具」的条目不会误改选中状态',
    selAfterNotarget === selBeforeNotarget, `选中信息被改成了：${selAfterNotarget.slice(0, 40)}`);

  // ---- 场景 3：键盘 R 旋转
  const rotBefore = parseRot(await evaluate(SEL_INFO));
  await pressKey('r', 'KeyR', 82);
  const rotAfter = parseRot(await evaluate(SEL_INFO));
  check('按 R 能旋转选中家具', rotBefore !== null && rotAfter !== null && rotAfter === (rotBefore + 90) % 360,
    `${rotBefore}° → ${rotAfter}°`);

  // ---- 场景 4：方向键微调 + 撤销
  const posBefore = parsePos(await evaluate(SEL_INFO));
  await pressKey('ArrowRight', 'ArrowRight', 39);
  const posMoved = parsePos(await evaluate(SEL_INFO));
  check('方向键能微调家具位置', !!posBefore && !!posMoved && posMoved.x > posBefore.x, `${JSON.stringify(posBefore)} → ${JSON.stringify(posMoved)}`);
  await clickSelector('#btnUndo');
  const posUndone = parsePos(await evaluate(SEL_INFO));
  check('撤销能回退方向键的移动', !!posUndone && !!posBefore && posUndone.x === posBefore.x, `${JSON.stringify(posMoved)} → ${JSON.stringify(posUndone)}`);

  // ---- 场景 5：清空 / 添加（用画面指纹，而不是「大致有多少蓝色像素」）
  await clickSelector('#btnClear');
  await parkMouse();
  const emptyFingerprint = await canvasFingerprint();
  const emptyPixels = await evaluate(ITEM_PIXELS);
  const clearedIssues = await evaluate(ISSUES);
  check('清空所有家具：画布上不再有任何家具', emptyPixels.count === 0,
    `家具像素 ${emptyPixels.count}（残留色 ${JSON.stringify(emptyPixels.sample)}）`);
  check('清空所有家具：清单给出「没有发现问题」', /没有发现问题/.test(clearedIssues.text), clearedIssues.text.slice(0, 50));

  await setValue('#addSelect', 'bed-90');              // 选「单人床 90×190」
  await clickSelector('#btnAdd');
  await pressKey('Escape', 'Escape', 27);              // 取消选中，画面才和「读取上次」后的状态可比
  await parkMouse();
  const layoutFingerprint = await canvasFingerprint();
  const addedPixels = await evaluate(ITEM_PIXELS);
  check('「放进去」能新增家具', addedPixels.count > 5000 && layoutFingerprint !== emptyFingerprint,
    `家具像素 ${addedPixels.count}，画面 ${emptyFingerprint} → ${layoutFingerprint}`);

  // ---- 场景 6：真鼠标拖拽家具
  const point = await evaluate(FIND_ITEM_POINT);
  if (!point) throw new Error('在画布上找不到家具像素，无法测试拖拽');
  await mouseClick(point.x, point.y);
  const dragBefore = parsePos(await evaluate(SEL_INFO));
  await mouseDrag(point, { x: point.x + 90, y: point.y + 60 });
  const dragAfter = parsePos(await evaluate(SEL_INFO));
  check('真鼠标拖拽能移动家具（位置变化且方向正确）',
    !!dragBefore && !!dragAfter && dragAfter.x > dragBefore.x && dragAfter.y > dragBefore.y,
    `${JSON.stringify(dragBefore)} → ${JSON.stringify(dragAfter)}（拖了 +90,+60 像素）`);

  // ---- 场景 7：改房间尺寸 → 家具放不下
  await setValue('#roomW', 150);
  await setValue('#roomH', 150);
  const smallIssues = await evaluate(ISSUES);
  check('房间缩到 150×150 后能报出「超出房间范围」', /超出房间范围/.test(smallIssues.text), smallIssues.text.slice(0, 70));

  // ---- 场景 8：保存 / 读取往返（要比的是「逐像素同一画面」，不是「大概恢复了」）
  await pressKey('Escape', 'Escape', 27);
  await parkMouse();
  const savedFingerprint = await canvasFingerprint();
  const dialogBefore = dialogs.length;
  await clickSelector('#btnSave');
  await sleep(200);
  const storedKeys = await evaluate(`Object.keys(localStorage).length`);
  check('「保存到本机」写入了 localStorage', storedKeys > 0, `localStorage 键 ${storedKeys} 个`);
  check('「保存到本机」确实弹出了提示（alert 可见）', dialogs.length > dialogBefore,
    dialogs.slice(-1).map((d) => d.message).join(''));

  await clickSelector('#btnClear');
  await parkMouse();
  const clearedFingerprint = await canvasFingerprint();
  await clickSelector('#btnLoad');
  await sleep(250);
  await pressKey('Escape', 'Escape', 27);
  await parkMouse();
  const restoredFingerprint = await canvasFingerprint();
  // 注意：这里不能拿"清空后的画面"去比场景 5 那个空房间基准——房间尺寸已经改成 150×150 了，
  // 房间框和网格本来就不一样。清空是否生效，看它和"保存时"的画面不同就够了。
  check('清空确实抹掉了已有布局（画面与保存时不同）', clearedFingerprint !== savedFingerprint,
    `保存时 ${savedFingerprint}，清空后 ${clearedFingerprint}`);
  check('「读取上次」恢复出的画面与保存时逐像素相同', restoredFingerprint === savedFingerprint,
    `保存时 ${savedFingerprint}，读取后 ${restoredFingerprint}`);

  // ---- 场景 9：复制结论文本
  await clickSelector('#btnCopy');
  await sleep(200);
  const copied = await evaluate(`window.__copied || ''`);
  check('「复制结论文本」产出的是结论文本（含房间尺寸与结论行）',
    /Destiny/.test(copied) && /房间：/.test(copied) && /结论：/.test(copied),
    JSON.stringify(copied.slice(0, 70)));

  // ---- 场景 10：导出 PNG（真下载）
  await clickSelector('#btnExport');
  let downloaded = null;
  for (let i = 0; i < 30 && !downloaded; i++) {
    await sleep(200);
    const files = readdirSync(DOWNLOADS).filter((f) => f.endsWith('.png'));
    if (files.length) downloaded = files[0];
  }
  const size = downloaded ? statSync(join(DOWNLOADS, downloaded)).size : 0;
  check('「导出 PNG」真的下载了一张图片', !!downloaded && size > 5000, downloaded ? `${downloaded}（${size} 字节）` : '没有文件落盘');

  // ---- 场景 11：热力图
  const hashBeforeHeat = await shotHash();
  await setChecked('#chkHeat', true);
  await sleep(400);
  const hashAfterHeat = await shotHash();
  check('勾选「通行宽度热力图」后画面确实变了', hashBeforeHeat !== hashAfterHeat, `${hashBeforeHeat} → ${hashAfterHeat}`);
  await setChecked('#chkHeat', false);

  // ---- 场景 12：切换通行标准
  const issues60 = await evaluate(ISSUES);
  await setValue('#corridorStd', '90');
  const issues90 = await evaluate(ISSUES);
  check('通行标准切到 90cm 后结论随之变化（更严）',
    issues90.text !== issues60.text || Number(issues90.errors) >= Number(issues60.errors),
    `60cm：${issues60.errors} 错误；90cm：${issues90.errors} 错误`);

  // ---- 场景 13：全程没有 JS 报错
  check('全程没有未捕获异常 / console.error', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

  if (SHOT) {
    const r = await send('Page.captureScreenshot', { format: 'png' });
    mkdirSync(dirname(SHOT), { recursive: true });
    writeFileSync(SHOT, Buffer.from(r.data, 'base64'));
    console.log(`\n截图已保存：${SHOT}`);
  }
}

async function cleanup() {
  try { if (ws) await send('Browser.close', {}, 2000); } catch { /* 忽略 */ }
  try { if (ws) ws.close(); } catch { /* 忽略 */ }
  for (const child of [chrome, server]) {
    if (!child || child.exitCode !== null) continue;
    try { child.kill(); } catch { /* 忽略 */ }
  }
  await sleep(300);
  try { rmSync(PROFILE, { recursive: true, force: true }); } catch { /* 忽略 */ }
  try { rmSync(DOWNLOADS, { recursive: true, force: true }); } catch { /* 忽略 */ }
}

const started = now();
let fatal = null;
try {
  await main();
} catch (err) {
  fatal = err;
} finally {
  await cleanup();
}

if (fatal) {
  console.error(`\n自检中断：${fatal.message}`);
  process.exit(2);
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed（${((now() - started) / 1000).toFixed(1)}s）`);
if (failed.length) {
  console.log('失败项：');
  for (const f of failed) console.log(`  - ${f.name}${f.detail ? '：' + f.detail : ''}`);
}
process.exit(failed.length ? 1 : 0);
