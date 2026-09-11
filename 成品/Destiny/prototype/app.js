/* Destiny 原型 v0.1 —— 界面与交互（零依赖，纯本地）
 * 许可：本文件为 KineWorld Destiny 项目自写代码，MIT 许可。
 * 说明：所有计算在浏览器本地完成，不联网、不上传、不调用任何模型。
 */
(function () {
  'use strict';

  var Catalog = window.Catalog, Rules = window.Rules;
  var STORE_KEY = 'destiny.layout.v1';
  var SNAP = 5;

  var state = {
    room: { w: 600, h: 360 },
    items: [],
    door: { wall: 'bottom', offset: 60, width: 90, hinge: 'start' },
    window: { wall: 'top', offset: 180, width: 180 },
    sockets: [],
    selected: null,
    hover: [],
    socketMode: false,
    templateId: 'dorm-4'
  };

  var view = { scale: 1, ox: 0, oy: 0 };
  var history = [];
  var seq = 1;
  var result = null;
  var heatCanvas = null, heatKey = '';

  var $ = function (id) { return document.getElementById(id); };
  var canvas = $('scene');
  var ctx = canvas.getContext('2d');

  // ---------- 场景构造 ----------
  function newItem(typeId, x, y, rot) {
    var t = Catalog.find(typeId);
    if (!t) return null;
    return { id: 'f' + (seq++), type: typeId, w: t.w, d: t.d, x: x, y: y, rot: rot || 0 };
  }

  function loadTemplate(id, keepSizes) {
    var tpl = Catalog.template(id);
    state.templateId = tpl.id;
    if (!keepSizes) { state.room = { w: tpl.w, h: tpl.h }; }
    state.door = JSON.parse(JSON.stringify(tpl.door));
    state.window = JSON.parse(JSON.stringify(tpl.window));
    state.sockets = [];
    state.items = (tpl.presetItems || []).map(function (p) { return newItem(p.type, p.x, p.y, p.rot); }).filter(Boolean);
    state.selected = null;
    syncInputs();
    update();
  }

  function snapshot() {
    // selected 也要进快照：否则撤销一次就把选中状态清掉，用户得重新点一遍家具。
    history.push(JSON.stringify({
      room: state.room, items: state.items, door: state.door,
      window: state.window, sockets: state.sockets, selected: state.selected
    }));
    if (history.length > 60) history.shift();
  }

  function undo() {
    var s = history.pop();
    if (!s) return;
    var o = JSON.parse(s);
    state.room = o.room; state.items = o.items; state.door = o.door; state.window = o.window; state.sockets = o.sockets;
    // 恢复当时的选中项（那件家具可能已经不在——例如撤销"删除"之前它本来就在，撤销后就该重新选中它）
    var stillThere = o.selected && o.items.some(function (x) { return x.id === o.selected; });
    state.selected = stillThere ? o.selected : null;
    syncInputs(); update();
  }

  // ---------- 计算与刷新 ----------
  function update() {
    result = Rules.analyze(state);
    render();
    renderReport();
  }

  function renderReport() {
    var v = $('verdict');
    v.textContent = result.summary.verdict + '（共 ' + result.issues.length + ' 条）';
    v.className = 'verdict ' + (result.summary.errors ? 'error' : (result.summary.warns ? 'warn' : 'ok'));
    $('issueCount').textContent = result.summary.errors + ' 错误 / ' + result.summary.warns + ' 提示';

    var list = $('issueList');
    list.innerHTML = '';
    if (!result.issues.length) {
      var li0 = document.createElement('li');
      li0.innerHTML = '<b>没有发现问题</b><span>在当前规则与建议值下，这套摆放可行。注意：这不代表设计合理，只代表没有触发我们列出的检查项。</span>';
      list.appendChild(li0);
    }
    result.issues.forEach(function (iss) {
      var li = document.createElement('li');
      // 条目分两类：指向具体家具的可以点（点了在图上高亮），整间房/通道类的点不了。
      // 以前两类长得一模一样、都没有指针提示，点了没反应会被当成"点坏了"。
      var points = iss.targets.length > 0;
      li.className = iss.level + (points ? ' clickable' : ' notarget');
      li.title = points
        ? '点击可在图上高亮这条问题指向的家具'
        : '这条问题涉及整间房间或通道宽度，不指向某一件家具——请对照图上的红框自行判断';
      li.innerHTML = '<b>' + (iss.level === 'error' ? '✕ ' : '! ') + escapeHtml(iss.title) + '</b><span>' + escapeHtml(iss.detail) + '</span>'
        + (points ? '' : '<span class="no-target-tag">不指向单件家具</span>');
      li.onmouseenter = function () { state.hover = iss.targets.slice(); render(); };
      li.onmouseleave = function () { state.hover = []; render(); };
      if (points) {
        li.onclick = function () {
          state.selected = iss.targets[0]; state.hover = iss.targets.slice(); render();
        };
      }
      list.appendChild(li);
    });

    var bl = $('bottleneckList');
    bl.innerHTML = '';
    if (result.summary.bottlenecks.length === 0) {
      bl.innerHTML = '<li><span>暂无（未找到可通行路径）</span></li>';
    } else {
      result.summary.bottlenecks.forEach(function (b) {
        var li = document.createElement('li');
        var std = Catalog.THRESHOLDS.corridorMin;
        var cls = b.width < std ? 'color:var(--error)'
          : (b.width < Catalog.THRESHOLDS.corridorComfort ? 'color:var(--warn)' : 'color:var(--ok)');
        li.innerHTML = '<span>门口 → ' + escapeHtml(b.name) + '</span><span style="' + cls + '">最窄约 ' + b.width + ' cm</span>';
        bl.appendChild(li);
      });
    }
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(function () { fallbackCopy(text); });
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) { window.prompt('复制失败，请手动复制下面的内容：', text); }
    document.body.removeChild(ta);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // ---------- 绘制 ----------
  function computeView() {
    var dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || 800, cssH = canvas.clientHeight || 480;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var pad = 28;
    var s = Math.min((cssW - pad * 2) / state.room.w, (cssH - pad * 2) / state.room.h);
    view.scale = s;
    view.ox = (cssW - state.room.w * s) / 2;
    view.oy = (cssH - state.room.h * s) / 2;
  }

  function X(x) { return view.ox + x * view.scale; }
  function Y(y) { return view.oy + y * view.scale; }
  function S(v) { return v * view.scale; }
  function toRoom(px, py) {
    var r = canvas.getBoundingClientRect();
    return { x: (px - r.left - view.ox) / view.scale, y: (py - r.top - view.oy) / view.scale };
  }

  function render() {
    computeView();
    var cssW = canvas.clientWidth, cssH = canvas.clientHeight;
    ctx.clearRect(0, 0, cssW, cssH);

    if ($('chkHeat').checked && result && result.field) drawHeat(result.field);

    if ($('chkGrid').checked) drawGrid();

    // 使用空间（虚线）
    if ($('chkZones').checked) {
      var wr = Rules.windowInnerRect(state.window, state.room);
      if (wr) {
        ctx.save();
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = 'rgba(47,109,246,.45)';
        ctx.fillStyle = 'rgba(47,109,246,.06)';
        ctx.fillRect(X(wr.x), Y(wr.y), S(wr.w), S(wr.d));
        ctx.strokeRect(X(wr.x), Y(wr.y), S(wr.w), S(wr.d));
        ctx.restore();
      }
      state.items.forEach(function (it) {
        var type = Catalog.find(it.type); if (!type) return;
        var ar = Rules.activityRect(it, type);
        if (ar) {
          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = 'rgba(217,131,36,.75)';
          ctx.fillStyle = 'rgba(217,131,36,.10)';
          ctx.fillRect(X(ar.x), Y(ar.y), S(ar.w), S(ar.d));
          ctx.strokeRect(X(ar.x), Y(ar.y), S(ar.w), S(ar.d));
          ctx.restore();
        }
        Rules.sideBands(it, type).forEach(function (b) {
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = 'rgba(217,131,36,.45)';
          ctx.strokeRect(X(b.x), Y(b.y), S(b.w), S(b.d));
          ctx.restore();
        });
      });
    }

    drawWalls();

    // 家具
    state.items.forEach(function (it) {
      var type = Catalog.find(it.type); if (!type) return;
      var r = Rules.rectOf(it);
      var isSel = state.selected === it.id;
      var isHover = state.hover.indexOf(it.id) >= 0;
      ctx.fillStyle = isSel ? '#e3edff' : '#dbe7fb';
      ctx.strokeStyle = isHover ? '#d64545' : (isSel ? '#2f6df6' : '#5b7bb5');
      ctx.lineWidth = isHover ? 3 : (isSel ? 2 : 1);
      roundRect(X(r.x), Y(r.y), S(r.w), S(r.d), 3);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#1c2530';
      ctx.font = '12px "Microsoft YaHei", sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(type.name, X(r.x + r.w / 2), Y(r.y + r.d / 2) - 6);
      ctx.fillStyle = '#6b7684';
      ctx.font = '11px "Microsoft YaHei", sans-serif';
      ctx.fillText(r.w + '×' + r.d, X(r.x + r.w / 2), Y(r.y + r.d / 2) + 9);
    });

    // 插座
    state.sockets.forEach(function (s) {
      ctx.beginPath();
      ctx.arc(X(s.x), Y(s.y), 4, 0, Math.PI * 2);
      ctx.fillStyle = '#2e9e5b'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
    });

    // 可拖拽的手柄：门/窗两端常显，家具在选中后显示
    ['window', 'door'].forEach(function (kind) {
      var op = state[kind];
      if (!op || !(op.width > 0)) return;
      drawHandles(openingHandles(op), kind === 'door' ? '#d64545' : '#3b82f6');
    });
    var selIt = state.selected ? findItem(state.selected) : null;
    if (selIt) drawHandles(itemHandles(selIt), '#2f6df6');

    renderSelInfo();
  }

  function drawHandles(list, color) {
    if (!list || !list.length) return;
    ctx.save();
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = color || '#2f6df6';
    ctx.lineWidth = 1.5;
    list.forEach(function (h) {
      ctx.beginPath();
      ctx.rect(X(h.x) - HANDLE_PX / 2, Y(h.y) - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
      ctx.fill(); ctx.stroke();
    });
    ctx.restore();
  }

  function renderSelInfo() {
    var el = $('selInfo');
    if (!el) return;
    var it = findItem(state.selected);
    if (!it) { el.textContent = ''; return; }
    var t = Catalog.find(it.type);
    var r = Rules.rectOf(it);
    var custom = (t.w !== it.w || t.d !== it.d);
    el.textContent = '已选中：' + t.name + '，占地 ' + r.w + '×' + r.d + 'cm，高 ' + t.h +
      'cm，旋转 ' + it.rot + '°，位置 x=' + Math.round(it.x) + ' y=' + Math.round(it.y) + '。' +
      (custom ? '尺寸已改（默认 ' + t.w + '×' + t.d + 'cm，可用「恢复默认尺寸」还原）。' : '') +
      ' 拖四角或四边的方块可改尺寸；按 R 旋转，Delete 删除，Esc 取消选中。';
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function drawGrid() {
    ctx.save();
    ctx.strokeStyle = '#eef1f5'; ctx.lineWidth = 1;
    for (var x = 0; x <= state.room.w; x += 50) { line(X(x), Y(0), X(x), Y(state.room.h)); }
    for (var y = 0; y <= state.room.h; y += 50) { line(X(0), Y(y), X(state.room.w), Y(y)); }
    ctx.strokeStyle = '#e2e7ee';
    line(X(0), Y(0), X(state.room.w), Y(0));
    ctx.restore();
    ctx.fillStyle = '#8d97a5';
    ctx.font = '11px "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(state.room.w + ' cm', X(state.room.w / 2), Y(0) - 12);
    ctx.save();
    ctx.translate(X(0) - 12, Y(state.room.h / 2)); ctx.rotate(-Math.PI / 2);
    ctx.fillText(state.room.h + ' cm', 0, 0);
    ctx.restore();
  }

  function line(x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }

  function drawWalls() {
    var wpx = 8;
    ctx.save();
    ctx.strokeStyle = '#334155'; ctx.lineWidth = wpx;
    ctx.strokeRect(X(0), Y(0), S(state.room.w), S(state.room.h));
    ctx.restore();

    // 门：抹掉墙段 + 画开启扇形
    var dg = Rules.openingGeom(state.door, state.room);
    if (dg) {
      eraseOpening(dg, state.door.width);
      var width = state.door.width;
      var hinge = state.door.hinge === 'end' ? dg.b : dg.a;
      var other = state.door.hinge === 'end' ? dg.a : dg.b;
      var a0 = Math.atan2(other.y - hinge.y, other.x - hinge.x);
      var a1 = Math.atan2(dg.inward.y, dg.inward.x);
      var delta = a1 - a0;
      while (delta <= -Math.PI) delta += Math.PI * 2;
      while (delta > Math.PI) delta -= Math.PI * 2;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(X(hinge.x), Y(hinge.y));
      ctx.arc(X(hinge.x), Y(hinge.y), S(width), a0, a1, delta < 0);
      ctx.closePath();
      ctx.fillStyle = 'rgba(214,69,69,.10)'; ctx.fill();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(214,69,69,.55)'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.restore();
      // 开启后的门扇
      ctx.save();
      ctx.strokeStyle = '#334155'; ctx.lineWidth = 3;
      line(X(hinge.x), Y(hinge.y), X(hinge.x + dg.inward.x * width), Y(hinge.y + dg.inward.y * width));
      ctx.restore();
    }

    // 窗
    var wg = Rules.openingGeom(state.window, state.room);
    if (wg && state.window.width > 0) {
      eraseOpening(wg, state.window.width);
      ctx.save();
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 3;
      var inset = 2;
      var a = { x: wg.a.x + wg.inward.x * inset, y: wg.a.y + wg.inward.y * inset };
      var b = { x: wg.b.x + wg.inward.x * inset, y: wg.b.y + wg.inward.y * inset };
      line(X(a.x), Y(a.y), X(b.x), Y(b.y));
      ctx.restore();
    }
  }

  function eraseOpening(g, width) {
    var t = 12; // 抹掉的墙体厚度(px)
    ctx.save();
    ctx.fillStyle = '#fff';
    var x, y, w, h;
    if (g.axis.x === 1) { x = Math.min(g.a.x, g.b.x); w = width; y = g.a.y - 6 / view.scale; h = 12 / view.scale; }
    else { y = Math.min(g.a.y, g.b.y); h = width; x = g.a.x - 6 / view.scale; w = 12 / view.scale; }
    ctx.fillRect(X(x), Y(y), S(w), S(h));
    ctx.restore();
  }

  function drawHeat(field) {
    var key = field.gw + 'x' + field.gh + ':' + state.items.length + ':' + state.items.map(function (i) { return i.x + ',' + i.y + ',' + i.rot; }).join('|');
    if (!heatCanvas || heatKey !== key) {
      heatCanvas = document.createElement('canvas');
      heatCanvas.width = field.gw; heatCanvas.height = field.gh;
      var hc = heatCanvas.getContext('2d');
      var img = hc.createImageData(field.gw, field.gh);
      var half = Catalog.THRESHOLDS.gridCell / 2;
      for (var i = 0; i < field.blocked.length; i++) {
        var o = i * 4;
        if (field.blocked[i]) { img.data[o] = 90; img.data[o + 1] = 96; img.data[o + 2] = 105; img.data[o + 3] = 200; continue; }
        var width = field.dist[i] * 2;
        var std = Catalog.THRESHOLDS.corridorMin;
        if (width < std) { img.data[o] = 214; img.data[o + 1] = 69; img.data[o + 2] = 69; }
        else if (width < Catalog.THRESHOLDS.corridorComfort) { img.data[o] = 230; img.data[o + 1] = 175; img.data[o + 2] = 46; }
        else { img.data[o] = 46; img.data[o + 1] = 158; img.data[o + 2] = 91; }
        img.data[o + 3] = 70;
      }
      hc.putImageData(img, 0, 0);
      heatKey = key;
    }
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    var p = field.pad || 0;
    ctx.drawImage(heatCanvas, p, p, field.gw - p * 2, field.gh - p * 2, X(0), Y(0), S(state.room.w), S(state.room.h));
    ctx.restore();
  }

  // ---------- 交互：拖动与缩放（家具、门、窗都可以用鼠标改） ----------
  var HANDLE_PX = 9;      // 手柄在屏幕上的边长(px)
  var MIN_SIZE = 20;      // 家具最小边(cm)
  var DOOR_MIN = 60, DOOR_MAX = 120;
  var WIN_MIN = 30;
  var OPENING_GRAB = 15;  // 门/窗的抓取带宽(cm)

  var drag = null;

  function inRect(p, r) {
    return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.d;
  }

  function distToSegment(p, a, b) {
    var vx = b.x - a.x, vy = b.y - a.y;
    var L2 = vx * vx + vy * vy;
    var t = L2 ? ((p.x - a.x) * vx + (p.y - a.y) * vy) / L2 : 0;
    t = Math.max(0, Math.min(1, t));
    var dx = p.x - (a.x + t * vx), dy = p.y - (a.y + t * vy);
    return Math.sqrt(dx * dx + dy * dy);
  }

  // 家具的 8 个手柄：4 个角同时改两边，4 条边只改一边
  function itemHandles(it) {
    var r = Rules.rectOf(it);
    var mx = r.x + r.w / 2, my = r.y + r.d / 2;
    return [
      { name: 'nw', x: r.x, y: r.y, cursor: 'nwse-resize' },
      { name: 'se', x: r.x + r.w, y: r.y + r.d, cursor: 'nwse-resize' },
      { name: 'ne', x: r.x + r.w, y: r.y, cursor: 'nesw-resize' },
      { name: 'sw', x: r.x, y: r.y + r.d, cursor: 'nesw-resize' },
      { name: 'n', x: mx, y: r.y, cursor: 'ns-resize' },
      { name: 's', x: mx, y: r.y + r.d, cursor: 'ns-resize' },
      { name: 'w', x: r.x, y: my, cursor: 'ew-resize' },
      { name: 'e', x: r.x + r.w, y: my, cursor: 'ew-resize' }
    ];
  }

  function openingHandles(op) {
    var g = Rules.openingGeom(op, state.room);
    if (!g) return [];
    return [{ name: 'start', x: g.a.x, y: g.a.y }, { name: 'end', x: g.b.x, y: g.b.y }];
  }

  function hitHandle(p, handles) {
    var tol = HANDLE_PX / view.scale;
    for (var i = 0; i < handles.length; i++) {
      if (Math.abs(p.x - handles[i].x) <= tol && Math.abs(p.y - handles[i].y) <= tol) return handles[i];
    }
    return null;
  }

  function snap(v) { return Math.round(v / SNAP) * SNAP; }

  function resizeItem(id, handle, p) {
    var it = findItem(id); if (!it) return;
    var r = Rules.rectOf(it);
    var nr = Rules.resizeRect(r, handle, { x: snap(p.x), y: snap(p.y) }, state.room, MIN_SIZE);
    if (!nr) return;
    var swap = (it.rot === 90 || it.rot === 270);
    it.x = nr.x; it.y = nr.y;
    // 旋转 90/270 时，屏幕上的宽对应家具的"深"，别存反
    if (swap) { it.w = nr.d; it.d = nr.w; } else { it.w = nr.w; it.d = nr.d; }
  }

  function moveOpening(which, p, grab) {
    var op = state[which];
    op.wall = Rules.nearestWall(p, state.room, op.wall);
    var along = snap(Rules.projectOnWall(p, op.wall) - grab);
    op.offset = Rules.clampOpeningOffset(op, along, state.room);
  }

  function resizeOpening(which, end, p) {
    var op = state[which];
    var v = snap(Rules.projectOnWall(p, op.wall));
    var opts = (which === 'door') ? { min: DOOR_MIN, max: DOOR_MAX } : { min: WIN_MIN };
    var res = Rules.resizeOpening(op, end, v, state.room, opts);
    op.offset = res.offset;
    op.width = res.width;
  }

  canvas.addEventListener('pointerdown', function (e) {
    var p = toRoom(e.clientX, e.clientY);
    if (state.socketMode) {
      var hitIdx = -1;
      for (var i = 0; i < state.sockets.length; i++) {
        if (Math.abs(state.sockets[i].x - p.x) < 12 && Math.abs(state.sockets[i].y - p.y) < 12) { hitIdx = i; break; }
      }
      snapshot();
      if (hitIdx >= 0) state.sockets.splice(hitIdx, 1);
      else state.sockets.push({ x: Math.round(p.x), y: Math.round(p.y) });
      update();
      return;
    }

    // 1. 选中家具的手柄 → 伸缩
    var sel = state.selected ? findItem(state.selected) : null;
    if (sel) {
      var h = hitHandle(p, itemHandles(sel));
      if (h) {
        snapshot();
        drag = { kind: 'itemResize', id: sel.id, handle: h.name };
        canvas.setPointerCapture(e.pointerId);
        render();
        return;
      }
    }

    // 2. 家具本体 → 移动
    for (var k = state.items.length - 1; k >= 0; k--) {
      var r = Rules.rectOf(state.items[k]);
      if (inRect(p, r)) {
        state.selected = state.items[k].id;
        snapshot();
        drag = { kind: 'itemMove', id: state.items[k].id, dx: p.x - state.items[k].x, dy: p.y - state.items[k].y };
        canvas.setPointerCapture(e.pointerId);
        render();
        return;
      }
    }

    // 3. 门/窗两端的手柄 → 伸缩
    var kinds = ['door', 'window'];
    for (var ki = 0; ki < kinds.length; ki++) {
      var op = state[kinds[ki]];
      if (!op || !(op.width > 0)) continue;
      var hs = openingHandles(op);
      var hh = hitHandle(p, hs);
      if (hh) {
        snapshot();
        drag = { kind: 'openingResize', which: kinds[ki], end: hh.name };
        canvas.setPointerCapture(e.pointerId);
        render();
        return;
      }
    }

    // 4. 门/窗本体 → 沿墙移动（拖到别的墙会自动换墙）
    for (var kj = 0; kj < kinds.length; kj++) {
      var op2 = state[kinds[kj]];
      if (!op2 || !(op2.width > 0)) continue;
      var g2 = Rules.openingGeom(op2, state.room);
      if (!g2) continue;
      if (distToSegment(p, g2.a, g2.b) <= OPENING_GRAB) {
        snapshot();
        drag = { kind: 'openingMove', which: kinds[kj], grab: Rules.projectOnWall(p, op2.wall) - op2.offset };
        state.selected = null;
        canvas.setPointerCapture(e.pointerId);
        render();
        return;
      }
    }

    state.selected = null;
    render();
  });

  canvas.addEventListener('pointermove', function (e) {
    var p = toRoom(e.clientX, e.clientY);
    if (!drag) { updateCursor(p); return; }
    if (drag.kind === 'itemMove') {
      var it = findItem(drag.id);
      if (!it) return;
      var nx = Math.round((p.x - drag.dx) / SNAP) * SNAP;
      var ny = Math.round((p.y - drag.dy) / SNAP) * SNAP;
      var r = Rules.rectOf({ w: it.w, d: it.d, x: nx, y: ny, rot: it.rot });
      nx = Math.max(0, Math.min(state.room.w - r.w, nx));
      ny = Math.max(0, Math.min(state.room.h - r.d, ny));
      it.x = nx; it.y = ny;
      update();
    } else if (drag.kind === 'itemResize') {
      resizeItem(drag.id, drag.handle, p);
      update();
    } else if (drag.kind === 'openingMove') {
      moveOpening(drag.which, p, drag.grab);
      syncInputs(); update();
    } else if (drag.kind === 'openingResize') {
      resizeOpening(drag.which, drag.end, p);
      syncInputs(); update();
    }
  });

  canvas.addEventListener('pointerup', function () { drag = null; });
  canvas.addEventListener('pointercancel', function () { drag = null; });

  // 双击门 → 换个方向开（铰链换边）
  canvas.addEventListener('dblclick', function (e) {
    var p = toRoom(e.clientX, e.clientY);
    var g = Rules.openingGeom(state.door, state.room);
    if (g && distToSegment(p, g.a, g.b) <= 25) {
      snapshot();
      state.door.hinge = (state.door.hinge === 'start') ? 'end' : 'start';
      syncInputs(); update();
    }
  });

  function updateCursor(p) {
    var c = 'default';
    var sel = state.selected ? findItem(state.selected) : null;
    if (sel) {
      var h = hitHandle(p, itemHandles(sel));
      if (h) c = h.cursor;
      else if (inRect(p, Rules.rectOf(sel))) c = 'move';
    }
    if (c === 'default') {
      for (var i = state.items.length - 1; i >= 0; i--) {
        if (inRect(p, Rules.rectOf(state.items[i]))) { c = 'move'; break; }
      }
    }
    if (c === 'default') {
      ['door', 'window'].forEach(function (kind) {
        if (c !== 'default') return;
        var op = state[kind];
        if (!op || !(op.width > 0)) return;
        var hs = openingHandles(op);
        if (hitHandle(p, hs)) { c = 'pointer'; return; }
        if (distToSegment(p, hs[0], hs[1]) <= OPENING_GRAB) c = 'move';
      });
    }
    if (canvas.style.cursor !== c) canvas.style.cursor = c;
  }

  function findItem(id) {
    for (var i = 0; i < state.items.length; i++) if (state.items[i].id === id) return state.items[i];
    return null;
  }

  document.addEventListener('keydown', function (e) {
    var tag = (e.target && e.target.tagName) || '';
    // 正在填数字/选下拉框时，快捷键必须让给输入框，否则改尺寸会把家具删掉
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'Escape') { state.selected = null; render(); return; }
    if (!state.selected) return;
    var it = findItem(state.selected);
    if (!it) return;
    var moveKey = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].indexOf(e.key) >= 0;
    var rotKey = (e.key === 'r' || e.key === 'R');
    if ((moveKey || rotKey) && !e.repeat) snapshot(); // 长按只记一次，撤销不会碎成一地
    var moved = false;
    if (e.key === 'ArrowLeft') { it.x = Math.max(0, it.x - SNAP); moved = true; }
    if (e.key === 'ArrowRight') { it.x = Math.min(state.room.w, it.x + SNAP); moved = true; }
    if (e.key === 'ArrowUp') { it.y = Math.max(0, it.y - SNAP); moved = true; }
    if (e.key === 'ArrowDown') { it.y = Math.min(state.room.h, it.y + SNAP); moved = true; }
    if (e.key === 'r' || e.key === 'R') { it.rot = (it.rot + 90) % 360; moved = true; }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      snapshot();
      state.items = state.items.filter(function (x) { return x.id !== state.selected; });
      state.selected = null; update(); e.preventDefault(); return;
    }
    if (moved) {
      var r = Rules.rectOf(it);
      it.x = Math.max(0, Math.min(state.room.w - r.w, it.x));
      it.y = Math.max(0, Math.min(state.room.h - r.d, it.y));
      update(); e.preventDefault();
    }
  });

  // ---------- 控件 ----------
  function syncInputs() {
    $('roomW').value = state.room.w;
    $('roomH').value = state.room.h;
    $('doorWall').value = state.door.wall;
    $('doorOffset').value = state.door.offset;
    $('doorWidth').value = state.door.width;
    $('doorHinge').value = state.door.hinge;
    $('winWall').value = state.window.wall;
    $('winOffset').value = state.window.offset;
    $('winWidth').value = state.window.width;
    $('tplSelect').value = state.templateId;
  }

  function initControls() {
    Catalog.ROOM_TEMPLATES.forEach(function (t) {
      var o = document.createElement('option');
      o.value = t.id; o.textContent = t.name + ' · ' + t.w + '×' + t.h + 'cm（' + t.caveat + '）';
      $('tplSelect').appendChild(o);
    });
    Catalog.FURNITURE.forEach(function (f) {
      var o = document.createElement('option');
      o.value = f.id; o.textContent = f.name + ' ' + f.w + '×' + f.d + 'cm';
      $('addSelect').appendChild(o);
    });

    $('tplSelect').onchange = function () { snapshot(); loadTemplate(this.value, false); };
    $('roomW').onchange = function () { snapshot(); state.room.w = clampInt(this.value, 150, 1200); clampItems(); syncInputs(); update(); };
    $('roomH').onchange = function () { snapshot(); state.room.h = clampInt(this.value, 150, 1200); clampItems(); syncInputs(); update(); };

    $('btnAdd').onclick = function () {
      var id = $('addSelect').value;
      var t = Catalog.find(id);
      var x = Math.round((state.room.w - t.w) / 2 / SNAP) * SNAP;
      var y = Math.round((state.room.h - t.d) / 2 / SNAP) * SNAP;
      snapshot();
      var it = newItem(id, x, y, 0);
      state.items.push(it);
      state.selected = it.id;
      update();
    };
    $('btnRotate').onclick = function () {
      var it = findItem(state.selected); if (!it) return;
      snapshot(); it.rot = (it.rot + 90) % 360; update();
    };
    $('btnDelete').onclick = function () {
      if (!state.selected) return;
      snapshot();
      state.items = state.items.filter(function (x) { return x.id !== state.selected; });
      state.selected = null; update();
    };
    $('btnUndo').onclick = undo;
    $('btnReset').onclick = function () { snapshot(); loadTemplate(state.templateId, false); };

    $('btnDup').onclick = function () {
      var it = findItem(state.selected); if (!it) return;
      snapshot();
      var copy = newItem(it.type, it.x + 20, it.y + 20, it.rot);
      var r = Rules.rectOf(copy);
      copy.x = Math.max(0, Math.min(state.room.w - r.w, copy.x));
      copy.y = Math.max(0, Math.min(state.room.h - r.d, copy.y));
      state.items.push(copy);
      state.selected = copy.id;
      update();
    };

    function alignSelected(ax, ay) {
      var it = findItem(state.selected); if (!it) return;
      snapshot();
      var r = Rules.rectOf(it);
      if (ax === -1) it.x = 0; else if (ax === 1) it.x = state.room.w - r.w;
      if (ay === -1) it.y = 0; else if (ay === 1) it.y = state.room.h - r.d;
      update();
    }
    $('btnAlignL').onclick = function () { alignSelected(-1, 0); };
    $('btnAlignR').onclick = function () { alignSelected(1, 0); };
    $('btnAlignT').onclick = function () { alignSelected(0, -1); };
    $('btnAlignB').onclick = function () { alignSelected(0, 1); };
    $('btnCenter').onclick = function () {
      var it = findItem(state.selected); if (!it) return;
      snapshot();
      var r = Rules.rectOf(it);
      it.x = Math.round((state.room.w - r.w) / 2 / SNAP) * SNAP;
      it.y = Math.round((state.room.h - r.d) / 2 / SNAP) * SNAP;
      update();
    };

    $('btnResetSize').onclick = function () {
      var it = findItem(state.selected); if (!it) return;
      var t = Catalog.find(it.type);
      snapshot();
      it.w = t.w; it.d = t.d;
      update();
    };

    $('btnClear').onclick = function () {
      if (!state.items.length) return;
      snapshot();
      state.items = []; state.sockets = []; state.selected = null;
      update();
    };

    $('btnCopy').onclick = function () {
      var T = Catalog.THRESHOLDS;
      var lines = [
        'Destiny 原型 v0.1 · 小空间布局与动线检查（规则计算结论，非设计/施工依据）',
        '房间：' + state.room.w + ' × ' + state.room.h + ' cm；家具 ' + state.items.length + ' 件',
        '通行标准：' + T.corridorMin + ' cm（' + (T.corridorMin >= 90 ? '接近 GB 50096 套内过道最低要求' : '单人通行的行业经验值，非标准') + '）',
        '结论：' + result.summary.verdict,
        ''
      ];
      result.issues.forEach(function (iss, i) {
        lines.push((i + 1) + '. [' + (iss.level === 'error' ? '错误' : '提示') + '] ' + iss.title + ' —— ' + iss.detail);
      });
      if (result.summary.bottlenecks.length) {
        lines.push('');
        lines.push('通道实测：' + result.summary.bottlenecks.map(function (b) {
          return b.name + ' 最窄约 ' + b.width + 'cm';
        }).join('；'));
      }
      copyText(lines.join('\n'));
      alert('结论已复制，可以直接粘贴发给别人。只包含这套方案的文字，不含任何个人信息。');
    };

    $('corridorStd').onchange = function () {
      var cm = Number(this.value);
      Catalog.THRESHOLDS.corridorMin = cm;
      Catalog.THRESHOLDS.corridorHalf = cm / 2;
      heatKey = '';
      update();
    };

    $('chkZones').onchange = render;
    $('chkGrid').onchange = render;
    $('chkHeat').onchange = render;

    $('btnSocket').onclick = function () {
      state.socketMode = !state.socketMode;
      this.classList.toggle('active', state.socketMode);
      $('hint').textContent = state.socketMode
        ? '插座模式：在地面点击可添加插座，点击已有插座可删除，再按一次按钮退出。'
        : '拖动家具可以移动，选中后按 R 旋转、Delete 删除，方向键微调 5cm。';
    };

    $('btnExport').onclick = function () {
      var prev = state.selected; state.selected = null; state.hover = [];
      render();
      var url = canvas.toDataURL('image/png');
      var a = document.createElement('a');
      a.href = url; a.download = 'destiny-layout-' + Date.now() + '.png';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      state.selected = prev; render();
    };

    $('btnSave').onclick = function () {
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify({
          v: 1, templateId: state.templateId, room: state.room, items: state.items,
          door: state.door, window: state.window, sockets: state.sockets, savedAt: new Date().toISOString()
        }));
        alert('已保存到本机浏览器（不会上传到任何服务器）。');
      } catch (err) { alert('保存失败：' + err.message); }
    };

    $('btnLoad').onclick = function () {
      try {
        var raw = localStorage.getItem(STORE_KEY);
        if (!raw) { alert('本机没有找到保存记录。'); return; }
        var o = JSON.parse(raw);
        snapshot();
        state.templateId = o.templateId || 'custom';
        state.room = { w: Number(o.room && o.room.w) || 300, h: Number(o.room && o.room.h) || 200 };
        state.items = (o.items || []).filter(function (x) { return !!Catalog.find(x.type); });
        state.door = o.door || state.door;
        state.window = o.window || state.window;
        state.sockets = o.sockets || [];
        state.selected = null;
        syncInputs(); update();
      } catch (err) { alert('读取失败：' + err.message); }
    };

    var dz = ['doorWall', 'doorOffset', 'doorWidth', 'doorHinge', 'winWall', 'winOffset', 'winWidth'];
    dz.forEach(function (id) {
      $(id).onchange = function () {
        snapshot();
        state.door = { wall: $('doorWall').value, offset: Number($('doorOffset').value), width: Number($('doorWidth').value), hinge: $('doorHinge').value };
        state.window = { wall: $('winWall').value, offset: Number($('winOffset').value), width: Number($('winWidth').value) };
        update();
      };
    });

    window.addEventListener('resize', render);
  }

  function clampInt(v, lo, hi) {
    v = Number(v); if (!isFinite(v)) return lo;
    return Math.max(lo, Math.min(hi, Math.round(v)));
  }

  function clampItems() {
    state.items.forEach(function (it) {
      var r = Rules.rectOf(it);
      it.x = Math.max(0, Math.min(state.room.w - r.w, it.x));
      it.y = Math.max(0, Math.min(state.room.h - r.d, it.y));
    });
  }

  initControls();
  loadTemplate('dorm-4', false);
})();
