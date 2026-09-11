/* Destiny 原型 v0.1 —— 规则引擎（纯本地计算，不调用任何外部服务或模型）
 * 许可：本文件为 KineWorld Destiny 项目自写代码，MIT 许可。
 * 输入：场景 { room:{w,h}, items:[{id,type,x,y,rot}], door:{...}, window:{...}, sockets:[{x,y}] }
 * 输出：{ issues:[...], field:{...} }
 *
 * 方法说明（可直接核验，无黑箱）：
 *  1. 几何：家具视为旋转后的矩形，做「越界」「互相重叠」判定。
 *  2. 门：以铰链为圆心的 1/4 圆区域（半径=门宽）内不得有家具，否则门无法完全打开。
 *  3. 使用空间：柜/桌/冰箱正面的预留矩形内不得有其他家具。
 *  4. 通行：把房间栅格化（5cm），用倒角距离变换求每个空格到最近障碍的距离（≈半个通道宽），
 *     再在「距离≥30cm」的格子上做 BFS，判断从门口能否走到各家具的使用点。
 *  5. 遮挡：窗前 60cm 内若有高度≥120cm 的家具，判定为遮挡。
 */
(function (root) {
  'use strict';

  var Catalog = (typeof require === 'function' && typeof module !== 'undefined')
    ? require('./catalog.js') : root.Catalog;
  var T = (Catalog && Catalog.THRESHOLDS) || {
    corridorHalf: 30, bedSide: 50, windowFront: 60, windowBlockHeight: 120, gridCell: 5
  };

  // ---------- 基础几何 ----------
  function normRot(rot) { var r = (Number(rot) || 0) % 360; return r < 0 ? r + 360 : r; }

  function rectOf(item) {
    var rot = normRot(item.rot);
    var swap = (rot === 90 || rot === 270);
    return { x: item.x, y: item.y, w: swap ? item.d : item.w, d: swap ? item.w : item.d };
  }

  // 正面朝向（屏幕坐标 y 向下）：0->下, 90->左, 180->上, 270->右
  function frontDir(rot) {
    switch (normRot(rot)) {
      case 90: return { x: -1, y: 0 };
      case 180: return { x: 0, y: -1 };
      case 270: return { x: 1, y: 0 };
      default: return { x: 0, y: 1 };
    }
  }

  function rectOverlap(a, b) {
    return !(a.x + a.w <= b.x + 1e-6 || b.x + b.w <= a.x + 1e-6 ||
             a.y + a.d <= b.y + 1e-6 || b.y + b.d <= a.y + 1e-6);
  }

  function insideRoom(r, room) {
    return r.x >= -0.01 && r.y >= -0.01 &&
           r.x + r.w <= room.w + 0.01 && r.y + r.d <= room.h + 0.01;
  }

  function activityRect(item, type) {
    var depth = (type.activity && type.activity.front) || 0;
    if (!depth) return null;
    var r = rectOf(item), dir = frontDir(item.rot);
    if (dir.x === 1) return { x: r.x + r.w, y: r.y, w: depth, d: r.d };
    if (dir.x === -1) return { x: r.x - depth, y: r.y, w: depth, d: r.d };
    if (dir.y === 1) return { x: r.x, y: r.y + r.d, w: r.w, d: depth };
    return { x: r.x, y: r.y - depth, w: r.w, d: depth };
  }

  // 床等需要「长边一侧留空」的家具，返回两侧的检查带
  function sideBands(item, type) {
    var r = rectOf(item);
    var band = type.needsSide || 0;
    if (!band) return [];
    if (r.w >= r.d) {
      return [{ x: r.x, y: r.y - band, w: r.w, d: band },
              { x: r.x, y: r.y + r.d, w: r.w, d: band }];
    }
    return [{ x: r.x - band, y: r.y, w: band, d: r.d },
            { x: r.x + r.w, y: r.y, w: band, d: r.d }];
  }

  // ---------- 门 / 窗 ----------
  function openingGeom(opening, room) {
    if (!opening) return null;
    var wall = opening.wall, off = Number(opening.offset) || 0, width = Number(opening.width) || 80;
    if (wall === 'top')    return { a: { x: off, y: 0 }, b: { x: off + width, y: 0 }, inward: { x: 0, y: 1 }, axis: { x: 1, y: 0 }, len: room.w };
    if (wall === 'bottom') return { a: { x: off, y: room.h }, b: { x: off + width, y: room.h }, inward: { x: 0, y: -1 }, axis: { x: 1, y: 0 }, len: room.w };
    if (wall === 'left')   return { a: { x: 0, y: off }, b: { x: 0, y: off + width }, inward: { x: 1, y: 0 }, axis: { x: 0, y: 1 }, len: room.h };
    return { a: { x: room.w, y: off }, b: { x: room.w, y: off + width }, inward: { x: -1, y: 0 }, axis: { x: 0, y: 1 }, len: room.h };
  }

  // 门扇扫过的 1/4 圆（铰链为圆心，半径=门宽）
  function doorSwingPoints(door, room, step) {
    var g = openingGeom(door, room);
    if (!g) return [];
    step = step || 5;
    var width = Number(door.width) || 80;
    var hinge = (door.hinge === 'end') ? g.b : g.a;
    var other = (door.hinge === 'end') ? g.a : g.b;
    var ux = Math.sign(other.x - hinge.x), uy = Math.sign(other.y - hinge.y);
    var pts = [];
    for (var u = 0; u <= width; u += step) {
      for (var v = 0; v <= width; v += step) {
        if (u * u + v * v > width * width) continue;
        var p = { x: hinge.x + ux * u + g.inward.x * v, y: hinge.y + uy * u + g.inward.y * v };
        if (p.x < 0 || p.y < 0 || p.x > room.w || p.y > room.h) continue;
        pts.push(p);
      }
    }
    return pts;
  }

  function doorEntryPoint(door, room) {
    var g = openingGeom(door, room);
    if (!g) return { x: room.w / 2, y: room.h / 2 };
    var mx = (g.a.x + g.b.x) / 2, my = (g.a.y + g.b.y) / 2;
    return { x: mx + g.inward.x * 20, y: my + g.inward.y * 20 };
  }

  function windowInnerRect(win, room) {
    var g = openingGeom(win, room);
    if (!g) return null;
    var depth = T.windowFront;
    var x, y, w, d;
    if (g.axis.x === 1) { // 上下墙
      w = Math.abs(g.b.x - g.a.x); d = depth;
      x = Math.min(g.a.x, g.b.x);
      y = (g.inward.y === 1) ? 0 : room.h - depth;
    } else {
      d = Math.abs(g.b.y - g.a.y); w = depth;
      y = Math.min(g.a.y, g.b.y);
      x = (g.inward.x === 1) ? 0 : room.w - depth;
    }
    return { x: x, y: y, w: w, d: d };
  }

  // ---------- 栅格：障碍 / 距离场 / 连通性 ----------
  // 栅格外扩一圈「墙体」：否则贴墙的窄缝会被误判成宽通道
  function buildField(room, items, cell) {
    cell = cell || T.gridCell;
    var w0 = Math.max(1, Math.ceil(room.w / cell));
    var h0 = Math.max(1, Math.ceil(room.h / cell));
    var gw = w0 + 2, gh = h0 + 2, PAD = 1;
    var blocked = new Uint8Array(gw * gh);
    var x, y, i;
    for (x = 0; x < gw; x++) { blocked[x] = 1; blocked[(gh - 1) * gw + x] = 1; }
    for (y = 0; y < gh; y++) { blocked[y * gw] = 1; blocked[y * gw + gw - 1] = 1; }
    for (y = 0; y < h0; y++) {
      for (x = 0; x < w0; x++) {
        if ((x + 0.5) * cell > room.w || (y + 0.5) * cell > room.h) blocked[(y + PAD) * gw + (x + PAD)] = 1;
      }
    }
    for (i = 0; i < items.length; i++) {
      var r = rectOf(items[i]);
      var x0 = Math.max(0, Math.floor(r.x / cell));
      var x1 = Math.min(w0 - 1, Math.floor((r.x + r.w - 1e-6) / cell));
      var y0 = Math.max(0, Math.floor(r.y / cell));
      var y1 = Math.min(h0 - 1, Math.floor((r.y + r.d - 1e-6) / cell));
      for (var gy = y0; gy <= y1; gy++) {
        for (var gx = x0; gx <= x1; gx++) {
          var cx = (gx + 0.5) * cell, cy = (gy + 0.5) * cell;
          if (cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.d) blocked[(gy + PAD) * gw + (gx + PAD)] = 1;
        }
      }
    }
    return {
      gw: gw, gh: gh, w0: w0, h0: h0, pad: PAD, cell: cell,
      blocked: blocked, dist: chamfer(blocked, gw, gh, cell)
    };
  }

  // 两遍倒角距离变换：得到每个空格中心到最近障碍格中心的距离（cm）
  function chamfer(blocked, gw, gh, cell) {
    var INF = 1e9, d1 = cell, d2 = cell * Math.SQRT2;
    var dist = new Float32Array(gw * gh);
    for (var i = 0; i < dist.length; i++) dist[i] = blocked[i] ? 0 : INF;
    var x, y, idx, v;
    for (y = 0; y < gh; y++) {
      for (x = 0; x < gw; x++) {
        idx = y * gw + x;
        if (dist[idx] === 0) continue;
        v = dist[idx];
        if (y > 0) {
          if (x > 0)     v = Math.min(v, dist[idx - gw - 1] + d2);
                          v = Math.min(v, dist[idx - gw] + d1);
          if (x < gw - 1) v = Math.min(v, dist[idx - gw + 1] + d2);
        }
        if (x > 0)        v = Math.min(v, dist[idx - 1] + d1);
        dist[idx] = v;
      }
    }
    for (y = gh - 1; y >= 0; y--) {
      for (x = gw - 1; x >= 0; x--) {
        idx = y * gw + x;
        if (dist[idx] === 0) continue;
        v = dist[idx];
        if (y < gh - 1) {
          if (x < gw - 1) v = Math.min(v, dist[idx + gw + 1] + d2);
                          v = Math.min(v, dist[idx + gw] + d1);
          if (x > 0)      v = Math.min(v, dist[idx + gw - 1] + d2);
        }
        if (x < gw - 1)   v = Math.min(v, dist[idx + 1] + d1);
        dist[idx] = v;
      }
    }
    return dist;
  }

  function cellAt(field, p) {
    var gx = Math.floor(p.x / field.cell) + field.pad, gy = Math.floor(p.y / field.cell) + field.pad;
    gx = Math.max(field.pad, Math.min(field.gw - field.pad - 1, gx));
    gy = Math.max(field.pad, Math.min(field.gh - field.pad - 1, gy));
    return { gx: gx, gy: gy, idx: gy * field.gw + gx };
  }

  // 在 p 附近 radiusCm 范围内，找一个「净宽最大」的可站立格子。
  // 不能用"最近的格子"：起点/终点若取到紧贴家具或墙的位置，会把整条通道的宽度上限压死。
  function freestCell(field, p, halfWidth, radiusCm) {
    var c = cellAt(field, p);
    var rMax = Math.max(1, Math.ceil((radiusCm || 40) / field.cell));
    var best = -1, bestV = -1, bestD = Infinity;
    for (var dy = -rMax; dy <= rMax; dy++) {
      for (var dx = -rMax; dx <= rMax; dx++) {
        var gx = c.gx + dx, gy = c.gy + dy;
        if (gx < 0 || gy < 0 || gx >= field.gw || gy >= field.gh) continue;
        var idx = gy * field.gw + gx;
        if (field.blocked[idx] || field.dist[idx] < halfWidth) continue;
        var d2 = dx * dx + dy * dy;
        if (field.dist[idx] > bestV + 1e-6 ||
            (Math.abs(field.dist[idx] - bestV) <= 1e-6 && d2 < bestD)) {
          bestV = field.dist[idx]; bestD = d2; best = idx;
        }
      }
    }
    return best;
  }

  // 最宽路（最大瓶颈路）：在所有「半宽 ≥ halfWidth」的格子中，
  // 找一条让「路径上最小净宽」尽可能大的路，并返回该路径。
  // 用最短路会贴着墙和家具走，从而低估通道宽度，所以这里不用最短路。
  function findPath(field, startIdx, goalIdx, halfWidth) {
    if (startIdx < 0 || goalIdx < 0) return null;
    if (field.blocked[startIdx] || field.dist[startIdx] < halfWidth) return null;
    if (field.blocked[goalIdx] || field.dist[goalIdx] < halfWidth) return null;
    var n = field.gw * field.gh;
    var best = new Float32Array(n); best.fill(-1);
    var prev = new Int32Array(n); prev.fill(-1);
    var hIdx = [], hKey = [];

    function swap(a, b) {
      var ti = hIdx[a]; hIdx[a] = hIdx[b]; hIdx[b] = ti;
      var tk = hKey[a]; hKey[a] = hKey[b]; hKey[b] = tk;
    }
    function push(i, k) {
      hIdx.push(i); hKey.push(k);
      var c = hIdx.length - 1;
      while (c > 0) { var p = (c - 1) >> 1; if (hKey[p] >= hKey[c]) break; swap(c, p); c = p; }
    }
    function pop() {
      var ti = hIdx[0], tk = hKey[0];
      var li = hIdx.pop(), lk = hKey.pop();
      if (hIdx.length) {
        hIdx[0] = li; hKey[0] = lk;
        var c = 0;
        for (;;) {
          var l = 2 * c + 1, r = l + 1, m = c;
          if (l < hKey.length && hKey[l] > hKey[m]) m = l;
          if (r < hKey.length && hKey[r] > hKey[m]) m = r;
          if (m === c) break;
          swap(m, c); c = m;
        }
      }
      return { i: ti, k: tk };
    }

    var gw = field.gw, gh = field.gh;
    best[startIdx] = field.dist[startIdx];
    push(startIdx, best[startIdx]);
    while (hIdx.length) {
      var cur = pop();
      if (cur.k < best[cur.i] - 1e-6) continue;
      if (cur.i === goalIdx) break;
      var cx = cur.i % gw, cy = (cur.i - cx) / gw;
      for (var k = 0; k < 4; k++) {
        var nx = cx + (k === 0 ? 1 : k === 1 ? -1 : 0);
        var ny = cy + (k === 2 ? 1 : k === 3 ? -1 : 0);
        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
        var ni = ny * gw + nx;
        if (field.blocked[ni] || field.dist[ni] < halfWidth) continue;
        var cand = Math.min(cur.k, field.dist[ni]);
        if (cand > best[ni] + 1e-6) { best[ni] = cand; prev[ni] = cur.i; push(ni, cand); }
      }
    }
    if (best[goalIdx] < 0) return null;
    var path = [], node = goalIdx, guard = 0;
    while (node !== -1 && guard++ < n) { path.push(node); if (node === startIdx) break; node = prev[node]; }
    if (path[path.length - 1] !== startIdx) return null;
    return path.reverse();
  }

  // ---------- 主分析 ----------
  function analyze(scene) {
    var room = scene.room, items = scene.items || [];
    var issues = [];
    var seq = 0;
    function add(level, code, title, detail, targets) {
      issues.push({ id: 'i' + (++seq), level: level, code: code, title: title, detail: detail, targets: targets || [] });
    }

    var types = {};
    for (var i = 0; i < items.length; i++) types[items[i].id] = Catalog.find(items[i].type) || { name: items[i].type, w: items[i].w, d: items[i].d, h: 0 };
    function nameOf(it) { return (types[it.id] && types[it.id].name) || it.type; }

    // 1. 越界
    for (i = 0; i < items.length; i++) {
      var r = rectOf(items[i]);
      if (!insideRoom(r, room)) add('error', 'outside', nameOf(items[i]) + ' 超出房间范围', '请把它完整移入房间（当前位置有一部分在墙外）。', [items[i].id]);
    }

    // 2. 互相重叠
    for (i = 0; i < items.length; i++) {
      for (var j = i + 1; j < items.length; j++) {
        var a = rectOf(items[i]), b = rectOf(items[j]);
        if (rectOverlap(a, b)) {
          add('error', 'overlap', nameOf(items[i]) + ' 与 ' + nameOf(items[j]) + ' 重叠', '两件家具占了同一块地方，现实中放不下。挪开其中一件即可。', [items[i].id, items[j].id]);
        }
      }
    }

    // 3. 门扇开启区域
    if (scene.door) {
      var pts = doorSwingPoints(scene.door, room, 5);
      var hit = {};
      for (i = 0; i < items.length; i++) {
        var rr = rectOf(items[i]);
        for (var p = 0; p < pts.length; p++) {
          if (pts[p].x >= rr.x && pts[p].x <= rr.x + rr.w && pts[p].y >= rr.y && pts[p].y <= rr.y + rr.d) { hit[items[i].id] = true; break; }
        }
      }
      var hitIds = Object.keys(hit);
      if (hitIds.length) {
        var hitNames = hitIds.map(function (id) { return (types[id] && types[id].name) || id; });
        add('error', 'doorSwing', '房门无法完全打开', '门扇要扫过的扇形区域被：' + hitNames.join('、') + ' 占用。把这些家具移出扇形区域，否则门只能开一半。', hitIds);
      }
    }

    // 3b. 门口正前方被占（门开了也进不来）
    if (scene.door) {
      var dg2 = openingGeom(scene.door, room);
      if (dg2) {
        var depth = 60, fr;
        if (dg2.axis.x === 1) {
          fr = { x: Math.min(dg2.a.x, dg2.b.x), w: Number(scene.door.width) || 80,
                 y: (dg2.inward.y === 1) ? 0 : room.h - depth, d: depth };
        } else {
          fr = { y: Math.min(dg2.a.y, dg2.b.y), d: Number(scene.door.width) || 80,
                 x: (dg2.inward.x === 1) ? 0 : room.w - depth, w: depth };
        }
        var blockers2 = [];
        for (var n = 0; n < items.length; n++) {
          if (rectOverlap(fr, rectOf(items[n]))) blockers2.push(nameOf(items[n]));
        }
        if (blockers2.length) {
          add('error', 'doorBlocked', '门口被堵住', '门内侧 ' + depth + 'cm 范围内有：' + blockers2.join('、') + '，开门也进不来。先留出门口这块地方。', []);
        }
      }
    }

    // 4. 使用/开门空间与床侧空间
    for (i = 0; i < items.length; i++) {
      var it = items[i], ty = types[it.id];
      var ar = activityRect(it, ty);
      if (ar) {
        var blockers = [];
        for (var k = 0; k < items.length; k++) {
          if (items[k].id === it.id) continue;
          if (rectOverlap(ar, rectOf(items[k]))) blockers.push(nameOf(items[k]));
        }
        if (ar.x < -0.01 || ar.y < -0.01 || ar.x + ar.w > room.w + 0.01 || ar.y + ar.d > room.h + 0.01) {
          add('warn', 'frontWall', nameOf(it) + ' 正面空间不足', ty.name + ' 正面需要约 ' + (ty.activity.front) + 'cm（含开门/起身/取物），现在这一侧已经贴墙或出界。旋转或挪开它。', [it.id]);
        } else if (blockers.length) {
          add('error', 'frontBlocked', nameOf(it) + ' 前面被挡住', '需要约 ' + ty.activity.front + 'cm 才能开门/正常使用，当前被：' + blockers.join('、') + ' 占用。', [it.id]);
        }
      }
      var bands = sideBands(it, ty);
      if (bands.length) {
        var okSide = 0, bandDetail = [];
        for (var bi = 0; bi < bands.length; bi++) {
          var band = bands[bi];
          var inRoom = band.x >= -0.01 && band.y >= -0.01 && band.x + band.w <= room.w + 0.01 && band.y + band.d <= room.h + 0.01;
          var free = inRoom;
          for (var k2 = 0; k2 < items.length && free; k2++) {
            if (items[k2].id === it.id) continue;
            if (rectOverlap(band, rectOf(items[k2]))) free = false;
          }
          if (free) okSide++; else bandDetail.push(bi === 0 ? '一侧' : '另一侧');
        }
        if (okSide === 0) {
          add('warn', 'bedSide', nameOf(it) + ' 四周上下床空间不足', '床的某一边需要约 ' + ty.needsSide + 'cm 才能上下床/整理床铺，现在两边都不够。', [it.id]);
        }
      }
    }

    // 5. 窗户遮挡
    var wr = windowInnerRect(scene.window, room);
    if (wr) {
      var tall = [];
      for (i = 0; i < items.length; i++) {
        var t2 = types[items[i].id];
        if ((t2.h || 0) >= T.windowBlockHeight && rectOverlap(wr, rectOf(items[i]))) tall.push(nameOf(items[i]));
      }
      if (tall.length) add('warn', 'windowBlock', '窗户被高家具挡住', '窗前约 ' + T.windowFront + 'cm 内有：' + tall.join('、') + '，会影响采光和开窗。', []);
    }

    // 6. 通行（距离场 + BFS）
    var field = buildField(room, items, T.gridCell);
    var half = T.corridorHalf;
    var bottlenecks = [];
    if (scene.door) {
      var entry = doorEntryPoint(scene.door, room);
      var startIdx = freestCell(field, entry, half, 60);
      var targets = [];
      for (i = 0; i < items.length; i++) {
        var itm = items[i], tt = types[itm.id];
        if ((tt.activity && tt.activity.front) || tt.needsSide) {
          targets.push({ id: itm.id, name: nameOf(itm), points: accessPoints(itm, tt) });
        }
      }
      if (wr) targets.push({ id: '__window', name: '窗户', points: [{ x: wr.x + wr.w / 2, y: wr.y + wr.d / 2 }], level: 'warn' });

      for (i = 0; i < targets.length; i++) {
        var tg = targets[i], reached = false, bestBottleneck = 0;
        for (var pi = 0; pi < tg.points.length; pi++) {
          var goalIdx = freestCell(field, tg.points[pi], half, 40);
          if (goalIdx < 0) continue;
          var path = findPath(field, startIdx, goalIdx, half);
          if (path) {
            reached = true;
            // 路径两端必然紧挨家具/门口，会低估通道宽度，因此各去掉 40cm 再取最小值
            var skip = Math.max(1, Math.ceil(40 / field.cell));
            var lo = skip, hi = path.length - 1 - skip, mn = Infinity;
            if (hi < lo) { lo = 0; hi = path.length - 1; }
            for (var q = lo; q <= hi; q++) mn = Math.min(mn, field.dist[path[q]]);
            bestBottleneck = Math.max(bestBottleneck, Math.round(mn * 2));
            break;
          }
        }
        if (!reached) {
          add(tg.level || 'error', 'noPath', '从门口到' + tg.name + '的路走不通', '按「单人通行至少 ' + (half * 2) + 'cm 净宽」的常用建议值，找不到一条能从门口走到' + tg.name + '的路。挪开挡路的家具，或把通道留宽一点。', tg.id === '__window' ? [] : [tg.id]);
        } else {
          bottlenecks.push({ name: tg.name, width: bestBottleneck });
        }
      }
    }

    // 7. 插座被遮挡
    if (scene.sockets && scene.sockets.length) {
      var covered = [];
      for (i = 0; i < scene.sockets.length; i++) {
        var s = scene.sockets[i];
        for (var m = 0; m < items.length; m++) {
          var r3 = rectOf(items[m]);
          if (s.x >= r3.x && s.x <= r3.x + r3.w && s.y >= r3.y && s.y <= r3.y + r3.d) { covered.push(nameOf(items[m])); break; }
        }
      }
      if (covered.length) add('warn', 'socket', '有插座被家具挡住', '被挡的插座会被：' + covered.join('、') + ' 压住，插拔困难。挪开家具或加一个插线板位置。', []);
    }

    var errors = issues.filter(function (x) { return x.level === 'error'; }).length;
    var warns = issues.filter(function (x) { return x.level === 'warn'; }).length;
    return {
      issues: issues,
      summary: {
        errors: errors, warns: warns,
        items: items.length,
        bottlenecks: bottlenecks,
        verdict: errors === 0 && warns === 0 ? '未发现明显问题' : (errors > 0 ? '有 ' + errors + ' 个必须解决的问题' : '可以住，但有 ' + warns + ' 处提示')
      },
      field: field
    };
  }

  function findType(items, id) {
    for (var i = 0; i < items.length; i++) if (items[i].id === id) return items[i].type;
    return '';
  }

  function accessPoints(item, type) {
    var r = rectOf(item), dir = frontDir(item.rot), out = [];
    if (type.activity && type.activity.front) {
      out.push({ x: r.x + r.w / 2 + dir.x * 25, y: r.y + r.d / 2 + dir.y * 25 });
    } else if (type.needsSide) {
      if (r.w >= r.d) {
        out.push({ x: r.x + r.w / 2, y: r.y - 25 }, { x: r.x + r.w / 2, y: r.y + r.d + 25 });
      } else {
        out.push({ x: r.x - 25, y: r.y + r.d / 2 }, { x: r.x + r.w + 25, y: r.y + r.d / 2 });
      }
    } else {
      out.push({ x: r.x + r.w / 2 + dir.x * 25, y: r.y + r.d / 2 + dir.y * 25 });
    }
    // 兜底：家具中心四周（用于贴墙等极端情况）
    out.push({ x: r.x + r.w / 2, y: r.y - 25 }, { x: r.x + r.w / 2, y: r.y + r.d + 25 },
             { x: r.x - 25, y: r.y + r.d / 2 }, { x: r.x + r.w + 25, y: r.y + r.d / 2 });
    return out;
  }

  // ---------- 鼠标拖动用的纯几何（抽出来是为了能被单测覆盖，不用靠点浏览器） ----------
  function wallLength(wall, room) {
    return (wall === 'top' || wall === 'bottom') ? room.w : room.h;
  }

  function projectOnWall(p, wall) {
    return (wall === 'top' || wall === 'bottom') ? p.x : p.y;
  }

  // 拖动门/窗时吸附到最近的那面墙；离得明显更近且足够近才换墙，避免来回抖动
  function nearestWall(p, room, current, maxDist, hysteresis) {
    var d = { top: p.y, bottom: room.h - p.y, left: p.x, right: room.w - p.x };
    var maxD = (maxDist == null) ? 60 : maxDist;
    var hys = (hysteresis == null) ? 3 : hysteresis;
    var best = (current && d[current] != null) ? current : 'top';
    var bestV = d[best];
    Object.keys(d).forEach(function (k) {
      if (k !== best && d[k] < bestV - hys && d[k] <= maxD) { bestV = d[k]; best = k; }
    });
    return best;
  }

  function clampOpeningOffset(op, along, room) {
    var len = wallLength(op.wall, room);
    return Math.max(0, Math.min(len - op.width, along));
  }

  // 拖动门/窗的一端改宽度：end = 'start' | 'end'，along = 沿墙坐标(cm)
  function resizeOpening(op, end, along, room, opts) {
    var len = wallLength(op.wall, room);
    var min = (opts && opts.min != null) ? opts.min : 30;
    var max = Math.min((opts && opts.max != null) ? opts.max : len, len);
    var out = { wall: op.wall, offset: op.offset, width: op.width, hinge: op.hinge };
    if (end === 'start') {
      var endPos = op.offset + op.width;
      var start = Math.max(0, Math.min(endPos - min, along));
      out.width = Math.min(max, endPos - start);
      out.offset = endPos - out.width;
    } else {
      out.width = Math.max(min, Math.min(max, along - op.offset));
      out.offset = clampOpeningOffset({ wall: op.wall, width: out.width }, op.offset, room);
    }
    return out;
  }

  // 家具缩放：输入当前屏幕轴对齐矩形、手柄名(n/s/e/w/ne/nw/se/sw)、指针位置，
  // 返回夹紧后的新矩形；放不下则返回 null。
  function resizeRect(rect, handle, p, room, minSize) {
    minSize = minSize || 20;
    var x0 = rect.x, y0 = rect.y, x1 = rect.x + rect.w, y1 = rect.y + rect.d;
    if (handle.indexOf('w') >= 0) x0 = p.x;
    if (handle.indexOf('e') >= 0) x1 = p.x;
    if (handle.indexOf('n') >= 0) y0 = p.y;
    if (handle.indexOf('s') >= 0) y1 = p.y;
    if (x1 - x0 < minSize) { if (handle.indexOf('w') >= 0) x0 = x1 - minSize; else x1 = x0 + minSize; }
    if (y1 - y0 < minSize) { if (handle.indexOf('n') >= 0) y0 = y1 - minSize; else y1 = y0 + minSize; }
    x0 = Math.max(0, x0); y0 = Math.max(0, y0);
    x1 = Math.min(room.w, x1); y1 = Math.min(room.h, y1);
    if (x1 - x0 < minSize || y1 - y0 < minSize) return null;
    return { x: x0, y: y0, w: x1 - x0, d: y1 - y0 };
  }

  var api = {
    VERSION: '0.1.0',
    analyze: analyze,
    wallLength: wallLength,
    projectOnWall: projectOnWall,
    nearestWall: nearestWall,
    clampOpeningOffset: clampOpeningOffset,
    resizeOpening: resizeOpening,
    resizeRect: resizeRect,
    rectOf: rectOf,
    frontDir: frontDir,
    activityRect: activityRect,
    sideBands: sideBands,
    doorSwingPoints: doorSwingPoints,
    windowInnerRect: windowInnerRect,
    openingGeom: openingGeom,
    buildField: buildField,
    doorEntryPoint: doorEntryPoint
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Rules = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
