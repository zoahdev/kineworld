/* 规则引擎自检（node tests/rules-smoke-test.js）
 * 目的：用可复现的场景验证规则引擎确实在工作，而不是"看起来在动"。
 * MIT 许可。
 */
'use strict';

const Catalog = require('../prototype/catalog.js');
const Rules = require('../prototype/rules.js');

let pass = 0, fail = 0;
function check(name, cond, extra) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (extra ? '  -> ' + extra : '')); }
}
function has(res, code) { return res.issues.some(i => i.code === code); }
function item(type, x, y, rot) {
  const t = Catalog.find(type);
  return { id: type + '-' + x + '-' + y + '-' + (rot || 0), type, w: t.w, d: t.d, h: t.h, x, y, rot: rot || 0 };
}
function scene(room, items, door, win, sockets) {
  return { room, items, door: door || null, window: win || null, sockets: sockets || [] };
}
function barrier(count, x, y0) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(item('box-60', x, y0 + i * 40, 0));
  return out;
}

console.log('Destiny rules smoke test');

// 1. 空房间不应报错
{
  const res = Rules.analyze(scene({ w: 400, h: 300 }, [],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('空房间无 error', res.summary.errors === 0, JSON.stringify(res.issues.map(i => i.code)));
}

// 2. 重叠检测
{
  const res = Rules.analyze(scene({ w: 400, h: 300 },
    [item('bed-90', 50, 50, 0), item('bed-90', 60, 60, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('重叠家具被检出', has(res, 'overlap'));
}

// 3. 越界检测
{
  const res = Rules.analyze(scene({ w: 300, h: 300 },
    [item('bed-90', 250, 50, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('越界家具被检出', has(res, 'outside'));
}

// 4. 门扇开启区域被占
{
  // 门在下墙 x=40..120，铰链在 x=40，向内(-y)扫 90 度，半径 80
  const res = Rules.analyze(scene({ w: 300, h: 300 },
    [item('box-60', 45, 240, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('门扇开启被挡被检出', has(res, 'doorSwing'), JSON.stringify(res.issues.map(i => i.code)));
}

// 5. 完全堵死 -> 走不通（房间 400x200，x=150..210 处用收纳箱堆满）
{
  const res = Rules.analyze(scene({ w: 400, h: 200 },
    barrier(5, 150, 0).concat([item('desk-120', 250, 60, 0)]),
    { wall: 'bottom', offset: 20, width: 80, hinge: 'start' }));
  check('完全堵死时无通路', has(res, 'noPath'), JSON.stringify(res.issues.map(i => i.code)));
}

// 6. 只留 40cm 缝 -> 仍不可通行（60cm 建议值）
{
  const res = Rules.analyze(scene({ w: 400, h: 200 },
    barrier(4, 150, 0).concat([item('desk-120', 250, 60, 0)]),
    { wall: 'bottom', offset: 20, width: 80, hinge: 'start' }));
  check('40cm 缝隙判定为不可通行', has(res, 'noPath'), JSON.stringify(res.issues.map(i => i.code)));
}

// 7. 留 80cm 通道 -> 可通行，且瓶颈宽度接近 80cm
{
  const res = Rules.analyze(scene({ w: 400, h: 200 },
    barrier(3, 150, 0).concat([item('desk-120', 250, 60, 0)]),
    { wall: 'bottom', offset: 20, width: 80, hinge: 'start' }));
  check('80cm 通道可通行', !has(res, 'noPath'), JSON.stringify(res.issues.map(i => i.code)));
  const b = res.summary.bottlenecks.find(x => x.name === '书桌');
  check('瓶颈宽度约 80cm（实测 ' + (b ? b.width : 'n/a') + 'cm）', !!b && b.width >= 65 && b.width <= 95);
}

// 8. 柜前被挡
{
  const res = Rules.analyze(scene({ w: 300, h: 300 },
    [item('wardrobe-90', 20, 20, 0), item('box-60', 20, 85, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('柜前空间被占被检出', has(res, 'frontBlocked'), JSON.stringify(res.issues.map(i => i.code)));
}

// 9. 窗户被高家具遮挡
{
  const res = Rules.analyze(scene({ w: 300, h: 300 },
    [item('wardrobe-90', 100, 0, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' },
    { wall: 'top', offset: 100, width: 100 }));
  check('高家具挡窗被检出', has(res, 'windowBlock'), JSON.stringify(res.issues.map(i => i.code)));
  const res2 = Rules.analyze(scene({ w: 300, h: 300 },
    [item('shelf-low', 100, 0, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' },
    { wall: 'top', offset: 100, width: 100 }));
  check('矮柜挡窗不算问题', !has(res2, 'windowBlock'));
}

// 10. 插座被埋
{
  const res = Rules.analyze(scene({ w: 300, h: 300 },
    [item('bed-90', 20, 20, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }, null,
    [{ x: 60, y: 60 }]));
  check('插座被家具压住被检出', has(res, 'socket'), JSON.stringify(res.issues.map(i => i.code)));
}

// 11. 门口内侧被堵（门开了也进不来）
{
  const res = Rules.analyze(scene({ w: 300, h: 300 },
    [item('box-60', 50, 220, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('门口内侧被堵被检出', has(res, 'doorBlocked'), JSON.stringify(res.issues.map(i => i.code)));
  const res2 = Rules.analyze(scene({ w: 300, h: 300 },
    [item('box-60', 50, 20, 0)],
    { wall: 'bottom', offset: 40, width: 80, hinge: 'start' }));
  check('门口空着时不误报', !has(res2, 'doorBlocked'), JSON.stringify(res2.issues.map(i => i.code)));
}

// 12. 家具缩放（鼠标拖手柄）——不靠点界面，直接测几何
{
  const room = { w: 400, h: 300 };
  const r1 = Rules.resizeRect({ x: 0, y: 0, w: 120, d: 60 }, 'e', { x: 200, y: 30 }, room, 20);
  check('拖右边手柄只改宽度', !!r1 && r1.w === 200 && r1.d === 60, JSON.stringify(r1));
  const r2 = Rules.resizeRect({ x: 0, y: 0, w: 120, d: 60 }, 'e', { x: 5, y: 30 }, room, 20);
  check('缩到最小尺寸被夹住', !!r2 && r2.w === 20, JSON.stringify(r2));
  const r3 = Rules.resizeRect({ x: 0, y: 0, w: 120, d: 60 }, 'e', { x: 9999, y: 30 }, room, 20);
  check('超出房间被夹在墙内', !!r3 && r3.w === 400, JSON.stringify(r3));
  const r4 = Rules.resizeRect({ x: 100, y: 100, w: 120, d: 60 }, 'se', { x: 300, y: 250 }, room, 20);
  check('拖右下角同时改两边', !!r4 && r4.w === 200 && r4.d === 150 && r4.x === 100 && r4.y === 100, JSON.stringify(r4));
  const r5 = Rules.resizeRect({ x: 100, y: 100, w: 120, d: 60 }, 'nw', { x: 40, y: 60 }, room, 20);
  check('拖左上角会同时移动位置', !!r5 && r5.x === 40 && r5.y === 60 && r5.w === 180 && r5.d === 100, JSON.stringify(r5));
  const r6 = Rules.resizeRect({ x: 0, y: 0, w: 120, d: 60 }, 'n', { x: 50, y: -5 }, room, 20);
  check('贴墙时拖不出房间', !!r6 && r6.y === 0, JSON.stringify(r6));
  const r7 = Rules.resizeRect({ x: 300, y: 200, w: 60, d: 40 }, 'e', { x: 290, y: 220 }, room, 20);
  check('拖过头不会翻转或变负', !r7 || r7.w >= 20, JSON.stringify(r7));
}

// 13. 门 / 窗 伸缩
{
  const room = { w: 400, h: 300 };
  const door = { wall: 'bottom', offset: 40, width: 80, hinge: 'start' };
  const a = Rules.resizeOpening(door, 'end', 200, room, { min: 60, max: 120 });
  check('拖门另一端改宽度（受最大门宽限制）', a.width === 120 && a.offset === 40, JSON.stringify(a));
  const b = Rules.resizeOpening(door, 'start', 100, room, { min: 60, max: 120 });
  check('拖门起点端：终点不动、起点移动', b.offset === 60 && b.width === 60, JSON.stringify(b));
  const c = Rules.resizeOpening(door, 'start', 0, room, { min: 60, max: 120 });
  check('门宽受起点到墙头的距离限制', c.offset === 0 && c.width === 120, JSON.stringify(c));
  const d = Rules.resizeOpening(door, 'end', 50, room, { min: 60, max: 120 });
  check('门宽不小于最小值', d.width === 60, JSON.stringify(d));
  const win = { wall: 'top', offset: 100, width: 120, hinge: 'start' };
  const e = Rules.resizeOpening(win, 'end', 350, room, { min: 30 });
  check('窗可以拉得比门更宽', e.width === 250 && e.offset === 100, JSON.stringify(e));
}

// 14. 门 / 窗 沿墙移动与换墙
{
  const room = { w: 400, h: 300 };
  check('沿墙移动被夹在墙内', Rules.clampOpeningOffset({ wall: 'bottom', width: 80 }, 999, room) === 320);
  check('沿墙移动不小于 0', Rules.clampOpeningOffset({ wall: 'bottom', width: 80 }, -50, room) === 0);
  check('在房间中间不会乱换墙', Rules.nearestWall({ x: 200, y: 150 }, room, 'bottom') === 'bottom');
  check('拖到左墙附近会换到左墙', Rules.nearestWall({ x: 4, y: 150 }, room, 'bottom') === 'left');
  check('贴着下墙拖不会换墙', Rules.nearestWall({ x: 200, y: 296 }, room, 'bottom') === 'bottom');
  check('沿墙坐标投影正确', Rules.projectOnWall({ x: 123, y: 45 }, 'bottom') === 123 &&
    Rules.projectOnWall({ x: 123, y: 45 }, 'right') === 45);
}

// 15. 模板可直接分析（保证内置模板不会崩）
{
  Catalog.ROOM_TEMPLATES.forEach(tpl => {
    const items = (tpl.presetItems || []).map((p, i) => {
      const t = Catalog.find(p.type);
      return { id: 'p' + i, type: p.type, w: t.w, d: t.d, h: t.h, x: p.x, y: p.y, rot: p.rot || 0 };
    });
    const res = Rules.analyze(scene({ w: tpl.w, h: tpl.h }, items, tpl.door, tpl.window));
    check('模板 ' + tpl.id + ' 可分析', Array.isArray(res.issues),
      (res.issues || []).map(i => i.code).join(','));
  });
}

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail === 0 ? 0 : 1);
