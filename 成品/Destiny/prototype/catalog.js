/* Destiny 原型 v0.1 —— 家具 / 房间模板 / 阈值数据
 * 许可：本文件为 KineWorld Destiny 项目自写代码与自编数据，MIT 许可，可自由使用。
 * 重要：下列尺寸是「公开常见建议值 / 常见家具规格」，不是实测数据，
 *      也不是专业设计或施工依据。房间模板尺寸为示例估算值，请按实际测量修改。
 * 来源说明见 ../notes/sources.md
 */
(function (root) {
  'use strict';

  // 说明：w = 未旋转时沿 X 方向的尺寸(cm)，d = 沿 Y 方向的尺寸(cm)，h = 高度(cm)
  // activity.front = 该家具正面需要预留的使用/开门空间(cm)
  // needsSide      = 该家具长边某一侧需要预留的空间(cm)，例如上下床
  const FURNITURE = [
    { id: 'bed-90',     name: '单人床',   w: 90,  d: 190, h: 45,  needsSide: 50, note: '常见学生宿舍床' },
    { id: 'bed-120',    name: '加宽单人床', w: 120, d: 200, h: 45,  needsSide: 50 },
    { id: 'bed-150',    name: '双人床',    w: 150, d: 200, h: 45,  needsSide: 50 },
    { id: 'bunk-90',    name: '上下铺',    w: 200, d: 90,  h: 180, needsSide: 50, note: '按占地 200×90 估算，实际以床体尺寸为准' },
    { id: 'desk-120',   name: '书桌',      w: 120, d: 60,  h: 75,  activity: { front: 60 }, note: '含椅子与起身空间' },
    { id: 'desk-100',   name: '小书桌',    w: 100, d: 50,  h: 75,  activity: { front: 60 } },
    { id: 'chair-45',   name: '椅子',      w: 45,  d: 45,  h: 90,  activity: { front: 50 } },
    { id: 'wardrobe-90',name: '衣柜',      w: 90,  d: 60,  h: 200, activity: { front: 60 }, note: '平开门，取物需柜前空间' },
    { id: 'wardrobe-60',name: '小衣柜',    w: 60,  d: 55,  h: 200, activity: { front: 60 } },
    { id: 'bookcase-80',name: '书架',      w: 80,  d: 30,  h: 180 },
    { id: 'shelf-low',  name: '矮柜',      w: 100, d: 30,  h: 45,  activity: { front: 40 } },
    { id: 'shoecab-100',name: '鞋柜',      w: 100, d: 35,  h: 100, activity: { front: 60 } },
    { id: 'fridge-60',  name: '冰箱',      w: 60,  d: 60,  h: 175, activity: { front: 90 } },
    { id: 'washer-60',  name: '洗衣机',    w: 60,  d: 60,  h: 85,  activity: { front: 60 } },
    { id: 'table-80',   name: '小餐桌',    w: 80,  d: 80,  h: 75,  activity: { front: 60 } },
    { id: 'nightstand', name: '床头柜',    w: 40,  d: 40,  h: 50,  activity: { front: 0 } },
    { id: 'box-60',     name: '收纳箱',    w: 60,  d: 40,  h: 40 }
  ];

  // 房间模板：尺寸均为「示例估算值」，必须按实际测量修改后才可信
  const ROOM_TEMPLATES = [
    {
      id: 'dorm-4', name: '大学 4 人间宿舍（示例估算）', w: 600, h: 360,
      door:   { wall: 'bottom', offset: 60, width: 90, hinge: 'start' },
      window: { wall: 'top', offset: 180, width: 180 },
      presetItems: [
        { type: 'bed-90', x: 20,  y: 20,  rot: 90 },
        { type: 'bed-90', x: 20,  y: 220, rot: 90 },
        { type: 'desk-120', x: 300, y: 20, rot: 0 }
      ],
      caveat: '示例估算尺寸，不代表任何具体学校宿舍'
    },
    {
      id: 'rent-studio', name: '单间出租屋（示例估算）', w: 360, h: 300,
      door:   { wall: 'bottom', offset: 40, width: 80, hinge: 'start' },
      window: { wall: 'top', offset: 120, width: 120 },
      presetItems: [
        { type: 'bed-150', x: 20, y: 20, rot: 0 },
        { type: 'wardrobe-90', x: 250, y: 20, rot: 90 },
        { type: 'desk-120', x: 130, y: 220, rot: 180 }
      ],
      caveat: '示例估算尺寸'
    },
    {
      id: 'bedroom', name: '小卧室（示例估算）', w: 330, h: 300,
      door:   { wall: 'left', offset: 30, width: 80, hinge: 'end' },
      window: { wall: 'top', offset: 100, width: 150 },
      presetItems: [
        { type: 'bed-150', x: 100, y: 20, rot: 0 },
        { type: 'wardrobe-90', x: 20, y: 200, rot: 0 },
        { type: 'desk-120', x: 180, y: 200, rot: 180 }
      ],
      caveat: '示例估算尺寸'
    },
    { id: 'custom', name: '空白房间（自定义尺寸）', w: 400, h: 300,
      door: { wall: 'bottom', offset: 40, width: 80, hinge: 'start' },
      window: { wall: 'top', offset: 140, width: 120 },
      presetItems: [], caveat: '需自行填写实际测量尺寸' }
  ];

  // 阈值（cm）：均为公开常用建议值，见 notes/sources.md
  const THRESHOLDS = {
    corridorMin: 60,      // 单人通行最小净宽
    corridorComfort: 90,  // 较舒适净宽
    corridorHalf: 30,     // corridorMin / 2，用于距离场判定
    bedSide: 50,          // 床侧上下床/整理床铺
    cabinetFront: 60,     // 柜前取物与开门
    deskFront: 60,        // 桌椅与起身
    windowFront: 60,      // 窗前空间
    windowBlockHeight: 120, // 高于此值的家具被认为遮挡窗户
    doorSwingClear: 0,    // 门扇开启范围内不应有家具（0 表示不允许任何占用）
    gridCell: 5,          // 计算网格边长
    tolerance: 5          // 网格量化带来的误差上限（用于文案提示）
  };

  const api = {
    VERSION: '0.1.0',
    FURNITURE: FURNITURE,
    ROOM_TEMPLATES: ROOM_TEMPLATES,
    THRESHOLDS: THRESHOLDS,
    find: function (id) {
      for (let i = 0; i < FURNITURE.length; i++) if (FURNITURE[i].id === id) return FURNITURE[i];
      return null;
    },
    template: function (id) {
      for (let i = 0; i < ROOM_TEMPLATES.length; i++) if (ROOM_TEMPLATES[i].id === id) return ROOM_TEMPLATES[i];
      return ROOM_TEMPLATES[0];
    }
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Catalog = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
