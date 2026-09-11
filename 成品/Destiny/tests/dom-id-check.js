/* 轻量检查：app.js 里引用的 DOM id 必须都存在于 index.html（避免运行时才发现空引用）
 * 用法: node tests/dom-id-check.js
 */
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'prototype', 'index.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'prototype', 'app.js'), 'utf8');

const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const used = new Set([...js.matchAll(/\$\('([^']+)'\)/g)].map(m => m[1]));
const missing = [...used].filter(i => !ids.has(i));

console.log('index.html id 数量: ' + ids.size);
console.log('app.js 引用 id 数量: ' + used.size);
if (missing.length) {
  console.log('缺失的 id: ' + missing.join(', '));
  process.exit(1);
}
console.log('OK: 所有引用的 id 都存在');
