/* Destiny 原型本地服务（零依赖，仅用于本机预览）
 * MIT 许可。只监听 127.0.0.1，不对外提供服务，不上传任何数据。
 * 用法: node server.js  [port]
 */
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const PORT = Number(process.argv[2]) || 8123;
const ROOT = __dirname;
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);
  // 浏览器总会自动请求 favicon。以前这会返回 404，在开发者工具控制台留下一条红色报错，
  // 让人以为页面坏了。这里明确回「没有图标」，不留假报错（也不引入图片资源）。
  if (urlPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  let rel = urlPath === '/' ? '/prototype/index.html' : urlPath;
  const target = path.join(ROOT, path.normalize(rel));
  if (!target.startsWith(ROOT)) {
    res.writeHead(403); res.end('403'); return;
  }
  fs.readFile(target, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found: ' + rel);
      return;
    }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(target).toLowerCase()] || 'application/octet-stream' });
    res.end(data);
  });
});

server.listen(PORT, '127.0.0.1', () => {
  const url = 'http://127.0.0.1:' + PORT + '/prototype/index.html';
  console.log('Destiny prototype running at ' + url);
  console.log('Press Ctrl+C in this window to stop.');
  if (process.platform === 'win32' && !process.env.DESTINY_NO_OPEN) exec('start "" "' + url + '"');
});
