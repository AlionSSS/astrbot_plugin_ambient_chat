/** 无状态的小工具：DOM 查询、深拷贝、转义、路径读写、时间与数量格式化。 */
export const $ = (selector) => document.querySelector(selector);

export function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function setPath(target, path, value) {
  const keys = path.split('.');
  let cursor = target;
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i];
    if (cursor[key] === undefined || cursor[key] === null) cursor[key] = {};
    cursor = cursor[key];
  }
  cursor[keys[keys.length - 1]] = value;
}

export function getPath(target, path) {
  return path.split('.').reduce((acc, key) => (acc == null ? acc : acc[key]), target);
}

export function toLocalInput(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  const pad = (value) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function fromLocalInput(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toISOString();
}

export function kindLabel(messageType) {
  if (messageType === 'GroupMessage') return '群聊';
  if (messageType === 'FriendMessage') return '私聊';
  return '会话';
}

export function formatCountdown(seconds) {
  if (seconds === null || seconds === undefined) return '-';
  if (seconds <= 0) return '即将触发';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  if (hours > 0) return `${hours} 小时 ${minutes} 分后`;
  if (minutes > 0) return `${minutes} 分 ${secs} 秒后`;
  return `${secs} 秒后`;
}

/**
 * 把后端返回的 ISO 时刻渲染成便于扫读的本地时间。
 * 今天的只留时分秒；跨天补「月-日」，跨年再补年份，避免整列被冗余日期撑宽。
 */
export function formatMoment(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  const pad = (number) => String(number).padStart(2, '0');
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  const now = new Date();
  if (date.toDateString() === now.toDateString()) return time;
  const day = `${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  if (date.getFullYear() === now.getFullYear()) return `${day} ${time}`;
  return `${date.getFullYear()}-${day} ${time}`;
}

/**
 * 运行管理列表里的时刻：不看今天与否，统一「MM-DD HH:mm:ss」。
 * 没有值也给同样宽度的占位符，避免从「—」变成真实时刻时把相邻列挤动。
 */
export function formatStamp(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  const pad = (number) => String(number).padStart(2, '0');
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}
