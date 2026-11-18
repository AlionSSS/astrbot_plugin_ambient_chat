/** 插件信息页：插件元数据与运行信息，全部只读。 */
import { on, state } from '../store.js';
import { $ } from '../util.js';

/** 空值统一显示成「-」，避免表格里出现空白单元格。 */
const DASH = '-';

function setText(selector, value) {
  const node = $(selector);
  if (!node) return;
  const empty = value === undefined || value === null || value === '';
  node.textContent = empty ? DASH : String(value);
}

/** 列表字段（支持平台、标签）用顿号连接；空列表给「-」。 */
function joinList(items) {
  const list = (items || []).map((item) => String(item).trim()).filter(Boolean);
  return list.length ? list.join('、') : DASH;
}

function renderRepo(repo) {
  const link = $('#info-plugin-repo');
  if (!link) return;
  const url = String(repo || '').trim();
  link.textContent = url || DASH;
  if (url) link.href = url;
  else link.removeAttribute('href');
}

/** 运行环境：AstrBot 与 Python 版本，排查问题时最常被问到。 */
function environmentText(environment) {
  const parts = [];
  const astrbot = String(environment?.astrbot_version || '').trim();
  if (astrbot) parts.push(`AstrBot ${astrbot}`);
  const python = String(environment?.python_version || '').trim();
  if (python) parts.push(`Python ${python}`);
  return parts.join(' · ') || DASH;
}

export function renderAbout() {
  const meta = state.meta;
  if (!meta || !$('#section-about')) return;

  const plugin = meta.plugin || {};
  setText('#info-plugin-name', plugin.name);
  setText('#info-plugin-display', plugin.display_name);
  setText('#info-plugin-version', plugin.version);
  setText('#info-plugin-author', plugin.author);
  setText('#info-plugin-desc', plugin.desc);
  renderRepo(plugin.repo);
  setText('#info-plugin-platforms', joinList(plugin.support_platforms));
  setText('#info-plugin-tags', joinList(plugin.tags));
  setText('#info-plugin-astrbot', plugin.astrbot_version);

  setText('#info-enabled', meta.enabled ? '已开启（调度运行中）' : '已关闭（仅暂停调度）');
  setText('#info-timezone', meta.timezone);
  setText('#info-path', meta.config_path);
  setText('#info-media-path', meta.media_path);
  setText('#info-decision-log-path', meta.debug_paths?.decision);
  setText('#info-chat-log-path', meta.debug_paths?.chat);
  setText('#info-schema', meta.schema_version === undefined ? DASH : `v${meta.schema_version}`);
  setText('#info-error', meta.load_error || '无');
  setText('#info-environment', environmentText(meta.environment));
}

on('overview', () => renderAbout());
on('section', (name) => {
  if (name === 'about') renderAbout();
});
