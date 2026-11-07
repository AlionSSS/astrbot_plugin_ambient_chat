/** 绑定页：左侧机器人应用清单 + 右侧该机器人下的绑定列表。 */
import { bindingDisplayName, bindingRuleName, selectionText } from '../model.js';
import { on, state } from '../store.js';
import { $, escapeHtml, kindLabel } from '../util.js';

/**
 * 左列的机器人清单，见 ADR-0013：
 * - `platforms` 里的全部平台都列出（WebChat 也列，只是不给它新增绑定）；
 * - 绑定引用但平台配置里已不存在的「孤儿」也列出，否则这些绑定看不见也删不掉。
 */
function botEntries() {
  const bots = [];
  const index = new Map();
  for (const item of state.meta?.platforms || []) {
    const bot = {
      id: item.id,
      desc: String(item.display || '').trim(),
      online: item.online,
      bindable: item.bindable !== false,
      bindings: [],
    };
    index.set(bot.id, bot);
    bots.push(bot);
  }
  for (const binding of state.bindings) {
    if (!index.has(binding.platform_id)) {
      const bot = {
        id: binding.platform_id,
        desc: '已不在平台配置中',
        online: null,
        bindable: true,
        bindings: [],
      };
      index.set(bot.id, bot);
      bots.push(bot);
    }
    index.get(binding.platform_id).bindings.push(binding);
  }
  return bots;
}

export function canCreateBinding(bot) {
  return Boolean(bot) && bot.bindable;
}

/** 选中项回落：保持当前选择，否则第一条有绑定的机器人，再否则第一条。 */
function ensureSelectedPlatform() {
  const bots = botEntries();
  if (bots.some((bot) => bot.id === state.selectedPlatform)) return;
  const withBindings = bots.find((bot) => bot.bindings.length);
  state.selectedPlatform = (withBindings || bots[0])?.id || '';
}

export function selectedBot() {
  return botEntries().find((bot) => bot.id === state.selectedPlatform) || null;
}

export function blankBinding() {
  return {
    id: '',
    platform_id: state.selectedPlatform,
    rule_id: state.rules[0]?.id || '',
    name: '',
    targets: [],
    // 新建默认按关注度挑：同样的名额，比等概率随机更聪明（ADR-0034）。
    selection: { mode: 'weighted', count: 3 },
    // 新建即停止：启停只在「运行管理」页操作，见 ADR-0015。
    enabled: false,
  };
}

/**
 * 该机器人应用下已被其它绑定占用的会话（umo → 占用它的绑定）。
 * 编辑既有绑定时排除自己；目标会话在机器人应用内唯一，见 ADR-0012。
 */
export function occupiedTargets(platformId) {
  const occupied = new Map();
  for (const binding of state.bindings) {
    if (binding.platform_id !== platformId) continue;
    if (binding.id && binding.id === state.editingBindingId) continue;
    for (const umo of binding.targets || []) occupied.set(umo, binding);
  }
  return occupied;
}

export function renderBindings() {
  ensureSelectedPlatform();
  renderBotList();
  renderBindingPanel();
}

function renderBotList() {
  const list = $('#bot-list');
  const bots = botEntries();
  $('#bot-count').textContent = bots.length ? `${bots.length} 个` : '';
  if (!bots.length) {
    list.innerHTML = '<li class="empty-item">AstrBot 里还没有平台适配器</li>';
    return;
  }
  list.innerHTML = bots
    .map((bot) => {
      const parts = [`${bot.bindings.length} 条绑定`];
      if (!bot.bindable) parts.push('不可新增');
      else if (bot.online === false) parts.push('未连接');
      const active = bot.id === state.selectedPlatform ? 'active' : '';
      return `<li data-bot-id="${escapeHtml(bot.id)}" class="${active}">
        <span class="bot-id">${escapeHtml(bot.id)}</span>
        ${bot.desc ? `<span class="bot-desc">${escapeHtml(bot.desc)}</span>` : ''}
        <span class="meta">${escapeHtml(parts.join(' · '))}</span>
      </li>`;
    })
    .join('');
}

function renderBindingPanel() {
  const bot = selectedBot();
  const list = $('#binding-list');
  const empty = $('#binding-empty');
  const newButton = $('#btn-new-binding');
  const hint = $('#binding-bot-hint');
  const label = $('#binding-bot-label');
  const desc = $('#binding-bot-desc');
  const meta = $('#binding-bot-meta');
  // 顶部保留标题与描述；UMO 前缀提示说明已移除，不再展示。
  hint.classList.add('hidden');
  // 绑定条数统计在左侧机器人应用处已有展示，右侧顶部不再重复。
  meta.classList.add('hidden');

  if (!bot) {
    label.textContent = '未选择机器人';
    desc.textContent = '';
    meta.textContent = '';
    hint.textContent = '';
    list.innerHTML = '';
    empty.textContent = 'AstrBot 里还没有平台适配器，请先在「消息平台」里添加一个机器人应用。';
    empty.classList.remove('hidden');
    newButton.disabled = true;
    return;
  }

  label.textContent = bot.id;
  desc.textContent = bot.desc;
  desc.classList.toggle('hidden', !bot.desc);

  const chips = [`${bot.bindings.length} 条绑定`];
  if (!bot.bindable) chips.push('不能主动发消息');
  if (bot.online === false) chips.push('未连接');
  meta.textContent = chips.join(' · ');

  newButton.disabled = !canCreateBinding(bot);
  newButton.title = bot.bindable ? '' : 'WebChat 不能主动发消息，无法新增绑定';

  list.innerHTML = bot.bindings.map(bindingRow).join('');
  const isEmpty = !bot.bindings.length;
  empty.classList.toggle('hidden', !isEmpty);
  if (isEmpty) {
    empty.innerHTML = `<code>${escapeHtml(bot.id)}</code> 还没有绑定。点击右上角「新增绑定」为它配第一条。`;
  }
}

/** 悬停「N 个会话」时展示的明细：超过 8 条折叠。 */
function umoTooltip(targets) {
  if (!targets.length) return '';
  const rows = targets.slice(0, 8).map((umo) => {
    const parts = umo.split(':');
    return `<span><em>${escapeHtml(kindLabel(parts[1]))}</em>${escapeHtml(parts[2] || umo)}</span>`;
  });
  if (targets.length > 8) {
    rows.push(`<span class="more">… 还有 ${targets.length - 8} 个</span>`);
  }
  return `<span class="tooltip" role="tooltip">${rows.join('')}</span>`;
}

function bindingRow(binding) {
  const targets = binding.targets || [];
  const sessions = targets.length
    ? `<span class="sessions" tabindex="0">${targets.length} 个会话${umoTooltip(targets)}</span>`
    : '<span class="muted">无目标会话</span>';
  // 本页只管配置：启停状态给个标记，免得以为保存完就在跑（启停在「运行管理」）。
  // 运行中的绑定只读：能点开查看，但不能改配置、不能删，见 ADR-0016。
  const running = binding.enabled !== false;
  const stopping = binding.lifecycle === 'stopping';
  const statePill = stopping
    ? '<span class="state-pill">停止收尾中</span>'
    : binding.lifecycle === 'starting'
      ? '<span class="state-pill">启动预热中</span>'
      : running
    ? '<span class="state-pill running">运行中</span>'
    : '<span class="state-pill">未启动</span>';
  const editTitle = stopping ? '停止收尾中，完成后才能修改' : running ? '运行中只能查看，停止后才能保存修改' : '';
  const deleteTitle = running ? '运行中不能删除，请先停止' : '';
  return `<li class="binding-row" data-binding-id="${escapeHtml(binding.id)}">
    <span class="binding-name">${escapeHtml(bindingDisplayName(binding))}</span>
    ${statePill}
    <span class="rule-pill">规则「${escapeHtml(bindingRuleName(binding))}」</span>
    <span class="binding-meta">${sessions}<span class="sep">·</span><span>${escapeHtml(selectionText(binding.selection))}</span></span>
    <span class="binding-row-actions">
      <button class="btn small ghost" type="button" data-edit-binding="${escapeHtml(binding.id)}" title="${escapeHtml(editTitle)}"${stopping ? ' disabled' : ''}>编辑</button>
      <button class="btn small ghost danger" type="button" data-delete-binding="${escapeHtml(binding.id)}" title="${escapeHtml(deleteTitle)}"${running || stopping ? ' disabled' : ''}>删除</button>
    </span>
  </li>`;
}

on('overview', () => renderBindings());
on('section', (name) => {
  if (name === 'bindings') renderBindings();
});
