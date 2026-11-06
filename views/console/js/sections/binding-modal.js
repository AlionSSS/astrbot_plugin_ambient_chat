/** 绑定编辑弹窗：目标会话、挑选策略与保存/删除。 */
import { apiGet, apiPost } from '../api.js';
import {
  bindingDisplayName,
  bindingRuleName,
  currentBindingPhase,
  platformParts,
} from '../model.js';
import { finishSaved } from '../shell.js';
import { state } from '../store.js';
import { askConfirm, toast } from '../ui.js';
import { $, clone, escapeHtml, kindLabel } from '../util.js';
import { blankBinding, canCreateBinding, occupiedTargets, selectedBot } from './bindings.js';

function bindingPlatformId() {
  return state.editingBinding?.platform_id || state.selectedPlatform;
}

export function openBindingModal(bindingId) {
  const bot = selectedBot();
  if (!bot) {
    toast('先在左侧选择一个机器人应用', true);
    return;
  }
  const binding = bindingId
    ? state.bindings.find((item) => item.id === bindingId)
    : null;
  if (!binding && !canCreateBinding(bot)) {
    toast('WebChat 不能主动发消息，无法新增绑定', true);
    return;
  }
  state.editingBindingId = binding?.id || null;
  state.editingBinding = binding ? clone(binding) : blankBinding();
  // 新建与编辑在打开时都还没动过表单，关掉不必提示；任何输入/变更会把它置为脏。
  state.bindingDirty = false;
  state.sessions = [];
  $('#binding-modal-title').textContent = binding
    ? (binding.enabled ? '编辑绑定（运行中，只读）' : '编辑绑定')
    : '新增绑定';
  renderBindingModal();
  $('#binding-modal').classList.remove('hidden');
  $('#binding-name').focus();
}

function closeBindingModal() {
  $('#binding-modal').classList.add('hidden');
  state.editingBindingId = null;
  state.editingBinding = null;
  state.bindingDirty = false;
  state.sessions = [];
}

/** 有未保存修改时先确认，避免误点遮罩丢掉刚填的目标会话。 */
export async function requestCloseBindingModal() {
  if (!state.editingBinding) return;
  if (state.bindingDirty) {
    const ok = await askConfirm(
      '关闭后本次修改会丢弃，确定放弃？',
      '放弃未保存的修改',
      '放弃修改',
    );
    if (!ok) return;
  }
  closeBindingModal();
}

function renderBindingModal() {
  const binding = state.editingBinding;
  if (!binding) return;
  const bot = selectedBot();
  const platform = bot || platformParts(binding.platform_id);
  $('#binding-modal-bot').textContent = platform.id;
  // 弹窗右上角只显示机器人 id（如「机器人 default」），不展示 id 后面的描述文案。
  $('#binding-modal-desc').textContent = '';
  $('#binding-modal-desc').classList.add('hidden');
  $('#binding-umo-prefix').textContent = bindingPlatformId() || '-';
  $('#binding-name').value = binding.name || '';

  const ruleSelect = $('#binding-rule');
  if (!state.rules.length) {
    ruleSelect.innerHTML = '<option value="">（还没有规则，请先创建）</option>';
  } else {
    ruleSelect.innerHTML = state.rules
      .map(
        (rule) =>
          `<option value="${escapeHtml(rule.id)}">${escapeHtml(rule.name)}</option>`,
      )
      .join('');
  }
  ruleSelect.value = binding.rule_id || '';

  $('#binding-targets').value = (binding.targets || []).join('\n');
  $('#binding-selection-mode').value = binding.selection?.mode || 'all';
  $('#binding-selection-count').value = Number(binding.selection?.count) || 1;
  renderNameHint();
  renderSelectionHint();
  renderTargetConflicts();
  renderSessionPicker();
  // 运行中只读：可以打开查看，但保存按下去只会被后端拒绝（ADR-0016），提前禁用。
  const locked = binding.enabled !== false || binding.lifecycle === 'stopping';
  const saveButton = $('#btn-save-binding');
  saveButton.disabled = locked;
  saveButton.textContent = binding.lifecycle === 'stopping' ? '停止收尾中' : locked ? '运行中不可保存' : '保存';
  saveButton.title = locked ? '请先到「运行管理」停止该绑定后再修改' : '';
}

function renderNameHint() {
  const hint = $('#binding-name-hint');
  const name = $('#binding-name').value.trim();
  const platformId = bindingPlatformId();
  const clash =
    Boolean(name) &&
    state.bindings.some(
      (item) =>
        item.platform_id === platformId &&
        item.id !== state.editingBindingId &&
        item.name === name,
    );
  hint.textContent = clash ? '该机器人下已有同名绑定（允许重名，仅作提醒）。' : '';
  hint.classList.toggle('hidden', !clash);
}

/** 与同一机器人应用的其它绑定冲突的目标会话（后端才是权威，这里只提前告知）。 */
function targetConflicts() {
  const platformId = bindingPlatformId();
  if (!platformId) return [];
  const occupying = occupiedTargets(platformId);
  const conflicts = [];
  for (const umo of parseTargets()) {
    const owner = occupying.get(umo);
    if (!owner) continue;
    conflicts.push(`会话 ${umo} 已被绑定「${bindingDisplayName(owner)}」占用`);
  }
  return conflicts;
}

function renderTargetConflicts() {
  const conflicts = targetConflicts();
  const hint = $('#target-conflict-hint');
  if (hint) {
    hint.classList.toggle('hidden', !conflicts.length);
    hint.textContent = conflicts.join('；');
  }
  const save = $('#btn-save-binding');
  if (save) save.disabled = conflicts.length > 0;
  return conflicts;
}

function renderSelectionHint() {
  const targets = parseTargets();
  const mode = $('#binding-selection-mode').value;
  const countInput = $('#binding-selection-count');
  const count = Number(countInput.value) || 1;
  const hint = $('#selection-hint');
  // 「全部会话」策略下 N 不生效，禁用输入以表达这一点。
  const randomMode = mode === 'random';
  countInput.disabled = !randomMode;
  countInput.title = randomMode ? '' : '选择「全部会话」时 N 不生效';
  countInput.closest('.field')?.classList.toggle('is-disabled', !randomMode);
  if (!randomMode) {
    hint.textContent = `当前 ${targets.length} 个目标会话，每次触发全部处理（按列表顺序），N 不生效。`;
    return;
  }
  const effective = targets.length ? Math.min(count, targets.length) : count;
  hint.textContent = `当前 ${targets.length} 个目标会话，每次触发随机挑 ${effective} 个，按挑选顺序依次走「拦截 → 意向 → 投递」。`;
}

function parseTargets() {
  return $('#binding-targets')
    .value.split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

function collectBinding() {
  const binding = state.editingBinding || blankBinding();
  return {
    id: binding.id || '',
    platform_id: bindingPlatformId(),
    rule_id: $('#binding-rule').value,
    name: $('#binding-name').value.trim(),
    targets: parseTargets(),
    selection: {
      mode: $('#binding-selection-mode').value,
      count: Number($('#binding-selection-count').value) || 1,
    },
    // 启停由「运行管理」页管理，配置页只原样保留当前状态（后端也按此规则兜底）。
    enabled: binding.enabled === true,
  };
}

function syncBindingFromDom() {
  if (!state.editingBinding) return;
  Object.assign(state.editingBinding, collectBinding());
}

async function loadSessions() {
  const platformId = bindingPlatformId();
  const notice = $('#session-notice');
  if (!platformId) {
    toast('先选择一个机器人应用', true);
    return;
  }
  notice.textContent = '读取中…';
  try {
    const data = await apiGet('sessions', { platform_id: platformId, limit: 60 });
    state.sessions = (data.sessions || []).filter((item) => item.platform_id === platformId);
    renderSessionPicker();
    notice.textContent = data.notice || (state.sessions.length ? '' : '没有找到该机器人的历史会话');
  } catch (error) {
    notice.textContent = '';
    toast(error.message || '读取会话失败', true);
  }
}

function renderSessionPicker() {
  const picker = $('#session-picker');
  if (!state.sessions.length) {
    picker.innerHTML = '';
    return;
  }
  const existing = new Set(parseTargets());
  const occupying = occupiedTargets(bindingPlatformId());
  picker.innerHTML = state.sessions
    .map((item) => {
      // 已被该机器人其它绑定占用的会话直接禁用：保存时后端也会拒绝（ADR-0012）。
      const owner = occupying.get(item.umo);
      const checked = existing.has(item.umo);
      return `<li class="${owner ? 'is-occupied' : ''}">
        <input type="checkbox" value="${escapeHtml(item.umo)}" ${checked ? 'checked' : ''} ${owner ? 'disabled' : ''} />
        <span>${kindLabel(item.message_type)} ${escapeHtml(item.session_id)}</span>
        <span class="muted">${escapeHtml(item.title || '')}</span>
        ${owner ? `<span class="occupied-tag">已绑定「${escapeHtml(bindingRuleName(owner))}」</span>` : ''}
        <span class="umo">${escapeHtml(item.umo)}</span>
      </li>`;
    })
    .join('');
}

function addSelectedSessions() {
  const checked = [...document.querySelectorAll('#session-picker input:checked')]
    .filter((input) => !input.disabled)
    .map((input) => input.value);
  if (!checked.length) {
    toast('请先勾选会话', true);
    return;
  }
  const current = parseTargets();
  for (const umo of checked) {
    if (!current.includes(umo)) current.push(umo);
  }
  $('#binding-targets').value = current.join('\n');
  state.bindingDirty = true;
  syncBindingFromDom();
  renderSelectionHint();
  renderTargetConflicts();
  toast(`已加入 ${checked.length} 个会话`);
}

async function saveBinding() {
  if (!state.editingBinding) return;
  if (state.editingBinding.enabled !== false || currentBindingPhase(state.editingBinding.id) === 'stopping') {
    toast('绑定运行中，请先到「运行管理」停止后再修改', true);
    return;
  }
  const binding = collectBinding();
  if (!binding.platform_id) {
    toast('请先在左侧选择一个机器人应用', true);
    return;
  }
  if (!binding.rule_id) {
    toast('请选择规则', true);
    return;
  }
  if (!binding.targets.length) {
    toast('至少填写一个目标会话', true);
    return;
  }
  const conflicts = renderTargetConflicts();
  if (conflicts.length) {
    toast(conflicts[0], true);
    return;
  }
  const button = $('#btn-save-binding');
  button.disabled = true;
  try {
    const data = await apiPost('bindings/save', { binding });
    closeBindingModal();
    // 保存不改变启停：新建的绑定仍是停止态，要生效得去「运行管理」启动（ADR-0015）。
    await finishSaved(
      data,
      data?.binding?.enabled
        ? '绑定已保存（保持运行中）'
        : '绑定已保存，到「运行管理」启动后才会生效',
    );
  } catch (error) {
    // 后端才是权威：会话冲突、规则不存在等都以它的报文为准，弹窗保持打开。
    toast(error.message || '保存失败', true);
  } finally {
    button.disabled = false;
    if (state.editingBinding) renderTargetConflicts();
  }
}

/** 删除在行内触发（弹窗里只有保存/取消），见 ADR-0013。 */
async function deleteBindingRow(bindingId) {
  const binding = state.bindings.find((item) => item.id === bindingId);
  if (!binding) return;
  if (binding.enabled || currentBindingPhase(bindingId) === 'stopping') {
    toast('绑定运行中，请先到「运行管理」停止后再删除', true);
    return;
  }
  const name = bindingDisplayName(binding);
  const count = (binding.targets || []).length;
  const ok = await askConfirm(
    `确定删除绑定「${name}」？该机器人下的 ${count} 个目标会话将一并释放。`,
    '删除绑定',
  );
  if (!ok) return;
  try {
    const data = await apiPost('bindings/delete', { binding_id: bindingId });
    if (state.editingBindingId === bindingId) closeBindingModal();
    await finishSaved(data, '绑定已删除');
  } catch (error) {
    toast(error.message || '删除失败', true);
  }
}

/* ------------------------------------------------------------ 弹窗内的输入 */
document.addEventListener('input', (event) => {
  if (!state.editingBinding) return;
  if (!event.target.closest('#binding-modal')) return;
  state.bindingDirty = true;
  syncBindingFromDom();
  if (event.target.id === 'binding-name') renderNameHint();
  if (
    event.target.id === 'binding-selection-mode' ||
    event.target.id === 'binding-selection-count' ||
    event.target.id === 'binding-targets'
  ) {
    renderSelectionHint();
    renderTargetConflicts();
  }
});

document.addEventListener('change', (event) => {
  if (!state.editingBinding) return;
  if (!event.target.closest('#binding-modal')) return;
  state.bindingDirty = true;
  syncBindingFromDom();
  if (
    event.target.id === 'binding-selection-mode' ||
    event.target.id === 'binding-selection-count'
  ) {
    renderSelectionHint();
  }
  if (event.target.id === 'binding-rule') renderTargetConflicts();
});

export const actions = {
  'btn-new-binding': () => openBindingModal(null),
  'btn-load-sessions': () => loadSessions(),
  'btn-add-sessions': () => addSelectedSessions(),
  'btn-save-binding': () => saveBinding(),
  'btn-cancel-binding': () => requestCloseBindingModal(),
  'data:editBinding': (bindingId) => openBindingModal(bindingId),
  'data:deleteBinding': (bindingId) => deleteBindingRow(bindingId),
};
