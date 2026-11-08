/** 规则页：列表、分页与规则编辑弹窗。 */
import { apiPost } from '../api.js';
import {
  BLOCKER_DEFAULTS,
  CRON_PRESETS,
  TRIGGER_DEFAULTS,
  WEEKDAY_LABELS,
  bindingDisplayName,
  bindingsOfRule,
  blankRule,
  findBlocker,
  normalizeRule,
  triggerExpired,
  unreadMinCount,
} from '../model.js';
import { finishSaved } from '../shell.js';
import { on, state } from '../store.js';
import { askConfirm, askInfo, toast } from '../ui.js';
import { $, clone, escapeHtml, fromLocalInput, getPath, setPath, toLocalInput } from '../util.js';

/** 规则列表每页条数，见 ADR-0014：只在前端切片，接口仍返回全量。 */
const RULES_PER_PAGE = 10;

/** 行内的修改时间：绝对时间，规则页不轮询，相对时间会停在旧值。 */
function ruleUpdatedText(rule) {
  const raw = rule?.updated_at || '';
  if (!raw) return '-';
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return '-';
  const pad = (value) => String(value).padStart(2, '0');
  const stamp = `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  return date.getFullYear() === new Date().getFullYear()
    ? `${stamp} 修改`
    : `${date.getFullYear()}-${stamp} 修改`;
}

function ruleRow(rule) {
  const bound = bindingsOfRule(rule.id).length;
  const triggers = (rule.triggers || []).length;
  const expired = (rule.triggers || []).filter(triggerExpired).length;
  const expiredText = expired
    ? `<span class="expired">${expired} 个入口已过期</span><span class="sep">·</span>`
    : '';
  const deleteTitle = bound ? `被 ${bound} 条绑定使用，需先解除绑定才能删除` : '';
  return `<li class="rule-row" data-rule-id="${escapeHtml(rule.id)}">
    <span class="rule-name">${escapeHtml(rule.name)}</span>
    <span class="rule-meta">
      <span>${bound} 条绑定</span><span class="sep">·</span>
      <span>${triggers} 个触发入口</span><span class="sep">·</span>
      ${expiredText}
      <span>${escapeHtml(ruleUpdatedText(rule))}</span>
    </span>
    <span class="rule-row-actions">
      <button class="btn small ghost" type="button" data-edit-rule="${escapeHtml(rule.id)}">编辑</button>
      <button class="btn small ghost" type="button" data-copy-rule="${escapeHtml(rule.id)}"
        title="以该规则为模板新建">复制</button>
      <button class="btn small ghost danger" type="button" data-delete-rule="${escapeHtml(rule.id)}"
        title="${escapeHtml(deleteTitle)}">删除</button>
    </span>
  </li>`;
}

export function renderRuleList() {
  const list = $('#rule-list');
  const total = state.rules.length;
  const pages = Math.max(1, Math.ceil(total / RULES_PER_PAGE));
  // 规则变少（删除）时把页码收回到合法范围，末页空了自然回退一页。
  state.rulePage = Math.min(Math.max(1, state.rulePage), pages);
  $('#rule-count').textContent = total ? `共 ${total} 条` : '';

  if (!total) {
    list.innerHTML =
      '<li class="empty-item">还没有规则，点击右上角「新建规则」创建第一条。</li>';
    renderRulePager(pages);
    return;
  }

  const start = (state.rulePage - 1) * RULES_PER_PAGE;
  list.innerHTML = state.rules.slice(start, start + RULES_PER_PAGE).map(ruleRow).join('');
  renderRulePager(pages);
}

function renderRulePager(pages) {
  const pager = $('#rule-pager');
  pager.classList.toggle('hidden', pages <= 1);
  if (pages <= 1) return;
  $('#rule-page-label').textContent = `第 ${state.rulePage}/${pages} 页`;
  $('#btn-rule-prev').disabled = state.rulePage <= 1;
  $('#btn-rule-next').disabled = state.rulePage >= pages;
}

export function openRuleModal(ruleId, draft) {
  const rule = ruleId ? state.rules.find((item) => item.id === ruleId) : null;
  state.editingRuleId = rule?.id || null;
  state.editingRule = rule ? clone(rule) : draft ? clone(draft) : blankRule();
  // 新建时用户还没动过表单，关掉不必提示；任何输入会把它置为脏。
  state.ruleDirty = false;
  $('#rule-modal-title').textContent = rule ? `编辑规则 · ${rule.name}` : '新建规则';
  renderRuleModal();
  $('#rule-modal').classList.remove('hidden');
  document.querySelector('#rule-editor [data-path="name"]')?.focus();
}

function closeRuleModal() {
  $('#rule-modal').classList.add('hidden');
  state.editingRuleId = null;
  state.editingRule = null;
  state.ruleDirty = false;
}

export async function requestCloseRuleModal() {
  if (!state.editingRule) return;
  if (state.ruleDirty) {
    const ok = await askConfirm(
      '关闭后本次修改会丢弃，确定放弃？',
      '放弃未保存的修改',
      '放弃修改',
    );
    if (!ok) return;
  }
  closeRuleModal();
}

function renderRuleModal() {
  const rule = state.editingRule;
  if (!rule) return;

  for (const input of document.querySelectorAll('#rule-editor [data-path]')) {
    const value = getPath(rule, input.dataset.path);
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else input.value = fieldToInput(input.dataset.path, value);
  }

  // 注入条数受「上下文缓冲」容量约束，超出部分取不到，这里把当前上限写明。
  const note = $('#context-turns-limit');
  if (note) {
    const bufferSize = Number(state.meta?.settings?.buffer_size);
    note.textContent = bufferSize > 0 ? `（缓冲区配置上限 ${bufferSize} 条）` : '';
  }

  fillProviderSelect(
    document.querySelector('#rule-editor [data-path="chat_provider_id"]'),
    state.meta.chat_providers,
    '使用 AstrBot 会话默认模型',
  );
  fillProviderSelect(
    document.querySelector('#rule-editor [data-path="intent.decision_provider_id"]'),
    state.meta.decision_providers,
    '使用 AstrBot 默认判断模型',
  );

  renderTriggers();
  renderBlockers();
  syncIntentFields();
}

/**
 * 选「使用 AstrBot 默认判断模型」时，`DecisionRouter` 会取官方判断提供商列表的第 0 个；
 * 一个都没配置，判定会直接失败（按设计静默不发言）。「更多设置」里的自定义模型不算，
 * 因为它不在 AstrBot 的提供商配置里，必须在下拉里显式选中才生效。
 */
function defaultDecisionProviderUnavailable() {
  const mode = document.querySelector('#rule-editor [data-path="intent.mode"]')?.value;
  const selected = document.querySelector(
    '#rule-editor [data-path="intent.decision_provider_id"]',
  )?.value;
  const hasOfficial = (state.meta?.decision_providers || []).some(
    (item) => item.source === 'official',
  );
  return mode === 'decision' && !selected && !hasOfficial;
}

/** 红字提示 + 禁用保存：这条规则存下去也只会一直判定失败。 */
function syncDecisionProviderWarning() {
  const blocked = defaultDecisionProviderUnavailable();
  $('#decision-provider-warning')?.classList.toggle('hidden', !blocked);
  const button = $('#btn-save-rule');
  if (!button) return;
  button.disabled = blocked;
  button.title = blocked
    ? 'AstrBot 未配置判断提供商，请先配置或改用自定义决策模型'
    : '';
}

function syncIntentFields() {
  const modeInput = document.querySelector('#rule-editor [data-path="intent.mode"]');
  if (!modeInput) return;
  const decisionMode = modeInput.value === 'decision';
  // 两种机制各自只用一部分字段：用不到的整块隐藏，只留下当前机制要填的项。
  const activePaths = decisionMode
    ? ['intent.decision_provider_id', 'intent.threshold']
    : ['intent.pass_probability'];
  for (const path of [
    'intent.decision_provider_id',
    'intent.threshold',
    'intent.pass_probability',
  ]) {
    const input = document.querySelector(`#rule-editor [data-path="${path}"]`);
    if (!input) continue;
    const active = activePaths.includes(path);
    input.disabled = !active;
    input.closest('.field')?.classList.toggle('hidden', !active);
  }
  const hint = $('#intent-hint');
  if (hint) {
    hint.textContent = decisionMode
      ? '决策模型会先给当前会话的消息打分，达到阈值视为对会话内容感兴趣或想发起新话题，才会发言。' +
        '下拉里带「（自定义）」的项来自「更多设置」的决策模型配置。'
      : '当前为按概率放行：只要通过第 2 步「拦截」，就按设定概率决定是否发言；' +
        '放行概率填 100 等于每次都发言，填 0 等于从不发言，决策模型不参与判断。';
  }
  syncDecisionProviderWarning();
}

function fillProviderSelect(select, providers, emptyLabel) {
  if (!select) return;
  const current = select.value;
  const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`];
  for (const provider of providers || []) {
    options.push(
      `<option value="${escapeHtml(provider.id)}">${escapeHtml(provider.label || provider.id)}</option>`,
    );
  }
  select.innerHTML = options.join('');
  const value = state.editingRule ? getPath(state.editingRule, select.dataset.path) : current;
  select.value = value ?? '';
}

function renderTriggers() {
  const container = $('#triggers');
  container.innerHTML = '';
  state.editingRule.triggers.forEach((trigger, index) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="card-head">
        <select data-trigger-type="${index}">
          <option value="cron">crontab</option>
          <option value="once">一次性时刻</option>
          <option value="window">时段随机</option>
          <option value="interval">固定间隔</option>
        </select>
        ${triggerExpired(trigger) ? '<span class="expired">已过期</span>' : ''}
        <button class="btn small danger" data-remove-trigger="${index}" type="button">删除</button>
      </div>
      ${triggerParams(trigger, index)}
    `;
    container.appendChild(card);
  });
  for (const select of container.querySelectorAll('[data-trigger-type]')) {
    const index = Number(select.dataset.triggerType);
    select.value = state.editingRule.triggers[index].type;
    select.addEventListener('change', (event) => {
      const target = Number(event.target.dataset.triggerType);
      state.editingRule.triggers[target] = TRIGGER_DEFAULTS[event.target.value]();
      state.ruleDirty = true;
      renderTriggers();
    });
  }
}

function triggerParams(trigger, index) {
  if (trigger.type === 'cron') {
    const expr = trigger.expr || '';
    const options = ['<option value="">自定义表达式</option>']
      .concat(
        CRON_PRESETS.map(
          (preset) =>
            `<option value="${escapeHtml(preset.expr)}" ${preset.expr === expr ? 'selected' : ''}>${escapeHtml(preset.label)}</option>`,
        ),
      )
      .join('');
    return `<div class="grid">
      <label class="field"><span>常用节奏</span>
        <select data-cron-preset="${index}">${options}</select></label>
      <label class="field"><span>crontab 表达式（分 时 日 月 周）</span>
        <input type="text" data-path="triggers.${index}.expr" value="${escapeHtml(expr)}" placeholder="*/30 * * * *" /></label>
    </div>
    <p class="hint">星期请写 mon-fri、sat,sun 这样的名字：表达式按 APScheduler 解析，数字里 0 表示周一。</p>`;
  }
  if (trigger.type === 'once') {
    return `<label class="field"><span>触发时间</span>
      <input type="datetime-local" data-once-at="${index}" value="${escapeHtml(toLocalInput(trigger.at))}" /></label>
      ${
        triggerExpired(trigger)
          ? '<p class="hint error">该时间已过期，此触发器不会再触发；改成一个将来的时间即可照常复用。</p>'
          : ''
      }`;
  }
  if (trigger.type === 'interval') {
    return `<div class="grid">
      <label class="field"><span>每隔多少分钟</span>
        <input type="number" min="1" max="1440" data-path="triggers.${index}.every_minutes" value="${Number(trigger.every_minutes) || 30}" /></label>
      <label class="field"><span>抖动比例（±%，0 表示严格等间隔）</span>
        <input type="number" min="0" max="60" data-path="triggers.${index}.jitter_percent" value="${Number(trigger.jitter_percent) || 0}" /></label>
    </div>
    <p class="hint">从绑定启动时刻起算，不按整点对齐；保存配置不会重新起算，插件重启后重新起算。</p>`;
  }

  const days = trigger.days || [];
  const dayChips = WEEKDAY_LABELS.map((label, offset) => {
    const day = offset + 1;
    const checked = !days.length || days.includes(day);
    return `<label class="day-chip">
      <input type="checkbox" data-trigger-day="${index}" data-day="${day}" ${checked ? 'checked' : ''} />
      <span>周${label}</span>
    </label>`;
  }).join('');
  return `<div class="grid">
    <label class="field"><span>开始</span>
      <input type="time" data-path="triggers.${index}.start" value="${escapeHtml(trigger.start || '')}" /></label>
    <label class="field"><span>结束</span>
      <input type="time" data-path="triggers.${index}.end" value="${escapeHtml(trigger.end || '')}" /></label>
    <label class="field"><span>窗口内随机触发次数</span>
      <input type="number" min="1" max="20" data-path="triggers.${index}.count" value="${Number(trigger.count) || 1}" /></label>
    <label class="field"><span>两次触发的最小间隔（分钟）</span>
      <input type="number" min="1" max="720" data-path="triggers.${index}.gap_minutes" value="${Number(trigger.gap_minutes) || 1}" /></label>
  </div>
  <div class="field"><span>生效星期（全选＝每天）</span>
    <div class="day-picker">${dayChips}</div>
  </div>
  <p class="hint">窗口内任意两次触发至少相隔「最小间隔」；窗口太窄放不下时保存会被拒绝。</p>`;
}

function toggleBlocker(type, enabled) {
  if (!state.editingRule) return;
  const current = findBlocker(state.editingRule, type);
  if (enabled && !current) {
    state.editingRule.blockers.push(BLOCKER_DEFAULTS[type]());
  } else if (!enabled && current) {
    state.editingRule.blockers = state.editingRule.blockers.filter(
      (item) => item.type !== type,
    );
  }
  state.ruleDirty = true;
  renderBlockers();
}

function renderBlockers() {
  const container = $('#blockers');
  const rule = state.editingRule;
  const silence = findBlocker(rule, 'silence');
  const cooldown = findBlocker(rule, 'cooldown');
  const unread = findBlocker(rule, 'unread');
  const unanswered = findBlocker(rule, 'unanswered');

  const silenceWindows = (silence?.windows || [])
    .map(
      (window, index) => `<div class="grid">
        <label class="field"><span>开始</span>
          <input type="time" data-silence="${index}" data-field="start" value="${escapeHtml(window.start)}" /></label>
        <label class="field"><span>结束（可跨午夜）</span>
          <input type="time" data-silence="${index}" data-field="end" value="${escapeHtml(window.end)}" /></label>
        <label class="field"><span>&nbsp;</span>
          <button class="btn small danger" data-remove-silence="${index}" type="button">删除该窗口</button></label>
      </div>`,
    )
    .join('');

  container.innerHTML = `
    <div class="card">
      <label class="field inline">
        <input type="checkbox" data-blocker="silence" ${silence ? 'checked' : ''} />
        <span>静默时段</span>
      </label>
      ${silence ? `${silenceWindows}<button class="btn ghost small" data-add-silence="1" type="button">+ 添加时段</button>` : '<p class="hint">未启用</p>'}
    </div>
    <div class="card">
      <label class="field inline">
        <input type="checkbox" data-blocker="cooldown" ${cooldown ? 'checked' : ''} />
        <span>自身活跃度冷却</span>
      </label>
      ${cooldown ? `<label class="field"><span>最近 N 分钟内已在本会话发过言则拦截</span>
        <input type="number" min="1" max="1440" data-cool-minutes="1" value="${Number(cooldown.minutes) || 5}" /></label>` : '<p class="hint">未启用</p>'}
    </div>
    <div class="card">
      <label class="field inline">
        <input type="checkbox" data-blocker="unread" ${unread ? 'checked' : ''} />
        <span>群未读消息条数不足</span>
      </label>
      ${unread ? `<label class="field"><span>未读消息少于该值则拦截（仅群聊）</span>
        <input type="number" min="0" max="10000" data-unread-min="1" value="${unreadMinCount(unread)}" /></label>` : '<p class="hint">未启用</p>'}
    </div>
    <div class="card">
      <label class="field inline">
        <input type="checkbox" data-blocker="unanswered" ${unanswered ? 'checked' : ''} />
        <span>连发无人回应，进入等待时间</span>
      </label>
      ${unanswered ? `<div class="grid">
        <label class="field"><span>机器人连续发言条数达到</span>
          <input type="number" min="1" max="100" data-unanswered-count="1" value="${Number(unanswered.count) || 3}" /></label>
        <label class="field"><span>此后无人回应则安静 N 分钟（填极大值即永久）</span>
          <input type="number" min="1" max="${Number.MAX_SAFE_INTEGER}" data-unanswered-minutes="1" value="${Number(unanswered.minutes) || 300}" /></label>
      </div>
      <p class="hint">从最后一次发言开始计时；期间有人发言立即解除，期满则连发条数重新累计（要再连发同样的条数才会再次等待）</p>` : '<p class="hint">未启用</p>'}
    </div>
  `;

  for (const checkbox of container.querySelectorAll('[data-blocker]')) {
    checkbox.addEventListener('change', (event) =>
      toggleBlocker(event.target.dataset.blocker, event.target.checked),
    );
  }
}

async function saveRule() {
  if (!state.editingRule) return;
  const rule = normalizeRule(clone(state.editingRule));
  if (!rule.name.trim()) {
    toast('规则名称不能为空', true);
    return;
  }
  // 按钮已禁用，这里再挡一层：防止绕过按钮触发保存，把跑不起来的配置写进磁盘。
  if (defaultDecisionProviderUnavailable()) {
    toast('AstrBot 未配置判断提供商，请先配置或改用自定义决策模型', true);
    return;
  }
  const button = $('#btn-save-rule');
  button.disabled = true;
  try {
    const data = await apiPost('rules/save', { rule });
    // 后端按修改时间倒序，刚保存的规则一定在第 1 页，回到第 1 页就等于落在它上面。
    state.rulePage = 1;
    closeRuleModal();
    await finishSaved(data, '规则已保存');
  } catch (error) {
    // 后端才是权威（名称空、触发入口为空等），弹窗保持打开。
    toast(error.message || '保存失败', true);
  } finally {
    // 不直接置 false：若期间判断提供商被移除，要恢复成禁用状态。
    syncDecisionProviderWarning();
  }
}

/** 删除在行内触发（弹窗里只有保存/取消），见 ADR-0014。 */
async function deleteRuleRow(ruleId) {
  const rule = state.rules.find((item) => item.id === ruleId);
  if (!rule) return;
  // 被绑定时不能删除：弹只读信息窗，逐条点名占用方及其机器人应用。
  const bindings = bindingsOfRule(ruleId);
  if (bindings.length) {
    const lines = bindings.map(
      (binding) => `· ${bindingDisplayName(binding)}（${binding.platform_id}）`,
    );
    if (bindings.length > 3) {
      lines.length = 3;
      lines.push(`… 等 ${bindings.length} 条`);
    }
    await askInfo(
      `规则「${rule.name}」正被 ${bindings.length} 条绑定使用，无法删除。\n` +
        `${lines.join('\n')}\n请先到「氛围组绑定」解除后再删除。`,
      '无法删除规则',
      '知道了',
    );
    return;
  }
  const ok = await askConfirm(`确定删除规则「${rule.name}」？`, '删除规则');
  if (!ok) return;
  try {
    const data = await apiPost('rules/delete', { rule_id: ruleId });
    if (state.editingRuleId === ruleId) closeRuleModal();
    await finishSaved(data, '规则已删除');
  } catch (error) {
    toast(error.message || '删除失败', true);
  }
}

/* -------------------------------------------------------- 规则编辑弹窗的输入 */
/** 以百分比展示的字段：模型里存 0~1，输入框里填 0~100。 */
const PERCENT_FIELDS = new Set(['intent.pass_probability']);

/** 模型值 → 输入框显示值。 */
function fieldToInput(path, value) {
  if (!PERCENT_FIELDS.has(path)) return value ?? '';
  const ratio = Number(value);
  return String(Number.isFinite(ratio) ? Math.round(ratio * 100) : 100);
}

/** 输入框显示值 → 模型值：清空或非法时回落到默认 100%。 */
function inputToField(path, raw) {
  if (!PERCENT_FIELDS.has(path)) return raw;
  if (raw === '') return 1;
  const percent = Number(raw);
  if (!Number.isFinite(percent)) return 1;
  return Math.min(100, Math.max(0, percent)) / 100;
}

/** 手改规则表单：把输入写回 `state.editingRule`，只标记脏、不整块重绘。 */
function handleEditorInput(target) {
  if (!state.editingRule) return;
  if (!target.closest('#rule-editor')) return;

  if (target.dataset.path && !target.dataset.triggerType) {
    let value = target.value;
    if (target.type === 'checkbox') value = target.checked;
    else value = inputToField(target.dataset.path, value);
    setPath(state.editingRule, target.dataset.path, value);
    state.ruleDirty = true;
    if (target.dataset.path === 'intent.mode') syncIntentFields();
    // 切换决策模型下拉后立即重算「默认判断模型不可用」的提示与保存按钮状态。
    if (target.dataset.path === 'intent.decision_provider_id') {
      syncDecisionProviderWarning();
    }
    // 手改表达式后让「常用节奏」跟着回到匹配项或「自定义」。
    const exprMatch = /^triggers\.(\d+)\.expr$/.exec(target.dataset.path);
    if (exprMatch) {
      const presetSelect = document.querySelector(
        `#triggers [data-cron-preset="${exprMatch[1]}"]`,
      );
      if (presetSelect) {
        const known = CRON_PRESETS.some((preset) => preset.expr === target.value);
        presetSelect.value = known ? target.value : '';
      }
    }
    return;
  }
  if (target.dataset.onceAt) {
    const index = Number(target.dataset.onceAt);
    state.editingRule.triggers[index].at = fromLocalInput(target.value);
    state.ruleDirty = true;
    return;
  }
  if (target.dataset.cronPreset !== undefined) {
    const index = Number(target.dataset.cronPreset);
    if (!target.value) return;
    state.editingRule.triggers[index].expr = target.value;
    // 预设只改表达式，不整块重绘，避免输入框失焦。
    const exprInput = document.querySelector(
      `#triggers [data-path="triggers.${index}.expr"]`,
    );
    if (exprInput) exprInput.value = target.value;
    state.ruleDirty = true;
    return;
  }
  if (target.dataset.triggerDay !== undefined) {
    const index = Number(target.dataset.triggerDay);
    const day = Number(target.dataset.day);
    const trigger = state.editingRule.triggers[index];
    const current = new Set(
      trigger.days && trigger.days.length ? trigger.days : [1, 2, 3, 4, 5, 6, 7],
    );
    if (target.checked) current.add(day);
    else current.delete(day);
    if (!current.size) {
      // 空列表的语义是「每天」，所以不允许把最后一天也取消掉。
      target.checked = true;
      return;
    }
    trigger.days = current.size === 7 ? [] : [...current].sort((a, b) => a - b);
    state.ruleDirty = true;
    return;
  }
  if (target.dataset.silence) {
    const blocker = findBlocker(state.editingRule, 'silence');
    const index = Number(target.dataset.silence);
    if (blocker?.windows?.[index]) {
      blocker.windows[index][target.dataset.field] = target.value;
      state.ruleDirty = true;
    }
    return;
  }
  if (target.dataset.coolMinutes) {
    const blocker = findBlocker(state.editingRule, 'cooldown');
    if (blocker) {
      blocker.minutes = Number(target.value) || 5;
      state.ruleDirty = true;
    }
    return;
  }
  if (target.dataset.unreadMin) {
    const blocker = findBlocker(state.editingRule, 'unread');
    if (blocker) {
      blocker.min_count = unreadMinCount({ min_count: target.value });
      state.ruleDirty = true;
    }
    return;
  }
  if (target.dataset.unansweredCount) {
    const blocker = findBlocker(state.editingRule, 'unanswered');
    if (blocker) {
      blocker.count = Number(target.value) || 3;
      state.ruleDirty = true;
    }
    return;
  }
  if (target.dataset.unansweredMinutes) {
    const blocker = findBlocker(state.editingRule, 'unanswered');
    if (blocker) {
      blocker.minutes = Number(target.value) || 300;
      state.ruleDirty = true;
    }
  }
}

document.addEventListener('input', (event) => handleEditorInput(event.target));

/* -------------------------------------------------------------- 订阅与动作 */
on('overview', () => {
  renderRuleList();
  if (state.editingRule) renderRuleModal();
});

export const actions = {
  'btn-new-rule': () => openRuleModal(null),
  'btn-save-rule': () => saveRule(),
  'btn-cancel-rule': () => requestCloseRuleModal(),
  'btn-rule-prev': () => {
    state.rulePage -= 1;
    renderRuleList();
  },
  'btn-rule-next': () => {
    state.rulePage += 1;
    renderRuleList();
  },
  'btn-add-trigger': () => {
    state.editingRule.triggers.push(TRIGGER_DEFAULTS.interval());
    state.ruleDirty = true;
    renderTriggers();
  },
  'data:editRule': (ruleId) => openRuleModal(ruleId),
  'data:copyRule': (ruleId) => {
    const source = state.rules.find((item) => item.id === ruleId);
    if (!source) return;
    const draft = clone(source);
    // 复制的是"字段"，不是那条规则的标识：id 置空（保存时后端重新生成），
    // 名称加「（副本）」避免和源规则同名、方便一眼区分。
    delete draft.id;
    draft.name = `${source.name}（副本）`;
    openRuleModal(null, draft);
  },
  'data:deleteRule': (ruleId) => deleteRuleRow(ruleId),
  'data:removeTrigger': (index) => {
    state.editingRule.triggers.splice(Number(index), 1);
    if (!state.editingRule.triggers.length) {
      state.editingRule.triggers.push(TRIGGER_DEFAULTS.interval());
    }
    state.ruleDirty = true;
    renderTriggers();
  },
  'data:addSilence': () => {
    const blocker = findBlocker(state.editingRule, 'silence');
    blocker.windows.push({ start: '12:00', end: '13:00' });
    state.ruleDirty = true;
    renderBlockers();
  },
  'data:removeSilence': (index) => {
    const blocker = findBlocker(state.editingRule, 'silence');
    blocker.windows.splice(Number(index), 1);
    if (!blocker.windows.length) blocker.windows.push({ start: '23:00', end: '09:00' });
    state.ruleDirty = true;
    renderBlockers();
  },
};
