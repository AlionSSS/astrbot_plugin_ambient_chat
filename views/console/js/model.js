/**
 * 与后端数据结构绑定的常量与纯函数。
 * 这里的函数可以从 `state` 读数据，但不接触 DOM，也不发请求。
 */
import { state } from './store.js';

export const DEFAULT_UNREAD_MIN_COUNT = 10;

export const BLOCKER_DEFAULTS = {
  silence: () => ({ type: 'silence', windows: [{ start: '23:00', end: '09:00' }] }),
  cooldown: () => ({ type: 'cooldown', minutes: 5 }),
  unread: () => ({ type: 'unread', min_count: DEFAULT_UNREAD_MIN_COUNT }),
  unanswered: () => ({ type: 'unanswered', count: 3, minutes: 180 }),
};

/** 未读条数允许为 0（等于不拦截），只在缺失或非法时回落到默认值。 */
export function unreadMinCount(blocker) {
  const raw = blocker?.min_count;
  if (raw === null || raw === undefined || raw === '') return DEFAULT_UNREAD_MIN_COUNT;
  const value = Number(raw);
  return Number.isFinite(value) ? value : DEFAULT_UNREAD_MIN_COUNT;
}

export const TRIGGER_DEFAULTS = {
  cron: () => ({ type: 'cron', expr: '*/30 * * * *' }),
  once: () => ({ type: 'once', at: '' }),
  window: () => ({
    type: 'window',
    start: '14:00',
    end: '18:00',
    count: 3,
    gap_minutes: 1,
    days: [],
  }),
  interval: () => ({ type: 'interval', every_minutes: 60, jitter_percent: 15 }),
};

/**
 * crontab 预设：只覆盖常见节奏，选「自定义」仍可手写。
 * 星期用 mon-fri / sat,sun 这样的名字而不是数字：APScheduler 的 0 是周一，
 * 与标准 crontab 的 0=周日 不一致，用数字容易写反。
 */
export const CRON_PRESETS = [
  { label: '每 15 分钟', expr: '*/15 * * * *' },
  { label: '每 30 分钟', expr: '*/30 * * * *' },
  { label: '每小时整点', expr: '0 * * * *' },
  { label: '每天 10:00', expr: '0 10 * * *' },
  { label: '每天 20:00', expr: '0 20 * * *' },
  { label: '工作日 10:00', expr: '0 10 * * mon-fri' },
  { label: '周末 15:00', expr: '0 15 * * sat,sun' },
];

export const WEEKDAY_LABELS = ['一', '二', '三', '四', '五', '六', '日'];

/** 一次性触发是否已经过期：后端允许保存过去时刻，只在页面上标红。 */
export function triggerExpired(trigger) {
  if (trigger?.type !== 'once' || !trigger.at) return false;
  const at = new Date(trigger.at).getTime();
  return !Number.isNaN(at) && at <= Date.now();
}

export function blankRule() {
  return {
    id: '',
    name: '新规则',
    triggers: [TRIGGER_DEFAULTS.cron()],
    blockers: [BLOCKER_DEFAULTS.silence(), BLOCKER_DEFAULTS.unanswered()],
    intent: {
      mode: 'direct',
      decision_provider_id: '',
      threshold: 0.7,
      pass_probability: 1,
    },
    chat_provider_id: '',
    context_turns: 20,
    inject_persona: true,
  };
}

export function normalizeRule(rule) {
  rule.context_turns = Number(rule.context_turns) || 20;
  rule.intent.threshold = Number(rule.intent.threshold);
  // 历史规则没有该字段：缺省按 100% 处理，与后端默认值一致。
  const probability = Number(rule.intent.pass_probability);
  rule.intent.pass_probability = Number.isFinite(probability) ? probability : 1;
  rule.triggers = (rule.triggers || []).map((trigger) => {
    if (trigger.type === 'window') {
      trigger.count = Number(trigger.count) || 1;
      trigger.gap_minutes = Number(trigger.gap_minutes) || 1;
      trigger.days = (trigger.days || [])
        .map((day) => Number(day))
        .filter((day) => Number.isInteger(day) && day >= 1 && day <= 7);
    }
    if (trigger.type === 'interval') {
      trigger.every_minutes = Number(trigger.every_minutes) || 60;
      trigger.jitter_percent = Number(trigger.jitter_percent) || 0;
    }
    return trigger;
  });
  rule.blockers = (rule.blockers || []).map((blocker) => {
    if (blocker.type === 'cooldown') blocker.minutes = Number(blocker.minutes) || 5;
    if (blocker.type === 'unread') blocker.min_count = unreadMinCount(blocker);
    if (blocker.type === 'unanswered') {
      blocker.count = Number(blocker.count) || 3;
      blocker.minutes = Number(blocker.minutes) || 180;
    }
    return blocker;
  });
  return rule;
}

export function findBlocker(rule, type) {
  return (rule.blockers || []).find((item) => item.type === type) || null;
}

export function bindingsOfRule(ruleId) {
  return state.bindings.filter((binding) => binding.rule_id === ruleId);
}

/**
 * 机器人应用的展示片段：`id`（UMO 前缀）是主标识，适配器自带的介绍文字退为次要说明。
 * 平台已不在配置里的历史 id 用同一结构表达，避免出现两种排版。
 */
export function platformParts(platformId) {
  const item = (state.meta?.platforms || []).find((p) => p.id === platformId);
  const id = platformId || '未选择机器人';
  if (!item) {
    return { id, desc: platformId ? '已不在平台配置中' : '' };
  }
  const desc = String(item.display || '').trim();
  return { id: item.id, desc: desc === item.id ? '' : desc };
}

export function selectionText(selection) {
  if (!selection || selection.mode !== 'random') return '全部会话';
  return `随机 ${selection.count} 个`;
}

export function bindingRuleName(binding) {
  return binding?.rule_name || binding?.rule_id || '';
}

/** 行的主标识：名称恒有值（后端兜底生成「未命名-N」），缺名时回退规则名。 */
export function bindingDisplayName(binding) {
  return binding?.name || bindingRuleName(binding);
}

/** 绑定当前的运行阶段：先看列表数据，轮询回来的运行状态兜底。 */
export function currentBindingPhase(id) {
  return state.bindings.find((item) => String(item.id) === String(id))?.lifecycle
    || state.status.find((item) => String(item.binding_id) === String(id))?.lifecycle;
}
