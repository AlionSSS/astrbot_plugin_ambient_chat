/**
 * 首次打开控制台时的一次性提示。
 *
 * 本插件替代的是「什么时候主动开口」这一层，AstrBot 侧有两处配置会直接影响它：
 * 群消息没持久化时缓冲预热读不到历史，主动回复开着则会和本插件重复开口。
 * 用户改没改核心配置插件无从判断，因此只按「提示过没有」记录，不反复弹。
 */
import { apiPost } from './api.js';
import { state } from './store.js';
import { askInfo } from './ui.js';

const ASTRBOT_ADVICE_TITLE = '建议调整两处 AstrBot 配置';

const ASTRBOT_ADVICE =
  '都在「配置 → 扩展功能 → 群聊上下文感知」里：\n\n' +
  '1. 开启「持久化群聊消息记录」，并把「持久化最大消息数量」设为不小于本插件' +
  '「更多设置 → 上下文缓冲」的条数（默认 100）。插件启动时会从核心历史预热群聊缓冲，' +
  '没开持久化就只能等后面的消息慢慢攒，刚跑起来那段时间上下文是空的，拦截与决策都容易判错。\n\n' +
  '2. 关闭「主动回复」。它和本插件在做同一件事，同时开启会重复发言、互相抢话；' +
  '主动开口交给本插件的规则、拦截与决策模型即可。';

/** 正在等待提示结果：初始化与重绘并发时不重复弹窗。 */
let astrbotAdvicePending = false;

/**
 * 提示一次「建议调整 AstrBot 侧配置」。
 *
 * 提示状态记在 `settings.json` 的 `settings.astrbot_config_advice_seen`，
 * 跨会话只提示一次；标记写入失败时保持未提示，下次打开控制台再弹。
 */
export async function maybeShowAstrbotConfigAdvice() {
  if (astrbotAdvicePending) return;
  if (state.meta?.settings?.astrbot_config_advice_seen) return;
  astrbotAdvicePending = true;
  try {
    await askInfo(ASTRBOT_ADVICE, ASTRBOT_ADVICE_TITLE);
    await apiPost('settings/astrbot_advice_seen');
    if (state.meta?.settings) {
      state.meta.settings.astrbot_config_advice_seen = true;
    }
  } catch {
    // 标记没写进去：保持未提示，下次打开控制台再弹。
  } finally {
    astrbotAdvicePending = false;
  }
}
