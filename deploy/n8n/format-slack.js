// n8n Code node ("Run Once for Each Item"): meetscribe payload -> Slack text + file.
// Kept as a separate file so it can be reviewed; the workflow JSON embeds a copy.
const p = $json.body ?? $json;
if (p.source !== 'meetscribe') {
  throw new Error('Not a meetscribe payload');
}
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const list = (items) => items.map((s) => `• ${esc(s)}`).join('\n');
const s = p.summary ?? {};
const lines = [`*${esc(p.title)}*  ${esc(p.started_at)}`];
if (p.participants?.length) lines.push(`參與者：${p.participants.map(esc).join('、')}`);
if (s.overview) lines.push('', esc(s.overview));
if (s.key_points?.length) lines.push('', '*重點*', list(s.key_points));
if (s.decisions?.length) lines.push('', '*決議*', list(s.decisions));
if (s.action_items?.length) {
  lines.push('', '*待辦*');
  for (const a of s.action_items) {
    const owner = a.owner ? `（${esc(a.owner)}）` : '';
    const due = a.due ? ` — ${esc(a.due)}` : '';
    lines.push(`• ${esc(a.task)}${owner}${due}`);
  }
}
const markdown = p.note?.markdown ?? '';
return {
  json: { text: lines.join('\n'), title: p.title, fileName: p.note?.filename ?? 'meeting.md' },
  binary: {
    note: {
      data: Buffer.from(markdown, 'utf8').toString('base64'),
      mimeType: 'text/markdown',
      fileName: p.note?.filename ?? 'meeting.md',
      fileExtension: 'md',
    },
  },
};
