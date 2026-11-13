# meetscribe → n8n → Slack

Use this when you cannot get a Slack bot token. With a token, `[publish] kind = "slack"`
posts directly without n8n (see `docs/local-setup.md`).

`meetscribe run` (or `meetscribe publish NOTE.md`) POSTs the finished note to
an n8n webhook. The flow posts the summary, decisions, and action items as a
Slack message and attaches the full Markdown note to it.

```
laptop: meetscribe ──POST JSON + X-Meetscribe-Token──▶ n8n webhook
                                                        ├─ Format for Slack (Code)
                                                        └─ Upload note to Slack (summary as the comment)
```

Files:

- `meetscribe-slack.workflow.json`: import into n8n.
- `format-slack.js`: the Code node's source, for review. Same code as in the workflow.

## Where the data goes

The summary and the full transcript leave the laptop. They pass through n8n
and end up in Slack. On n8n.cloud they are also stored in n8n's cloud. The
workflow turns off saving successful executions (`saveDataSuccessExecution:
none`). Failed runs are still saved so you can debug them; delete them after
fixing. Confirm that your team's policy allows meeting transcripts on n8n.cloud.

## Set up on the team n8n.cloud

1. **Import**: in n8n, open *Workflows → Import from file* and pick
   `meetscribe-slack.workflow.json`.
2. **Webhook auth**: on the *meetscribe webhook* node, create a *Header Auth*
   credential:
   - Name: `X-Meetscribe-Token`
   - Value: a random secret, e.g. `openssl rand -hex 32`

   Without it the public webhook URL accepts posts from anyone.
3. **Slack**: on *Upload note to Slack*, pick the team's existing Slack
   credential. The bot needs `files:write` and `chat:write`, and it must be a
   member of the channel (`/invite @<bot>`). Replace `SET_CHANNEL_ID` with the
   channel ID: *channel details → bottom of the About tab*.

   The Slack node's fields differ between n8n versions. After the import, open
   the node and check that it is set to *File → Upload*, the binary field is
   `note`, and the channel, file name, title, and initial comment are filled in.
4. **Activate** the workflow and copy the webhook node's **Production URL**.
5. **On the laptop**:

   ```bash
   # ~/.config/meetscribe/config.toml
   [publish]
   url = "https://<team>.app.n8n.cloud/webhook/meetscribe"
   auto = true
   ```

   Keep the token out of the config file and out of the repo:

   ```bash
   install -m 600 /dev/null ~/.config/meetscribe/publish-token
   $EDITOR ~/.config/meetscribe/publish-token        # paste the secret
   echo 'export MEETSCRIBE_PUBLISH_TOKEN="$(cat ~/.config/meetscribe/publish-token)"' >> ~/.bashrc
   ```

6. **Test** with an existing note:

   ```bash
   uv run --no-sync meetscribe publish ~/Meetings/meeting.md
   ```

## Self-hosted n8n instead

Import the same workflow into n8n on the home Linux host. Bind n8n to the
WireGuard address only, and set `url = "http://WIREGUARD_HOST:5678/webhook/meetscribe"`.
Slack is only called outbound, so the home host needs no public ingress.

## When the webhook is down

A note that cannot be delivered is saved in `~/.local/state/meetscribe/outbox/`
(mode 0600; override with `MEETSCRIBE_OUTBOX`). `run` only warns; it does not
fail. To resend:

```bash
uv run --no-sync meetscribe publish --flush
```

The webhook answers only after Slack accepts the upload. A Slack error therefore
also leaves the note queued instead of losing it.
