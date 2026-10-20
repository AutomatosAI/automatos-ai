# hello-a11b — the SDK-chat theme (PRD-261 A-11b)

The fixture behind `specs/sdk-chat.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a11b`, base Studio, one frame slot. |
| `entry.js` | A frame that only wraps the page, and the theme's own page on `/chat`, drawn from the SDK alone. |
| `theme.css` | The stylesheet. Every rule is under `html.hello-a11b`; none names the approval card. |
| `presets/default.look.json` | The preset the manifest needs. |
| `chat-turns.json` | Recorded chat turns, in the stream format `lib/chat/hooks.ts` reads. Test data only. |

## The page

`entry.js` registers `/chat` and reads the chat through `@automatos/theme-sdk`:

- `useChatController()` gives the conversation on screen (a new chat here: the spec answers the
  conversations list with none), or null while it opens;
- `useChatController(conversation)`, in a child keyed by `conversation.key`, gives that conversation's
  messages, its status, and Send, Stop and Try again;
- each approval a reply raised is drawn by the host's `ToolApproval` card (`cards`, A-12a), which posts the
  decision; the theme only places it and passes the card's `answered` and `closed` on.

## The turns

The spec answers the app's own `POST /api/chat` inside the page, before the app's scripts run, so no model is
called and the stream can stay open:

| Turn | Frames | Proves |
|---|---|---|
| `answer` | the new chat's id, two pieces of text, finish | the stream renders and the turn ends |
| `partial` | the new chat's id, one piece of text, then held open (`hold: true`) | Stop keeps the words written so far (P260-FX-30) and cancels the turn |
| `again` | one piece of text, finish | Try again sends the last message again, on the same conversation |
| `approval` | the new chat's id, text, a `tool-data` frame with a `tool_approval`, finish | the card files under the reply, and Approve & run posts the decision |
