// @vitest-environment node
/**
 * Theme voice, web quick-ask — toSpeakable (ported with automatos-go's speakable.test.ts, PR #15): Auto's markdown
 * becomes words a voice can say; code, tables and URLs point to the chat. Every case is also checked for
 * idempotence and for streaming character by character through the chunker.
 */
import { describe, expect, it } from 'vitest'

import { sentenceChunker } from '@/lib/voice/quick-ask/sentences'
import { toSpeakable } from '@/lib/voice/quick-ask/speakable'

const CASES: readonly (readonly [string, string, string])[] = [
  ['bold, italics and inline code', 'This is **bold**, _italic_, *also italic* and `npm test`.',
    'This is bold, italic, also italic and npm test.'],
  ['underscore bold and strikethrough', '__Note__: ~~old~~ new.', 'Note: old new.'],
  ['an identifier keeps its underscores', 'Edit file_name.ts and my_var.', 'Edit file_name.ts and my_var.'],
  ['a heading is a sentence', '# Your day\nThree things are due.', 'Your day. Three things are due.'],
  ['closing hashes', '## Done ##\nNext up.', 'Done. Next up.'],
  ['a heading underlined', 'Summary\n===\nAll good.', 'Summary. All good.'],
  ['bullets, each item a sentence', 'Your day:\n- Standup at 9\n- Lunch with Sam\n* Review the PR\n• Call Ana,\n',
    'Your day: Standup at 9. Lunch with Sam. Review the PR. Call Ana.'],
  ['a numbered list, each item a sentence', '1. First\n2) Second.\n3. Third:\n', 'First. Second. Third.'],
  ['a task list', '- [ ] Write tests\n- [x] Ship it\n', 'Write tests. Ship it.'],
  ['an item still streaming gets no full stop yet', '- Apples\n- Pea', 'Apples. Pea'],
  ['a link says its label', 'Read [the docs](https://docs.automatos.app/start) first.', 'Read the docs first.'],
  ['an image says its alt text', 'Look: ![Weekly chart](https://x.test/chart.png)', 'Look: Weekly chart'],
  ['a bare URL is a link', 'Open https://automatos.app/settings?tab=voice. Then sign in.', 'Open a link. Then sign in.'],
  ['www and angle-bracket links', 'Try www.example.com or <https://x.test/a>.', 'Try a link or a link.'],
  ['a code fence', 'Run this:\n```bash\nnpm test -- **all**\n```\nThen check the output.',
    "Run this. There's code in the chat. Then check the output."],
  ['an unclosed code fence is code to the end', 'Here it is\n```ts\nconst a = 1;\n# not a heading',
    "Here it is. There's code in the chat."],
  ['a table', 'Results:\n| Name | Score |\n|---|---|\n| Ana | 3 |\n| Sam | 5 |\nThat is all.',
    "Results. There's a table in the chat. That is all."],
  ['a horizontal rule', 'Above.\n\n---\n\nBelow.', 'Above. Below.'],
  ['a quote', '> Quoted **text**\n> - and an item\n', 'Quoted text and an item.'],
  ['whitespace collapsed', 'Hello  \n\n \t world.  ', 'Hello world.'],
  ['a pair still streaming', 'This is **very', 'This is very'],
  ['a lone star is left alone', '5 * 3 is 15.', '5 * 3 is 15.'],
  ['nothing but markdown', '**\n---\n', ''],
  ['empty', '', ''],
]

describe('toSpeakable', () => {
  it.each(CASES)('%s', (_name, markdown, spoken) => {
    expect(toSpeakable(markdown)).toBe(spoken)
  })

  it.each(CASES)('is idempotent: %s', (_name, markdown) => {
    const once = toSpeakable(markdown)

    expect(toSpeakable(once)).toBe(once)
  })

  it('leaves no markdown character or URL for the voice to read', () => {
    const spoken = CASES.map(([, markdown]) => toSpeakable(markdown))
      .filter((text) => text !== '5 * 3 is 15.')
      .join(' ')

    expect(spoken).not.toMatch(/[*`#|[\]~]|https?:|www\./)
  })

  it.each(CASES)('streamed character by character, says the final text once: %s', (_name, markdown) => {
    const chunker = sentenceChunker()
    const spoken = Array.from(markdown).flatMap((_char, index) =>
      chunker.push(toSpeakable(markdown.slice(0, index + 1))),
    )

    expect([...spoken, ...chunker.flush()].join(' ')).toBe(toSpeakable(markdown))
  })
})
