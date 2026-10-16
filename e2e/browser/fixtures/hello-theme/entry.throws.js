// @ts-check
// PRD-261 A-01: a hello-theme entry that fails while its module is evaluated. The host must
// report reason 'load' and keep Studio.
throw new Error('hello theme: deliberate load failure')
