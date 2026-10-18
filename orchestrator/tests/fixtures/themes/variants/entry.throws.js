// PRD-261 A-06a: the hello-throws variant's dist/entry.js. It fails while its module is
// evaluated, as A-01's entry.throws.js does, so the host must report reason 'load' and keep
// Studio.
throw new Error('hello theme: deliberate load failure')
