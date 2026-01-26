import Document, { Head, Html, Main, NextScript } from 'next/document'

// Added to prevent occasional Next build failures on the server
// complaining about missing "/_document". Safe for App Router projects.
// NOTE: This is for Pages Router compatibility only. App Router uses app/layout.tsx
export default class MyDocument extends Document {
  render() {
    return (
      <Html lang="en" suppressHydrationWarning>
        <Head />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}


