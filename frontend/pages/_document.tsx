import Document, { Head, Html, Main, NextScript } from 'next/document'

// Added to prevent occasional Next build failures on the server
// complaining about missing "/_document". Safe for App Router projects.
export default class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}

