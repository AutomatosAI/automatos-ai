/**
 * F237 (PRD-251B): the origins the browser may load media from, for the Content-Security-Policy
 * in next.config.js. Socials thumbnails and a post's media are presigned links to object storage
 * (a MinIO host locally, AWS S3 in SaaS), so img-src and media-src must allow its public origin.
 *
 * NEXT_PUBLIC_MEDIA_ORIGINS is read when the app is BUILT (next.config.js headers are baked into
 * the standalone build): one or more origins separated by commas or spaces, such as
 * "http://localhost:9000" (the local edition's MinIO, docker-compose.yml) or
 * "https://*.amazonaws.com" (the Dockerfile's default: the hosted product). Anything that is not a
 * bare scheme://host[:port] origin is left out, so a stray path or keyword never widens the policy.
 *
 * CommonJS on purpose: next.config.js requires it, and so does its test.
 */
'use strict'

const ORIGIN = /^https?:\/\/(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)*(:\d{1,5})?$/i

function mediaOrigins(raw) {
  return String(raw || '')
    .split(/[\s,]+/)
    .map((token) => token.trim().replace(/\/+$/, ''))
    .filter((token) => ORIGIN.test(token))
}

module.exports = { mediaOrigins }
