# Embeddings on AWS Bedrock

Automatos embeds text (document chunks, memories, search queries, tool descriptions) to store and find it by meaning. With `EMBEDDING_PROVIDER=bedrock` it does that on Amazon Bedrock (`bedrock-runtime`, the `InvokeModel` call), with the same AWS credentials and region as Bedrock chat. Any edition may set it. Requirements: [PRD-278](../PRDS/PRD-278-RUN-ON-MANAGED-CLOUD-SERVICES.md), sections 5.5 and 5.6.

## Where the text goes

The text of every chunk, memory and query is sent to the Bedrock runtime endpoint of one AWS region:

- **On the pod's own role** (`AWS_USE_WORKLOAD_IDENTITY=true`, no credential found): the region is `AWS_REGION`, and the account is the one the pod's role belongs to. No key is stored anywhere.
- **With a stored AWS Bedrock credential** (type `aws_bedrock_api`: a `bedrock_api_key`, or an `aws_access_key_id` with its `aws_secret_access_key`): the account the key belongs to, in the credential's `aws_region`. With no region in the credential: `AWS_REGION` when `AWS_USE_WORKLOAD_IDENTITY` is on, `us-east-1` when it is off.
- **With `AWS_USE_WORKLOAD_IDENTITY` off and no credential found:** the AWS SDK's own credential chain (the environment's key pair, a profile, an instance role), in `us-east-1`.

## Which credential

The same functions Bedrock chat uses find it, in this order:

1. **The stored credential.** The one the embeddings tier names (the system setting `embeddings.credential_name_aws_bedrock`), when it names one. Otherwise exactly the lookup Bedrock chat's tier runs (`orchestrator_llm`): the credential it names (`orchestrator_llm.credential_name_aws_bedrock`), then the same name guesses, then the first active credential of type `aws_bedrock_api`. So with no mapping of its own, embeddings use the credential chat uses.
2. **The operator workspace's Bedrock key**, when there is one, in place of that credential; else, with no credential, the environment's key.
3. **The pod's own role**, with none of them and `AWS_USE_WORKLOAD_IDENTITY` on.

## Settings

Everything Bedrock embeddings need is read from the environment:

| Setting | Value |
|---|---|
| `EMBEDDING_PROVIDER` | `bedrock` |
| `EMBEDDING_MODEL` | a model from the table below; unset means `amazon.titan-embed-text-v2:0` |
| `VECTOR_STORE_DIMENSIONS` | the size the vectors are cut to: the model's size, from the table |
| `S3_VECTORS_DIMENSION` | the same size, when `S3_VECTORS_ENABLED` is on |
| `FIELD_EMBEDDING_DIM` | the same size, when the memory store (`QDRANT_URL`) is in use |

**Which setting wins.** With `EMBEDDING_PROVIDER=bedrock`, the environment wins over every stored setting: `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` over the stored `embeddings` `provider` and `model` (a new install seeds them with OpenRouter), and `VECTOR_STORE_DIMENSIONS` over the stored `dimensions`. Any other provider keeps the usual order: the stored setting first, the environment second, and a stored `dimensions` other than its seeded 2048 in place of `VECTOR_STORE_DIMENSIONS`.

**Why.** There is no supported way to change a stored setting while the API is not running: the settings page is the API's. A Bedrock install that refuses to start (below) is fixed in the environment alone.

The settings page (Settings → System Settings → Embeddings) shows the stored rows. While the environment chooses Bedrock, those rows are not the ones in use.

The three sizes default to 2,048. Every supported Bedrock model produces fewer, so **set them before the first start**.

## Supported models

| Model | Model ID | Dimensions | `VECTOR_STORE_DIMENSIONS`, `S3_VECTORS_DIMENSION`, `FIELD_EMBEDDING_DIM` |
|---|---|---|---|
| Amazon Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` | 1,024 (it also produces 512 or 256) | `1024` (or all three `512`, or all three `256`) |

- **One text per call.** A batch (a document's chunks) is embedded with at most the caller's `max_concurrent` calls in flight (5 by default). Every Bedrock embedding call in one API process shares a pool of 10 threads of its own, so however many documents ingest at once, at most 10 calls are in flight per process, and the rest of the API's blocking work never waits behind them.
- **Input.** Up to 8,192 tokens or 50,000 characters, AWS says. Text past 32,768 characters (8,192 tokens at about four characters a token) is cut before the call. An empty or whitespace-only text is refused before any call, in a batch too (other providers do not skip one either: OpenAI refuses an empty string). Ingestion drops chunks under 20 characters before it embeds, so a document never sends one.
- **Output.** The size that holds the configured one (1,024, 512 or 256) is asked for, normalised. A configured size that is not one of those three (768, say) gets 1,024 dimensions, cut to fit.
- **Time limits.** Each call: `EMBEDDING_CONNECT_TIMEOUT_S` to connect, `EMBEDDING_QUERY_TIMEOUT_S` to read, `EMBEDDING_MAX_RETRIES` retries (botocore's adaptive mode, which backs off on Bedrock's `ThrottlingException`), and at most `EMBEDDING_BATCH_TIMEOUT_S` for the call with its retries, counted from when a thread takes it.
- **Permissions.** The role or key needs `bedrock:InvokeModel` on the model.

Any other model ID is refused at start-up, with this list.

Sources: AWS, [Amazon Titan Text Embeddings models](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html), [Titan Embeddings request and response](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html), [InvokeModel](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html) and [API error codes](https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html).

## At start-up

Each API worker, before it listens, compares Bedrock's vectors with each index that is in use. The vectors are `VECTOR_STORE_DIMENSIONS` long, or the model's largest size when that is smaller. The indexes:

- `S3_VECTORS_DIMENSION`, when `S3_VECTORS_ENABLED` is on;
- `FIELD_EMBEDDING_DIM`, when the memory store answers at start (within 3 seconds; `QDRANT_URL` has a default, so an address alone does not count). It is asked only when its size disagrees.

The check reads only the environment for Bedrock, so every worker and replica sees the same, whether or not the boot leader has seeded the system settings yet. It makes no call to Bedrock.

- **An index in use that expects another size:** the install refuses to start. The message names each such setting and its value, the model's size and the vectors' length, the value to set in the environment, and which setting wins. For example: `EMBEDDING_PROVIDER is bedrock: the model 'amazon.titan-embed-text-v2:0' produces at most 1024 dimensions, so its vectors here are 1024 long (VECTOR_STORE_DIMENSIONS is 2048, more than the model's 1024). These indexes in use expect another size: S3_VECTORS_DIMENSION is 2048; FIELD_EMBEDDING_DIM is 2048. Set each of them to 1024 in the environment and start again. …`
- **A model ID that is not supported:** the install refuses to start, and the message lists the supported models.
- **Sizes that agree, or no index in use:** it starts.

Any other provider never refuses. When its model's size is known and smaller than an index in use, the API writes one warning that names the settings and both numbers, and starts. That check reads the stored settings; a replica that starts before the boot leader has seeded them reads the environment. A stored `dimensions` that is not a number never stops a start: it is logged once and `VECTOR_STORE_DIMENSIONS` is used.

## When Bedrock refuses

Bedrock's own message can name a role and an account, so it stays in the original exception; the error the rest of the API sees, and logs, carries Bedrock's error code and the model ID only.

- **A refusal that will not get better** without a change to the role, the key, the region or the model: `AccessDeniedException`, `ResourceNotFoundException`, `ValidationException` (also a model ID the region does not offer), `UnrecognizedClientException`, `InvalidClientTokenId`, `InvalidSignatureException`, `IncompleteSignature`, `NotAuthorized`, `ExpiredTokenException`. The batch stops: calls not yet made are cancelled, and ingestion does not try the texts again one at a time. Bedrock embeddings then report themselves degraded, with the code, in every part of the API process: readiness and the memory surfaces say memory and knowledge search are off, as they do with no provider. For 30 seconds further calls fail at once without a request; after that the next call goes to Bedrock, and the first that succeeds clears it.
- **Any other** (`ThrottlingException`, `ModelNotReadyException`, `ModelTimeoutException`, `ModelErrorException`, `ServiceUnavailableException`, `InternalServerException`, `ServiceQuotaExceededException`, a lost connection): the call raises after botocore's retries, and ingestion keeps its fallback of one text at a time, as it does for every provider. The provider stays ready.

## Changing the model on a running install

An index built at one size cannot take vectors of another. Switching to Bedrock on an install that already has documents, memories or an S3 vector index at 2,048 means building them again at the new size. That is not automated.

The embedding cache used by document ingestion is keyed by the provider, the model and the size, so after a switch no vector the old model made is ever served for the new one. The first ingestion after the switch, or after this change reaches an install, finds the cache cold and embeds again; nothing already stored is embedded again unless it is ingested again. Entries cached before stay in Redis, unread.
