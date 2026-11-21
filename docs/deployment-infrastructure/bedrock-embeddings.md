# Embeddings on AWS Bedrock

Automatos embeds text (document chunks, memories, search queries, tool descriptions) to store and find it by meaning. With `EMBEDDING_PROVIDER=bedrock` it does that on Amazon Bedrock (`bedrock-runtime`, the `InvokeModel` call), with the same AWS credentials and region as Bedrock chat. Any edition may set it. Requirements: [PRD-278](../PRDS/PRD-278-RUN-ON-MANAGED-CLOUD-SERVICES.md), sections 5.5 and 5.6.

## Where the text goes

The text of every chunk, memory and query is sent to the Bedrock runtime endpoint of one AWS region:

- **On the pod's own role** (`AWS_USE_WORKLOAD_IDENTITY=true`, no key stored): the region is `AWS_REGION`, and the account is the one the pod's role belongs to. That is the deployment's own region and account; no key is stored anywhere.
- **With a stored AWS Bedrock credential** (type `aws_bedrock_api`: a `bedrock_api_key`, or an `aws_access_key_id` with its `aws_secret_access_key`): the account the key belongs to, in the credential's `aws_region`. With no region in the credential: `AWS_REGION` when `AWS_USE_WORKLOAD_IDENTITY` is on, `us-east-1` when it is off.
- **With `AWS_USE_WORKLOAD_IDENTITY` off and no credential stored:** the AWS SDK's own credential chain (the environment's key pair, a profile, an instance role), in `us-east-1`.

Credentials and region are found exactly as Bedrock chat finds them for a system setting: the stored credential, the operator workspace's key and the environment's key, then the pod's role. See [Credentials from the pod's role](../PRDS/PRD-278-RUN-ON-MANAGED-CLOUD-SERVICES.md) (section 5.1).

## Settings

| Setting | Value |
|---|---|
| `EMBEDDING_PROVIDER` | `bedrock` |
| `EMBEDDING_MODEL` | a model from the table below; unset means `amazon.titan-embed-text-v2:0` |
| `VECTOR_STORE_DIMENSIONS` | the model's size, from the table |
| `S3_VECTORS_DIMENSION` | the same size |
| `FIELD_EMBEDDING_DIM` | the same size |

`EMBEDDING_PROVIDER=bedrock` in the environment is read before the `embeddings` system settings (Settings → System Settings → Embeddings), which a new install seeds with OpenRouter. Any other value keeps the usual order: the system setting first, the environment second. The `dimensions` system setting, when it names a size other than its seeded 2048, takes the place of `VECTOR_STORE_DIMENSIONS`.

The three sizes default to 2,048. Every supported Bedrock model produces fewer, so **set all three before the first start**.

## Supported models

| Model | Model ID | Dimensions | `VECTOR_STORE_DIMENSIONS`, `S3_VECTORS_DIMENSION`, `FIELD_EMBEDDING_DIM` |
|---|---|---|---|
| Amazon Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` | 1,024 (it also produces 512 or 256) | `1024` (or all three `512`, or all three `256`) |

- **One text per call.** A batch (a document's chunks) is embedded with a bounded number of calls in flight (the caller's `max_concurrent`, 5 by default), never all at once.
- **Input.** Up to 8,192 tokens or 50,000 characters, AWS says. Text past 32,768 characters (8,192 tokens at about four characters a token) is cut before the call. An empty text is refused.
- **Output.** The size that holds the configured one (1,024, 512 or 256) is asked for, normalised. A configured size that is not one of those three (768, say) gets 1,024 dimensions, cut to fit.
- **Time limits.** Each call: `EMBEDDING_CONNECT_TIMEOUT_S` to connect, `EMBEDDING_QUERY_TIMEOUT_S` to read, `EMBEDDING_MAX_RETRIES` retries (botocore's adaptive mode, which backs off on Bedrock's `ThrottlingException`), and at most `EMBEDDING_BATCH_TIMEOUT_S` for the call with its retries.
- **Permissions.** The role or key needs `bedrock:InvokeModel` on the model.

Any other model ID is refused at start-up, with this list.

Sources: AWS, [Amazon Titan Text Embeddings models](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) and [Titan Embeddings request and response](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html).

## At start-up

With a real provider set, start-up compares the most dimensions the model produces with each size setting in use: `VECTOR_STORE_DIMENSIONS` (or the `dimensions` system setting), `S3_VECTORS_DIMENSION` when `S3_VECTORS_ENABLED` is on, and `FIELD_EMBEDDING_DIM` when `QDRANT_URL` is set.

- **A setting larger than the model:** the install refuses to start. The message names each such setting, its value, what the model produces and the value to set, for example: `The embedding model 'amazon.titan-embed-text-v2:0' (provider bedrock) produces at most 1024 dimensions, fewer than these settings expect: VECTOR_STORE_DIMENSIONS is 2048; S3_VECTORS_DIMENSION is 2048; FIELD_EMBEDDING_DIM is 2048. Set each to 1024 …`
- **A model ID that is not supported:** the install refuses to start, and the message lists the supported models.
- **A model that produces more than a setting:** it starts, and each vector is cut to fit.

The check reads the settings and nothing else: it makes no call to Bedrock. A wrong role, a missing permission or a region without the model shows at the first embedding, which fails with Bedrock's error code in the server log. Memory and knowledge search then report themselves off, as they do with no provider.

## Changing the model on a running install

An index built at one size cannot take vectors of another. Switching to Bedrock on an install that already has documents, memories or an S3 vector index at 2,048 means building them again at the new size. That is not automated.
