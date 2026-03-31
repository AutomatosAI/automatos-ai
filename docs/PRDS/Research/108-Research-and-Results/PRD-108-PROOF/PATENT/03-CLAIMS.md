# Claims

## Independent Claims

### Claim 1 — Multi-Agent Shared Semantic Field Coordination

A computer-implemented method for coordinating a plurality of large language model (LLM) agents executing tasks within a mission, the method comprising:

(a) creating, responsive to initiation of a mission comprising a plurality of agent tasks, a shared vector embedding collection in a vector database, the collection being scoped to said mission;

(b) for each agent task that completes during said mission, automatically injecting the task output into the shared collection as a vector point comprising a high-dimensional semantic embedding of the output content, an agent identifier, a strength value, a timestamp, and a content hash;

(c) during execution of a subsequent agent task, receiving from an executing agent a natural language query directed to the shared collection;

(d) computing, for each candidate vector point in the collection, a resonance score based on (i) the squared cosine similarity between the query embedding and the candidate embedding, (ii) a temporal decay factor computed from the elapsed time since the candidate was last accessed, and (iii) an access-based reinforcement factor derived from the candidate's retrieval count;

(e) returning to the executing agent the top-k candidate vector points ranked by resonance score; and

(f) destroying the shared collection responsive to the mission reaching a terminal state.

### Claim 2 — Resonance-Based Retrieval Scoring

A computer-implemented method for ranking query results in a shared multi-agent knowledge field, comprising:

(a) receiving a query embedding from an agent;

(b) retrieving candidate patterns from a vector database by approximate nearest neighbor search;

(c) for each candidate pattern, computing a resonance score according to the formula:

R = cos^2(theta) x S_0 x e^(-lambda x t) x min(1 + alpha x n, C)

where:
- theta is the angle between the query embedding and the candidate embedding,
- S_0 is the stored strength of the candidate pattern,
- lambda is a configurable decay rate,
- t is the elapsed time in hours since the candidate was last accessed,
- alpha is a per-access reinforcement increment,
- n is the candidate's access count, and
- C is a reinforcement cap;

(d) filtering candidates whose temporally-decayed strength falls below a configurable archival threshold; and

(e) returning the remaining candidates ranked by resonance score.

### Claim 3 — Hebbian Co-Access Reinforcement in Agent Knowledge Fields

A computer-implemented method for strengthening associative relationships between knowledge patterns in a shared agent field, comprising:

(a) executing a semantic query against the shared field on behalf of an agent;

(b) retrieving a result set of k patterns;

(c) for each pattern in the result set, incrementing an access count and updating a last-accessed timestamp;

(d) for each pattern in the result set, computing a co-access strength bonus based on the number of co-retrieved patterns:

S_new = min(S_current x (1 + beta x (k - 1)), S_current x C)

where beta is a co-access bonus coefficient and C is a reinforcement cap; and

(e) persisting the updated strength value to the vector database, whereby patterns that are frequently retrieved together in response to agent queries become mutually strengthened.

## Dependent Claims

### Claim 4

The method of Claim 1, further comprising, prior to step (b), seeding the shared collection with the mission goal as an initial pattern at a predetermined strength value, such that agent queries can discover the mission objective by semantic similarity.

### Claim 5

The method of Claim 1, wherein step (b) further comprises:
- computing a SHA-256 hash of the content to be injected;
- querying the collection for an existing pattern with a matching content hash;
- if a matching pattern exists, reinforcing the existing pattern by incrementing its access count and updating its last-accessed timestamp instead of creating a new vector point.

### Claim 6

The method of Claim 1, wherein the shared collection is implemented behind an abstract interface comprising create_context, inject, query, and destroy_context methods, enabling substitution of the vector field implementation with an alternative coordination backend without modification to mission orchestration code.

### Claim 7

The method of Claim 2, wherein the retrieval step (b) fetches a multiple of the requested top-k results from the vector database (over-fetching), applies the archival filtering of step (d), and returns the final top-k from the remaining candidates, compensating for candidates eliminated by the archival threshold.

### Claim 8

The method of Claim 1, further comprising computing a field stability metric:

stability = avg_strength x w_1 + organization x w_2

where avg_strength is the mean temporally-decayed strength of all patterns, organization is computed as max(0, 1 - (standard_deviation / mean)) of pattern strengths, and w_1 and w_2 are configurable weights, said metric indicating the degree to which agents are building shared understanding versus the field fragmenting.

### Claim 9

The method of Claim 1, wherein step (b) scales the initial strength of each injected pattern by a configurable boundary permeability coefficient between 0.0 and 1.0, controlling the rate at which new knowledge enters the field.

### Claim 10

The method of Claim 1, wherein the plurality of agents interact with the shared collection through agent-callable platform tools comprising:
- a query tool that accepts a natural language query and returns resonance-ranked patterns;
- an inject tool that accepts a semantic key and value and creates a new pattern with deduplication; and
- a stability tool that returns the field's current convergence metrics;

said tools being registered through a platform tool discovery pipeline and available to any agent assigned to a mission with an active field.
