                                               Table "public.learning_progress_tracking"
         Column          |            Type             | Collation | Nullable |                        Default                         
-------------------------+-----------------------------+-----------+----------+--------------------------------------------------------
 id                      | integer                     |           | not null | nextval('learning_progress_tracking_id_seq'::regclass)
 agent_id                | integer                     |           |          | 
 knowledge_items         | integer                     |           |          | 0
 memory_consolidations   | integer                     |           |          | 0
 performance_improvement | double precision            |           |          | 0.0
 knowledge_transfers     | integer                     |           |          | 0
 recorded_at             | timestamp without time zone |           |          | now()
Indexes:
    "learning_progress_tracking_pkey" PRIMARY KEY, btree (id)
    "idx_learning_progress_agent_time" btree (agent_id, recorded_at)
Foreign-key constraints:
    "learning_progress_tracking_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id)

                                                                         Table "public.learning_progress_tracking"
         Column          |            Type             | Collation | Nullable |                        Default                         | Storage | Compression | Stats target | Description 
-------------------------+-----------------------------+-----------+----------+--------------------------------------------------------+---------+-------------+--------------+-------------
 id                      | integer                     |           | not null | nextval('learning_progress_tracking_id_seq'::regclass) | plain   |             |              | 
 agent_id                | integer                     |           |          |                                                        | plain   |             |              | 
 knowledge_items         | integer                     |           |          | 0                                                      | plain   |             |              | 
 memory_consolidations   | integer                     |           |          | 0                                                      | plain   |             |              | 
 performance_improvement | double precision            |           |          | 0.0                                                    | plain   |             |              | 
 knowledge_transfers     | integer                     |           |          | 0                                                      | plain   |             |              | 
 recorded_at             | timestamp without time zone |           |          | now()                                                  | plain   |             |              | 
Indexes:
    "learning_progress_tracking_pkey" PRIMARY KEY, btree (id)
    "idx_learning_progress_agent_time" btree (agent_id, recorded_at)
Foreign-key constraints:
    "learning_progress_tracking_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id)
Access method: heap

                                                                   Table "public.memory_items"
         Column         |            Type             | Collation | Nullable |           Default            | Storage  | Compression | Stats target | Description 
------------------------+-----------------------------+-----------+----------+------------------------------+----------+-------------+--------------+-------------
 id                     | uuid                        |           | not null | uuid_generate_v4()           | plain    |             |              | 
 agent_id               | integer                     |           |          |                              | plain    |             |              | 
 content                | character varying           |           | not null |                              | extended |             |              | 
 memory_type            | character varying(100)      |           | not null |                              | extended |             |              | 
 memory_level           | character varying(50)       |           |          | 'working'::character varying | extended |             |              | 
 importance             | double precision            |           |          | 0.5                          | plain    |             |              | 
 embedding              | vector(384)                 |           |          |                              | external |             |              | 
 access_count           | integer                     |           |          | 0                            | plain    |             |              | 
 last_access            | timestamp without time zone |           |          | now()                        | plain    |             |              | 
 decay_rate             | double precision            |           |          | 0.1                          | plain    |             |              | 
 associations           | character varying[]         |           |          | '{}'::character varying[]    | extended |             |              | 
 metadata               | json                        |           |          | '{}'::json                   | extended |             |              | 
 created_at             | timestamp without time zone |           |          | now()                        | plain    |             |              | 
 success_rate           | double precision            |           |          | 0.0                          | plain    |             |              | 
 usage_in_solutions     | integer                     |           |          | 0                            | plain    |             |              | 
 average_retrieval_time | double precision            |           |          | 0.0                          | plain    |             |              | 
Indexes:
    "memory_items_pkey" PRIMARY KEY, btree (id)
    "idx_memory_items_agent" btree (agent_id)
    "idx_memory_items_agent_id" btree (agent_id)
    "idx_memory_items_agent_memory_level" btree (agent_id, memory_level)
    "idx_memory_items_created_at" btree (created_at)
    "idx_memory_items_embedding" ivfflat (embedding vector_cosine_ops)
    "idx_memory_items_importance" btree (importance DESC)
    "idx_memory_items_memory_level" btree (memory_level)
    "idx_memory_items_memory_type" btree (memory_type)
Foreign-key constraints:
    "memory_items_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id)
Access method: heap

                                                       Table "public.knowledge_nodes"
   Column    |            Type             | Collation | Nullable |      Default      | Storage  | Compression | Stats target | Description 
-------------+-----------------------------+-----------+----------+-------------------+----------+-------------+--------------+-------------
 id          | uuid                        |           | not null | gen_random_uuid() | plain    |             |              | 
 agent_id    | integer                     |           |          |                   | plain    |             |              | 
 concept     | character varying(255)      |           |          |                   | extended |             |              | 
 description | text                        |           |          |                   | extended |             |              | 
 node_type   | character varying(50)       |           |          |                   | extended |             |              | 
 embedding   | vector(384)                 |           |          |                   | external |             |              | 
 importance  | double precision            |           |          |                   | plain    |             |              | 
 confidence  | double precision            |           |          |                   | plain    |             |              | 
 metadata    | jsonb                       |           |          |                   | extended |             |              | 
 created_at  | timestamp without time zone |           |          | now()             | plain    |             |              | 
Indexes:
    "knowledge_nodes_pkey" PRIMARY KEY, btree (id)
    "idx_knowledge_nodes_agent_id" btree (agent_id)
    "idx_knowledge_nodes_concept" btree (concept)
    "idx_knowledge_nodes_embedding" ivfflat (embedding vector_cosine_ops)
    "idx_knowledge_nodes_node_type" btree (node_type)
Foreign-key constraints:
    "knowledge_nodes_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id)
Referenced by:
    TABLE "knowledge_edges" CONSTRAINT "knowledge_edges_from_node_id_fkey" FOREIGN KEY (from_node_id) REFERENCES knowledge_nodes(id)
    TABLE "knowledge_edges" CONSTRAINT "knowledge_edges_to_node_id_fkey" FOREIGN KEY (to_node_id) REFERENCES knowledge_nodes(id)
Access method: heap

