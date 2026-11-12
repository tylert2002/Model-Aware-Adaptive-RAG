# Restore the exact HotpotQA Elasticsearch snapshot

This package restores the `hotpotqa` index containing 5,233,329 documents. The
snapshot was created by Elasticsearch OSS 7.10.2 and should be restored with
the same version.

## Files to transfer

Transfer these three files together:

- `hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz`
- `hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz.sha256`
- `hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.manifest.json`

## 1. Verify the transfer

From the directory containing the files:

```bash
shasum -a 256 -c hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz.sha256
tar -tzf hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz >/dev/null
```

The expected SHA-256 is:

```text
48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54
```

## 2. Select the Elasticsearch 7.10.2 image

Use the architecture-appropriate OSS image:

```bash
# Apple Silicon / ARM64
export ES_IMAGE=docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2-arm64

# Intel/AMD x86_64 instead
# export ES_IMAGE=docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2
```

The snapshot format is portable across CPU architectures; the Docker image is
not. Do not use a newer Elasticsearch major version for this verification.

## 3. Import the repository into a Docker volume

The following creates new volumes and does not touch any existing Elasticsearch
data. Run it from the transfer directory.

```bash
docker volume create hotpotqa-restore-snapshots
docker volume create hotpotqa-restore-data

docker run --rm --user 0:0 --entrypoint sh \
  --mount type=bind,source="$PWD",target=/transfer,readonly \
  --mount source=hotpotqa-restore-snapshots,target=/mnt/es_snapshots \
  "$ES_IMAGE" -c \
  'tar -xzf /transfer/hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz -C /mnt/es_snapshots && chown -R 1000:0 /mnt/es_snapshots'
```

Do not copy only selected files from `hotpotqa_repository`; an Elasticsearch
filesystem repository must be transferred as a complete directory.

## 4. Start an isolated Elasticsearch 7.10.2 node

The dedicated empty data volume avoids deleting or overwriting any existing
index. Change the host port if `9200` is already occupied.

```bash
docker run -d --name hotpotqa-es-restore --restart no \
  -p 127.0.0.1:9200:9200 \
  --env 'discovery.type=single-node' \
  --env 'ES_JAVA_OPTS=-Xms2g -Xmx2g' \
  --env 'path.repo=/mnt/es_snapshots' \
  --mount source=hotpotqa-restore-data,target=/usr/share/elasticsearch/data \
  --mount source=hotpotqa-restore-snapshots,target=/mnt/es_snapshots,readonly \
  "$ES_IMAGE"

curl --retry 30 --retry-delay 2 --retry-connrefused \
  'http://127.0.0.1:9200/_cluster/health?wait_for_status=yellow&timeout=60s&pretty'
```

Yellow health is expected on a one-node cluster because the restored index has
one replica configured and the replica cannot be assigned to the same node as
its primary.

## 5. Register the transferred repository read-only

```bash
curl -sS -X PUT -H 'Content-Type: application/json' \
  'http://127.0.0.1:9200/_snapshot/hotpotqa_fs_repository?verify=true&pretty' \
  -d '{
    "type": "fs",
    "settings": {
      "location": "/mnt/es_snapshots/hotpotqa_repository",
      "readonly": true
    }
  }'

curl -sS \
  'http://127.0.0.1:9200/_snapshot/hotpotqa_fs_repository/hotpotqa_20261110t0630z?pretty'
```

The snapshot response must report `state: SUCCESS`, index `hotpotqa`, one
successful shard, and zero failed shards.

## 6. Restore

The target cluster must not already contain an open index named `hotpotqa`.
Using the dedicated empty data volume above is the safest option and requires
no deletion.

```bash
curl -sS -X POST -H 'Content-Type: application/json' \
  'http://127.0.0.1:9200/_snapshot/hotpotqa_fs_repository/hotpotqa_20261110t0630z/_restore?wait_for_completion=true&pretty' \
  -d '{
    "indices": "hotpotqa",
    "ignore_unavailable": false,
    "include_global_state": false,
    "include_aliases": true
  }'
```

If an existing cluster already has `hotpotqa`, do not delete it. Restore under
a different name with `rename_pattern` and `rename_replacement`, or use the
fresh dedicated data volume shown above.

## 7. Verify the restored index

### Snapshot and recovery state

```bash
curl -sS 'http://127.0.0.1:9200/_cat/recovery/hotpotqa?v'
curl -sS 'http://127.0.0.1:9200/_cluster/health/hotpotqa?pretty'
curl -sS 'http://127.0.0.1:9200/_cat/indices/hotpotqa?v&bytes=b'
```

### Document count

```bash
curl -sS 'http://127.0.0.1:9200/hotpotqa/_count?pretty'
```

Expected: `5233329`.

### Mapping

```bash
curl -sS 'http://127.0.0.1:9200/hotpotqa/_mapping?pretty'
```

Expected properties:

- `id`: text with a keyword subfield
- `is_abstract`: boolean
- `paragraph_index`: integer
- `paragraph_text`: text using the English analyzer
- `title`: text using the English analyzer
- `url`: text using the English analyzer

### Settings

```bash
curl -sS 'http://127.0.0.1:9200/hotpotqa/_settings?pretty'
```

Expected: one primary shard, one replica, creation version `7100299`, and index
UUID `8N5vCmxTSx6zcyJpkFlCgg`.

### Exact sample documents

```bash
curl -sS 'http://127.0.0.1:9200/hotpotqa/_doc/12001?pretty'
curl -sS 'http://127.0.0.1:9200/hotpotqa/_doc/12002?pretty'
```

Expected titles are `Luxembourg Football Federation` and `US Lacrosse`,
respectively; both have `paragraph_index: 0` and `is_abstract: true`.

### Representative BM25 retrieval

```bash
curl -sS -X POST -H 'Content-Type: application/json' \
  'http://127.0.0.1:9200/hotpotqa/_search?pretty' \
  -d '{
    "size": 3,
    "_source": ["title", "paragraph_index", "paragraph_text", "url", "is_abstract"],
    "query": {
      "multi_match": {
        "query": "Luxembourg Football Federation",
        "fields": ["title^1.25", "paragraph_text"]
      }
    }
  }'
```

Confirm that the response has one successful shard and these top three IDs and
titles in order:

1. `4795782` — `List of football clubs in Luxembourg`
2. `12001` — `Luxembourg Football Federation`
3. `1247980` — `List of Luxembourg international footballers`

The source run produced scores `24.022036`, `23.793234`, and `23.246744` under
Elasticsearch 7.10.2. Small score-display differences are less important than
the restored document count, mapping, settings, and hit identities/order.
