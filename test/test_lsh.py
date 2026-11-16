import pickle
import unittest
from unittest.mock import patch

import mockredis
import numpy as np

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


def first_band_hash(data):
    return data


def second_band_hash(data):
    return data[::-1]


class TestMinHashLSH(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        for _l in range(2, 128 + 1, 16):
            lsh = MinHashLSH(num_perm=128)
            m = MinHash()
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_unpacking(self):
        for b in range(2, 1024 + 1):
            lsh = MinHashLSH(num_perm=b * 4, params=(b, 4))
            m = MinHash(num_perm=b * 4)
            m.update(b"abcdefg")
            m.update(b"1234567")
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.add_to_query_buffer(m1)
        result = lsh.collect_query_buffer()
        self.assertTrue("a" in result)
        lsh.add_to_query_buffer(m2)
        result = lsh.collect_query_buffer()
        self.assertTrue("b" in result)
        m3 = MinHash(18)
        self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_query_buffer_matches_query_candidates(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=32)
        docs = []
        for tokens in ([b"a", b"b", b"c"], [b"a", b"b", b"d"], [b"x", b"y", b"z"]):
            minhash = MinHash(num_perm=32)
            for token in tokens:
                minhash.update(token)
            docs.append(minhash)
        for key, minhash in enumerate(docs):
            lsh.insert(key, minhash)

        lsh.add_to_query_buffer(docs[0])
        buffered_result = set(lsh.collect_query_buffer())
        direct_result = set(lsh.query(docs[0]))

        self.assertEqual(buffered_result, direct_result)
        self.assertEqual(buffered_result, {0, 1})

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))
        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)

    def test_insert_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            for t in lsh.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("a") in items)
                self.assertTrue(pickle.dumps("b") in items)
            self.assertTrue("a" in lsh)
            self.assertTrue("b" in lsh)
            for i, H in enumerate(lsh.keys[pickle.dumps("a")]):
                self.assertTrue(pickle.dumps("a") in lsh.hashtables[i][H])

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            result = lsh.query(m1)
            self.assertTrue("a" in result)
            result = lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.query, m3)

    def test_query_buffer_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.query(m1)
            lsh.add_to_query_buffer(m1)
            result = lsh.collect_query_buffer()
            self.assertTrue("a" in result)
            lsh.add_to_query_buffer(m2)
            result = lsh.collect_query_buffer()
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.add_to_query_buffer, m3)

    def test_insertion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        data = [("a", m1), ("b", m2)]
        with lsh.insertion_session() as session:
            for key, minhash in data:
                session.insert(key, minhash)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

    def test_deletion_session(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        m3 = MinHash(16)
        m3.update(b"c")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.insert("c", m3)

        keys_to_delete = ["a", "b"]
        with lsh.deletion_session() as session:
            for key in keys_to_delete:
                session.remove(key)

        # Verify deletions
        self.assertTrue("a" not in lsh.keys)
        self.assertTrue("b" not in lsh.keys)
        self.assertTrue("c" in lsh.keys)

        for table in lsh.hashtables:
            for H in table:
                self.assertTrue("a" not in table[H])
                self.assertTrue("b" not in table[H])

    def test_get_counts(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        counts = lsh.get_counts()
        self.assertEqual(len(counts), lsh.b)
        for table in counts:
            self.assertEqual(sum(table.values()), 2)

    def test_merge(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16)
        m1.update(b"a")
        m2 = MinHash(16)
        m2.update(b"b")
        lsh1.insert("a", m1)
        lsh1.insert("b", m2)

        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        m3 = MinHash(16)
        m3.update(b"c")
        m4 = MinHash(16)
        m4.update(b"d")
        lsh2.insert("c", m1)
        lsh2.insert("d", m2)

        lsh1.merge(lsh2)
        for t in lsh1.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("c" in items)
            self.assertTrue("d" in items)
        self.assertTrue("a" in lsh1)
        self.assertTrue("b" in lsh1)
        self.assertTrue("c" in lsh1)
        self.assertTrue("d" in lsh1)
        for i, H in enumerate(lsh1.keys["c"]):
            self.assertTrue("c" in lsh1.hashtables[i][H])

        self.assertTrue(lsh1.merge, lsh2)
        self.assertRaises(ValueError, lsh1.merge, lsh2, check_overlap=True)

        m5 = MinHash(16)
        m5.update(b"e")
        lsh3 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh3.insert("a", m5)

        self.assertRaises(ValueError, lsh1.merge, lsh3, check_overlap=True)

        lsh1.merge(lsh3)

        m6 = MinHash(16)
        m6.update(b"e")
        lsh4 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh4.insert("a", m6)

        lsh1.merge(lsh4, check_overlap=False)

    def test_merge_rejects_different_band_hash_functions(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        destination = MinHashLSH(threshold=0.5, num_perm=16, hashfunc=first_band_hash)
        source = MinHashLSH(threshold=0.5, num_perm=16, hashfunc=second_band_hash)
        source.insert("source", minhash)

        with self.assertRaisesRegex(ValueError, "different initialization parameters"):
            destination.merge(source)
        self.assertNotIn("source", destination)

    def test_merge_rejects_different_key_serialization(self):
        minhash = MinHash(16)
        minhash.update(b"value")
        destination = MinHashLSH(threshold=0.5, num_perm=16, prepickle=False)
        source = MinHashLSH(threshold=0.5, num_perm=16, prepickle=True)
        source.insert("source", minhash)

        with self.assertRaisesRegex(ValueError, "different initialization parameters"):
            destination.merge(source)
        self.assertNotIn("source", destination)

    def test_merge_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh1 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh2 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )

            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            lsh1.insert("a", m1)
            lsh1.insert("b", m2)

            m3 = MinHash(16)
            m3.update(b"c")
            m4 = MinHash(16)
            m4.update(b"d")
            lsh2.insert("c", m3)
            lsh2.insert("d", m4)

            lsh1.merge(lsh2)
            for t in lsh1.hashtables:
                self.assertTrue(len(t) >= 1)
                items = []
                for H in t:
                    items.extend(t[H])
                self.assertTrue(pickle.dumps("c") in items)
                self.assertTrue(pickle.dumps("d") in items)
            self.assertTrue("a" in lsh1)
            self.assertTrue("b" in lsh1)
            self.assertTrue("c" in lsh1)
            self.assertTrue("d" in lsh1)
            for i, H in enumerate(lsh1.keys[pickle.dumps("c")]):
                self.assertTrue(pickle.dumps("c") in lsh1.hashtables[i][H])

            self.assertTrue(lsh1.merge, lsh2)
            self.assertRaises(ValueError, lsh1.merge, lsh2, check_overlap=True)

            m5 = MinHash(16)
            m5.update(b"e")
            lsh3 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh3.insert("a", m5)

            self.assertRaises(ValueError, lsh1.merge, lsh3, check_overlap=True)

            m6 = MinHash(16)
            m6.update(b"e")
            lsh4 = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            lsh4.insert("a", m6)

            lsh1.merge(lsh4, check_overlap=False)

    def test_redis_deletion_session(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )

            # The library's RedisBuffer is not completely compatible with mockredis.
            # In order to account for this, we:
            # 1. Replace the buffer with mockredis's own simple pipeline, which can execute.
            # 2. Patch __init__ to prevent the buffer flush from resetting the storage object
            #    and wiping mock data.
            storage_objects = [lsh.keys, *lsh.hashtables]
            for storage in storage_objects:
                storage._buffer = storage._redis.pipeline()

            m1 = MinHash(16)
            m1.update(b"a")
            m2 = MinHash(16)
            m2.update(b"b")
            m3 = MinHash(16)
            m3.update(b"c")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            lsh.insert("c", m3)

            keys_to_delete = ["a", "b"]

            with (
                patch.object(type(lsh.keys), "__init__", lambda self, config, name: None),
                lsh.deletion_session() as session,
            ):
                for key in keys_to_delete:
                    session.remove(key)

            # Verify deletions
            self.assertTrue("a" not in lsh)
            self.assertTrue("b" not in lsh)
            self.assertTrue("c" in lsh)

            # Verify underlying storage
            for table in lsh.hashtables:
                for H in table:
                    items = table[H]
                    self.assertTrue("a".encode("utf8") not in items)
                    self.assertTrue("b".encode("utf8") not in items)

    # ------------------------------------------------------------------
    # Correction batches
    # ------------------------------------------------------------------
    def _make_minhash(self, token, num_perm=16):
        m = MinHash(num_perm)
        m.update(token)
        return m

    def _assert_index_internals_consistent(self, lsh):
        """Every key-table entry must have exactly one matching membership per
        band, and bucket sets must not contain keys absent from the key table.
        """
        indexed_keys = set(lsh.keys.keys())
        for Hs in lsh.keys.getmany(*indexed_keys):
            self.assertEqual(len(Hs), lsh.b)
        for hashtable in lsh.hashtables:
            for H in hashtable:
                for member in hashtable.get(H):
                    self.assertIn(member, indexed_keys)
        # Reconstruct membership from the key table and compare with buckets.
        for key in indexed_keys:
            for i, H in enumerate(lsh.keys.get(key)):
                self.assertIn(key, lsh.hashtables[i].get(H))

    def test_correction_batch_upsert_inserts_new_keys(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = self._make_minhash(b"a")
        with lsh.correction_batch() as batch:
            batch.upsert("a", m1)
            batch.commit()
        self.assertIn("a", lsh)
        self.assertIn("a", lsh.query(m1))
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_upsert_overwrites_existing_key_without_residue(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m_old = self._make_minhash(b"a")
        m_new = self._make_minhash(b"totally-different-content")
        lsh.insert("a", m_old)
        old_hashes = list(lsh.keys["a"])

        with lsh.correction_batch() as batch:
            batch.upsert("a", m_new)
            batch.commit()

        # The key table stores exactly the new band hashes.
        self.assertEqual(list(lsh.keys["a"]), lsh._band_hashes(m_new))
        # Old buckets no longer contain the key, even where hashes differ.
        for i, old_H in enumerate(old_hashes):
            self.assertNotIn("a", lsh.hashtables[i].get(old_H))
        # Queries and membership checks see only the new record.
        self.assertIn("a", lsh)
        self.assertIn("a", lsh.query(m_new))
        self.assertNotIn("a", lsh.query(m_old))
        # remove() works against the current record and leaves nothing behind.
        lsh.remove("a")
        self.assertNotIn("a", lsh)
        for table in lsh.hashtables:
            for H in table:
                self.assertNotIn("a", table[H])
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_remove_deletes_key_and_buckets(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2 = self._make_minhash(b"a"), self._make_minhash(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        with lsh.correction_batch() as batch:
            batch.remove("a")
            batch.commit()

        self.assertNotIn("a", lsh)
        self.assertIn("b", lsh)
        self.assertNotIn("a", lsh.query(m1))
        self.assertIn("b", lsh.query(m2))
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_same_key_uses_input_order(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2, m3 = (self._make_minhash(t) for t in (b"a", b"b", b"c"))
        lsh.insert("a", m1)

        # overwrite -> overwrite -> delete: deletion wins.
        with lsh.correction_batch() as batch:
            batch.upsert("a", m2)
            batch.upsert("a", m3)
            batch.remove("a")
            batch.commit()
        self.assertNotIn("a", lsh)
        self._assert_index_internals_consistent(lsh)

        # delete -> re-upsert within the batch: final upsert wins. The key
        # must exist when the deletion is staged (mirrors MinHashLSH.remove).
        lsh.insert("a", m1)
        with lsh.correction_batch() as batch:
            batch.remove("a")
            batch.upsert("a", m2)
            batch.commit()
        self.assertIn("a", lsh)
        self.assertEqual(list(lsh.keys["a"]), lsh._band_hashes(m2))
        self.assertIn("a", lsh.query(m2))
        self._assert_index_internals_consistent(lsh)

        # staged overwrite followed by another overwrite: last one wins.
        with lsh.correction_batch() as batch:
            batch.upsert("a", m1)
            batch.upsert("a", m3)
            batch.commit()
        self.assertEqual(list(lsh.keys["a"]), lsh._band_hashes(m3))
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_mixed_overwrite_delete_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2 = self._make_minhash(b"a"), self._make_minhash(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        m_corrected, m_new = self._make_minhash(b"a2"), self._make_minhash(b"c")

        with lsh.correction_batch() as batch:
            batch.upsert("a", m_corrected)
            batch.remove("b")
            batch.upsert("c", m_new)
            batch.commit()

        self.assertEqual(list(lsh.keys["a"]), lsh._band_hashes(m_corrected))
        self.assertNotIn("b", lsh)
        self.assertIn("c", lsh)
        self.assertEqual(set(lsh.query(m_corrected)), {"a"})
        self.assertEqual(set(lsh.query(m_new)), {"c"})
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_staging_failures_change_nothing(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = self._make_minhash(b"a")
        lsh.insert("a", m1)
        snapshot = {key: list(lsh.keys.get(key)) for key in lsh.keys}

        # Removing an unknown key fails while staging.
        with lsh.correction_batch() as batch:
            with self.assertRaises(ValueError):
                batch.remove("missing")
            batch.upsert("a", m1)
        # Leaving the context without commit discards the staged upsert.
        self.assertEqual({key: list(lsh.keys.get(key)) for key in lsh.keys}, snapshot)
        self.assertIn("a", lsh)
        self._assert_index_internals_consistent(lsh)

        # Wrong MinHash length fails while staging and leaves the index intact.
        with lsh.correction_batch() as batch, self.assertRaises(ValueError):
            batch.upsert("a", self._make_minhash(b"x", num_perm=32))
        self.assertEqual({key: list(lsh.keys.get(key)) for key in lsh.keys}, snapshot)

        # A committed batch cannot be reused.
        with lsh.correction_batch() as batch:
            batch.remove("a")
            batch.commit()
            with self.assertRaises(RuntimeError):
                batch.upsert("a", m1)
            with self.assertRaises(RuntimeError):
                batch.commit()
        self.assertNotIn("a", lsh)

    def test_correction_batch_commit_rolls_back_on_storage_failure(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2, m3 = (self._make_minhash(t) for t in (b"a", b"b", b"c"))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        snapshot = {key: list(lsh.keys.get(key)) for key in lsh.keys}
        bucket_snapshot = [{H: set(table.get(H)) for H in table} for table in lsh.hashtables]

        real_remove_val = lsh.hashtables[0].remove_val
        calls = {"n": 0}

        def failing_remove_val(H, key, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated storage failure")
            return real_remove_val(H, key, **kwargs)

        lsh.hashtables[0].remove_val = failing_remove_val
        try:
            with self.assertRaises(RuntimeError), lsh.correction_batch() as batch:
                batch.upsert("a", m3)
                batch.upsert("b", m3)
                batch.commit()
        finally:
            lsh.hashtables[0].remove_val = real_remove_val

        # The whole batch must be rolled back to the exact pre-commit state.
        self.assertEqual({key: list(lsh.keys.get(key)) for key in lsh.keys}, snapshot)
        for table, table_snapshot in zip(lsh.hashtables, bucket_snapshot):
            self.assertEqual({H: set(table.get(H)) for H in table}, table_snapshot)
        self.assertIn("a", lsh.query(m1))
        self.assertIn("b", lsh.query(m2))
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_empty_commit_is_noop(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = self._make_minhash(b"a")
        lsh.insert("a", m1)
        with lsh.correction_batch() as batch:
            batch.commit()
        self.assertIn("a", lsh)

    def test_correction_batch_rejects_wrong_scheme_on_commit(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=16)
        m1 = MinHash(16, scheme="affine32")
        m1.update(b"a")
        lsh.insert("a", m1)
        incompatible = MinHash(16, scheme="legacy")
        incompatible.update(b"b")
        snapshot = {key: list(lsh.keys.get(key)) for key in lsh.keys}
        with lsh.correction_batch() as batch:
            batch.upsert("a", incompatible)
            with self.assertRaises(ValueError):
                batch.commit()
        self.assertEqual({key: list(lsh.keys.get(key)) for key in lsh.keys}, snapshot)
        self._assert_index_internals_consistent(lsh)

    def test_correction_batch_redis(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSH(
                threshold=0.5,
                num_perm=16,
                storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
            )
            m1, m2 = self._make_minhash(b"a"), self._make_minhash(b"b")
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            m_corrected, m_new = self._make_minhash(b"a2"), self._make_minhash(b"c")

            with lsh.correction_batch() as batch:
                batch.upsert("a", m_corrected)
                batch.remove("b")
                batch.upsert("c", m_new)
                batch.commit()

            self.assertIn("a", lsh)
            self.assertNotIn("b", lsh)
            self.assertIn("c", lsh)
            self.assertIn("a", lsh.query(m_corrected))
            self.assertNotIn("a", lsh.query(m1))
            self.assertIn("c", lsh.query(m_new))
            # Follow-up remove works against the corrected record.
            lsh.remove("a")
            self.assertNotIn("a", lsh)
            lsh.remove("c")
            self.assertTrue(lsh.is_empty())

    # ------------------------------------------------------------------
    # Merge conflict policies
    # ------------------------------------------------------------------
    def test_merge_on_conflict_error_raises_without_touching_index(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2 = self._make_minhash(b"a"), self._make_minhash(b"b")
        lsh1.insert("a", m1)
        lsh1.insert("b", m2)
        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        m3 = self._make_minhash(b"c")
        lsh2.insert("a", m3)
        lsh2.insert("c", m3)
        snapshot = {key: list(lsh1.keys.get(key)) for key in lsh1.keys}

        with self.assertRaisesRegex(ValueError, "overlapping"):
            lsh1.merge(lsh2, on_conflict="error")
        # check_overlap=True is the same policy.
        with self.assertRaises(ValueError):
            lsh1.merge(lsh2, check_overlap=True)
        self.assertEqual({key: list(lsh1.keys.get(key)) for key in lsh1.keys}, snapshot)
        self.assertNotIn("c", lsh1)
        # Source index is untouched.
        self.assertIn("a", lsh2)
        self.assertIn("c", lsh2)

        with self.assertRaisesRegex(ValueError, "check_overlap"):
            lsh1.merge(lsh2, check_overlap=True, on_conflict="keep_local")

    def test_merge_on_conflict_keep_local(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2 = self._make_minhash(b"a"), self._make_minhash(b"b")
        lsh1.insert("a", m1)
        lsh1.insert("b", m2)
        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        m3 = self._make_minhash(b"c")
        lsh2.insert("a", m3)
        lsh2.insert("c", m3)

        lsh1.merge(lsh2, on_conflict="keep_local")

        # The local record of the overlapping key survives unchanged.
        self.assertEqual(list(lsh1.keys["a"]), lsh1._band_hashes(m1))
        self.assertIn("a", lsh1.query(m1))
        self.assertNotIn("a", lsh1.query(m3))
        self.assertIn("b", lsh1)
        self.assertIn("c", lsh1)
        self._assert_index_internals_consistent(lsh1)
        self.assertIn("a", lsh2)

    def test_merge_on_conflict_use_source_replaces_without_residue(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        m1, m2 = self._make_minhash(b"a"), self._make_minhash(b"b")
        lsh1.insert("a", m1)
        lsh1.insert("b", m2)
        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        m3 = self._make_minhash(b"c")
        lsh2.insert("a", m3)
        lsh2.insert("c", m3)

        lsh1.merge(lsh2, on_conflict="use_source")

        # The source record replaces the local one, including bucket layout.
        self.assertEqual(list(lsh1.keys["a"]), list(lsh2.keys["a"]))
        self.assertIn("a", lsh1.query(m3))
        self.assertNotIn("a", lsh1.query(m1))
        self.assertIn("c", lsh1)
        self.assertIn("b", lsh1)
        self._assert_index_internals_consistent(lsh1)
        # Follow-up remove against the replaced record leaves no candidates.
        lsh1.remove("a")
        self.assertNotIn("a", lsh1)
        self.assertNotIn("a", lsh1.query(m3))
        self.assertNotIn("a", lsh1.query(m1))
        self._assert_index_internals_consistent(lsh1)
        # Source index unaffected.
        self.assertIn("a", lsh2)

    def test_merge_invalid_on_conflict_policy(self):
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
        lsh1.insert("a", self._make_minhash(b"a"))
        with self.assertRaisesRegex(ValueError, "on_conflict"):
            lsh1.merge(lsh2, on_conflict="nonsense")


class TestWeightedMinHashLSH(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSH(threshold=0.8)
        self.assertTrue(lsh.is_empty())
        b1, r1 = lsh.b, lsh.r
        lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
        b2, r2 = lsh.b, lsh.r
        self.assertTrue(b1 < b2)
        self.assertTrue(r1 > r2)

    def test__H(self):
        """Check _H output consistent bytes length given
        the same concatenated hash value size.
        """
        mg = WeightedMinHashGenerator(100, sample_size=128)
        for _l in range(2, mg.sample_size + 1, 16):
            m = mg.minhash(np.random.randint(1, 99999999, 100))
            lsh = MinHashLSH(num_perm=128)
            lsh.insert("m", m)
            sizes = [len(H) for ht in lsh.hashtables for H in ht]
            self.assertTrue(all(sizes[0] == s for s in sizes))

    def test_insert(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            self.assertTrue(len(t) >= 1)
            items = []
            for H in t:
                items.extend(t[H])
            self.assertTrue("a" in items)
            self.assertTrue("b" in items)
        self.assertTrue("a" in lsh)
        self.assertTrue("b" in lsh)
        for i, H in enumerate(lsh.keys["a"]):
            self.assertTrue("a" in lsh.hashtables[i][H])

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        result = lsh.query(m1)
        self.assertTrue("a" in result)
        result = lsh.query(m2)
        self.assertTrue("b" in result)

        mg = WeightedMinHashGenerator(10, 5)
        m3 = mg.minhash(np.random.uniform(1, 10, 10))
        self.assertRaises(ValueError, lsh.query, m3)

    def test_remove(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)

        lsh.remove("a")
        self.assertTrue("a" not in lsh.keys)
        for table in lsh.hashtables:
            for H in table:
                self.assertGreater(len(table[H]), 0)
                self.assertTrue("a" not in table[H])

        self.assertRaises(ValueError, lsh.remove, "c")

    def test_pickle(self):
        lsh = MinHashLSH(threshold=0.5, num_perm=4)
        mg = WeightedMinHashGenerator(10, 4)
        m1 = mg.minhash(np.random.uniform(1, 10, 10))
        m2 = mg.minhash(np.random.uniform(1, 10, 10))
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh2 = pickle.loads(pickle.dumps(lsh))

        result = lsh2.query(m1)
        self.assertTrue("a" in result)
        result = lsh2.query(m2)
        self.assertTrue("b" in result)


if __name__ == "__main__":
    unittest.main()
