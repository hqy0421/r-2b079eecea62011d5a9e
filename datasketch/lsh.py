from __future__ import annotations

import pickle
import struct
from collections.abc import Hashable
from typing import Callable, List, Optional, Union

from scipy.integrate import quad as integrate

from datasketch.minhash import MinHash, _check_scheme_consistency
from datasketch.storage import (
    OrderedStorage,
    UnorderedStorage,
    _random_name,
    ordered_storage,
    unordered_storage,
)
from datasketch.weighted_minhash import WeightedMinHash

# Conflict resolution policies for :meth:`MinHashLSH.merge`.
MERGE_CONFLICT_ERROR = "error"
MERGE_CONFLICT_KEEP_LOCAL = "keep_local"
MERGE_CONFLICT_USE_SOURCE = "use_source"
_VALID_MERGE_CONFLICT_POLICIES = (
    MERGE_CONFLICT_ERROR,
    MERGE_CONFLICT_KEEP_LOCAL,
    MERGE_CONFLICT_USE_SOURCE,
)


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, _err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, _err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    """Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSH:
    """The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (Tuple[float, float]): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (Optiona[Tuple[int, int]]): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.
        storage_config (Optional[Dict]): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (Optional[bool]): If True, all keys are pickled to bytes before
            insertion. If not specified, a default value is chosen based on the
            `storage_config`.
        hashfunc (Optional[Callable[[bytes], bytes]]): If a hash function is provided it will be used to
            compress the index keys to reduce the memory footprint. This could cause a higher
            false positive rate.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).

    Examples:
        Create an index with 128 permutation functions optimized for Jaccard
        threshold 0.9:

        .. code-block:: python

            lsh = MinHashLSH(threshold=0.9, num_perm=128)
            print(lsh.b, lsh.r)
            # 5 25

        The built-in optimizer will try to minimize the weighted sum of
        probabilities of false positive and false negative. The algorithm is
        a simple grid search over the space of possible parameters.

        Note that it is possible to get :attr:`b` (number of bands) and
        :attr:`r` (band size) that do not sum to :attr:`num_perm`, leading to
        unused permutation values in the indexed MinHash.
        This is because the optimizer only considers bands of
        the same size, and the number of bands is not necessarily a divisor of
        :attr:`num_perm`.

        Instead of using the built-in optimizer, you can customize the LSH
        parameters your self. The snippet below creates an index with 128
        permutation functions and 16 bands each with size 8, skipping the
        optimization step:

        .. code-block:: python

            lsh = MinHashLSH(num_perm=128, params=(16, 8))
            print(lsh.b, lsh.r)
            # 16 8

        Create an index backed by Redis storage:

        .. code-block:: python

            lsh = MinHashLSH(
                threshold=0.9,
                num_perm=128,
                storage_config={
                    "type": "redis",
                    "basename": b"mylsh",  # optional, defaults to a random string.
                    "redis": {"host": "localhost", "port": 6379},
                },
            )

        The `basename` property is optional. It is used to generate key prefixes
        in the storage layer to uniquely identify data associated with this LSH.
        Thus, if you create a new LSH object with the same `basename`, you will
        be using the same underlying data in the storage layer associated with
        a previous LSH object. If you do not set this property, a random string
        will be generated instead.

    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: Optional[tuple[int, int]] = None,
        storage_config: Optional[dict] = None,
        prepickle: Optional[bool] = None,
        hashfunc: Optional[Callable[[bytes], bytes]] = None,
    ) -> None:
        storage_config = storage_config if storage_config else {"type": "dict"}
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    "The product of b and r in params is "
                    f"{self.b} * {self.r} = {self.b * self.r} -- it must be less than num_perm {num_perm}. "
                    "Did you forget to specify num_perm?"
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight)
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")

        self.prepickle = storage_config["type"] == "redis" if prepickle is None else prepickle
        self._require_bytes_keys = not (storage_config["type"] == "dict" or self.prepickle)

        self.hashfunc = hashfunc
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        basename = storage_config.get("basename", _random_name(11))
        self.hashtables: List[UnorderedStorage] = [
            unordered_storage(
                storage_config,
                name=b"".join([basename, b"_bucket_", struct.pack(">H", i)]),
            )
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.keys: OrderedStorage = ordered_storage(storage_config, name=b"".join([basename, b"_keys"]))
        # The permutation scheme of the indexed MinHash, learned from the
        # first insert. Note that an index attached to pre-existing external
        # storage (e.g. Redis) re-learns the scheme on its first insert.
        self._minhash_scheme: Optional[str] = None

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self.keys.buffer_size = value
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication: bool = True,
    ):
        """Insert a key to the index, together with a MinHash or Weighted MinHash
        of the set referenced by the key.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinHash]): The MinHash of the set.
            check_duplication (bool): To avoid duplicate keys in the storage
                (`default=True`). It's recommended to not change the default, but
                if you want to avoid the overhead during insert you can set
                `check_duplication = False`.

        """
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def merge(
        self,
        other: MinHashLSH,
        check_overlap: bool = False,
        on_conflict: Optional[str] = MERGE_CONFLICT_KEEP_LOCAL,
    ):
        """Merge the other MinHashLSH with this one, making this one the union
        of both.

        Note:
            The indexes must use the same number and size of bands, key
            serialization mode, band hash function, and byte-key requirements.
            Threshold, weights, and storage backend may differ when those
            effective index parameters are equivalent.

        Args:
            other (MinHashLSH): The other MinHashLSH. It is never modified.
            check_overlap (bool): Check if there are any overlapping keys before
                merging and raise if there are any. (`default=False`)
                Equivalent to ``on_conflict="error"``; it is an error to pass
                ``check_overlap=True`` together with any other ``on_conflict``
                policy.
            on_conflict (Optional[str]): How to resolve keys present in both
                indexes. One of:

                * :data:`MERGE_CONFLICT_ERROR` (``"error"``): raise a
                  :class:`ValueError` when an overlapping key exists, without
                  changing this index.
                * :data:`MERGE_CONFLICT_KEEP_LOCAL` (``"keep_local"``, the
                  default): keep this index's record for overlapping keys and
                  only add keys that exist solely in the other index.
                * :data:`MERGE_CONFLICT_USE_SOURCE` (``"use_source"``): replace
                  this index's record of every overlapping key with the other
                  index's record. Old bucket memberships are removed cleanly, so
                  a later :meth:`query`, ``key in lsh`` and :meth:`remove` all
                  see the source record and no stale candidates remain.

                The merge is applied atomically: any failure leaves this index
                untouched. The source index ``other`` is never modified.

        Raises:
            ValueError: If the two MinHashLSH have different initialization
                parameters or incompatible MinHash schemes, if both
                ``check_overlap`` and ``on_conflict`` are given in conflicting
                ways, or if an overlapping key exists under the
                :data:`MERGE_CONFLICT_ERROR` policy.

        """
        self._merge(other, check_overlap=check_overlap, on_conflict=on_conflict)

    def insertion_session(self, buffer_size: int = 50000) -> MinHashLSHInsertionSession:
        """Create a context manager for fast insertion into this index.

        Args:
        e         buffer_size (int): The buffer size for insert_session mode (default=50000).

        Returns:
                 MinHashLSHInsertionSession: The context manager.

        Example:
                 Insert 100 MinHashes into an Redis-backed index using a session:

                 .. code-block:: python

                     from datasketch import MinHash, MinHashLSH
                     import numpy as np

                     minhashes = []
                     for i in range(100):
                         m = MinHash(num_perm=128)
                         m.update_batch(np.random.randint(low=0, high=30, size=10))
                         minhashes.append(m)

                     lsh = MinHashLSH(
                         threshold=0.5,
                         num_perm=128,
                         storage_config={
                             "type": "redis",
                             "redis": {"host": "localhost", "port": 6379},
                         },
                     )
                     with lsh.insertion_session() as session:
                         for i, m in enumerate(minhashes):
                             session.insert(i, m)

        """
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def deletion_session(self, buffer_size: int = 50000) -> MinHashLSHDeletionSession:
        """Create a context manager for fast deletion from this index.

        Args:
                buffer_size (int): The buffer size for deletion operations (default=50000).

        Returns:
                MinHashLSHDeletionSession: The context manager.

        r    Example:
                Delete keys from a Redis-backed index using a session:

                .. code-block:: python

                    from datasketch import MinHashLSH

                    lsh = MinHashLSH(
                        threshold=0.5,
                        num_perm=128,
                        storage_config={
                            "type": "redis",
                            "redis": {"host": "localhost", "port": 6379},
                        },
                    )

                    # ... insert some data ...

                    keys_to_delete = ["key1", "key2", "key3"]
                    with lsh.deletion_session() as session:
                        for key in keys_to_delete:
                            session.remove(key)

        """
        return MinHashLSHDeletionSession(self, buffer_size=buffer_size)

    def correction_batch(self) -> MinHashLSHCorrectionBatch:
        """Create a batch of atomic record corrections (overwrites and deletes).

        The batch stages corrections and applies them all at once on
        :meth:`MinHashLSHCorrectionBatch.commit`. Corrections are validated
        while staging, so a commit either applies the whole batch or raises
        without changing a single stored record. On a successful commit,
        :meth:`query`, ``key in lsh`` and :meth:`remove` all reflect the same
        current records.

        Returns:
            MinHashLSHCorrectionBatch: The correction batch, usable as a context
            manager. Exiting the context manager without committing discards the
            staged corrections.

        Example:
            Correct and delete records atomically:

            .. code-block:: python

                with lsh.correction_batch() as batch:
                    batch.upsert("record-1", corrected_minhash)
                    batch.upsert("record-2", another_minhash)
                    batch.remove("record-3")
                    batch.commit()

        """
        return MinHashLSHCorrectionBatch(self)

    def _prepare_key(self, key: Hashable) -> Hashable:
        """Validate and serialize a user-provided key to its storage form."""
        if self._require_bytes_keys and not isinstance(key, bytes):
            raise TypeError(
                f"prepickle=False requires bytes keys for non-dict storage, got {type(key).__name__}. "
                "Either pass bytes keys or use prepickle=True for automatic serialization."
            )
        if self.prepickle:
            key = pickle.dumps(key)
        return key

    def _band_hashes(self, minhash: Union[MinHash, WeightedMinHash]) -> list:
        """Compute the per-band bucket hashes of a MinHash."""
        return [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]

    def _insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication: bool = True,
        buffer: bool = False,
    ):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        self._minhash_scheme = _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        key = self._prepare_key(key)
        if check_duplication and key in self.keys:
            raise ValueError("The given key already exists")
        Hs = self._band_hashes(minhash)
        self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def __equivalent(self, other: MinHashLSH) -> bool:
        """Returns:
        bool: Whether stored keys and band hashes can be copied safely.

        """
        return (
            type(self) is type(other)
            and self.h == other.h
            and self.b == other.b
            and self.r == other.r
            and self.prepickle == other.prepickle
            and self.hashfunc is other.hashfunc
            and self._require_bytes_keys == other._require_bytes_keys
        )

    def _merge(
        self,
        other: MinHashLSH,
        check_overlap: bool = False,
        on_conflict: Optional[str] = MERGE_CONFLICT_KEEP_LOCAL,
    ) -> None:
        if not self.__equivalent(other):
            if type(self) is not type(other):
                raise ValueError(f"Cannot merge type MinHashLSH and type {type(other).__name__}.")
            raise ValueError("Cannot merge MinHashLSH with different initialization parameters.")
        if check_overlap:
            if on_conflict not in (None, MERGE_CONFLICT_ERROR):
                raise ValueError(
                    "check_overlap=True is equivalent to on_conflict=%r and cannot be combined "
                    "with on_conflict=%r" % (MERGE_CONFLICT_ERROR, on_conflict)
                )
            on_conflict = MERGE_CONFLICT_ERROR
        if on_conflict is None:
            on_conflict = MERGE_CONFLICT_KEEP_LOCAL
        if on_conflict not in _VALID_MERGE_CONFLICT_POLICIES:
            raise ValueError("on_conflict must be one of %r, got %r" % (_VALID_MERGE_CONFLICT_POLICIES, on_conflict))

        known, other_known = getattr(self, "_minhash_scheme", None), getattr(other, "_minhash_scheme", None)
        if known is not None and other_known is not None and known != other_known:
            raise ValueError(
                "Cannot merge MinHashLSH indexed with MinHash scheme %r into one indexed with scheme %r"
                % (other_known, known)
            )

        # Pre-scan for overlapping keys. Everything below this point must be
        # applied atomically by the correction batch, so erroring here keeps
        # the index untouched and leaves the source index unaffected.
        if on_conflict == MERGE_CONFLICT_ERROR:
            overlapping = set(self.keys).intersection(set(other.keys))
            if overlapping:
                raise ValueError("The keys are overlapping, duplicate key exists.")

        batch = MinHashLSHCorrectionBatch(self, scheme=known or other_known)
        for key in other.keys:
            Hs = other.keys.get(key)
            if key in self.keys and on_conflict == MERGE_CONFLICT_KEEP_LOCAL:
                continue
            # New keys ("error" guarantees none overlap; "keep_local" skips
            # them above) and replaced keys are both expressed as staged
            # corrections, so overwrites remove stale bucket memberships.
            batch._stage_hashes(key, list(Hs))
        batch.commit()

    def query(self, minhash) -> list[Hashable]:
        """Giving the MinHash of the query set, retrieve
        the keys that reference sets with Jaccard
        similarities likely greater than the threshold.

        Results are based on minhash segment collision
        and are thus approximate. For more accurate results,
        filter again with :meth:`MinHash.jaccard`. For exact results,
        filter by computing Jaccard similarity using original sets.

        Args:
            minhash (MinHash): The MinHash of the query set.

        Returns:
            list: a list of unique keys.

        Example:
            Query and rank results using :meth:`MinHash.jaccard`.

            .. code-block:: python

                from datasketch import MinHash, MinHashLSH
                import numpy as np

                # Generate 100 random MinHashes.
                minhashes = MinHash.bulk(np.random.randint(low=0, high=30, size=(100, 10)), num_perm=128)

                # Create LSH index.
                lsh = MinHashLSH(threshold=0.5, num_perm=128)
                for i, m in enumerate(minhashes):
                    lsh.insert(i, m)

                # Get the initial results from LSH.
                query = minhashes[0]
                results = lsh.query(query)

                # Rank results using Jaccard similarity estimated by MinHash.
                results = [(query.jaccard(minhashes[key]), key) for key in results]
                results.sort(reverse=True)
                print(results)

            Output:

            .. code-block::

                [(1.0, 0), (0.421875, 4), (0.4140625, 19), (0.359375, 58),
                 (0.3359375, 78), (0.265625, 62), (0.2578125, 11), (0.25, 98), (0.171875, 21)]

            Note that although the threshold is set to 0.5, the results are not
            guaranteed to be above 0.5 because the LSH index is approximate and
            the Jaccard similarity is estimated by MinHash.

        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        return list(candidates)

    def add_to_query_buffer(self, minhash: Union[MinHash, WeightedMinHash]) -> None:
        """Giving the MinHash of the query set, buffer
        queries to retrieve the keys that references
        sets with Jaccard similarities greater than
        the threshold.

        Buffered queries can be executed using
        :meth:`collect_query_buffer`. The combination of these
        functions is way faster if cassandra backend
        is used with `shared_buffer`.

        Args:
            minhash (MinHash): The MinHash of the query set.

        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            hashtable.add_to_select_buffer([H])

    def collect_query_buffer(self) -> list[Hashable]:
        """Execute and return buffered queries given
        by :meth:`add_to_query_buffer`.

        If multiple query MinHash were added to the query buffer,
        the intersection of the results of all query MinHash will be returned.

        Returns:
            list: a list of unique keys.

        """
        collected_result_lists = [hashtable.collect_select_buffer() for hashtable in self.hashtables]
        if not any(collected_result_lists):
            return []

        # Each buffered query contributes one result list per hashtable. We first
        # union candidates across bands for each query, then intersect across the
        # buffered queries to match repeated calls to `query()`.
        per_query_result_sets = [
            set().union(*query_result_lists)
            for query_result_lists in zip(*collected_result_lists)
        ]
        if not per_query_result_sets:
            return []

        candidates = set.intersection(*per_query_result_sets)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        return list(candidates)

    def __contains__(self, key: Hashable) -> bool:
        """Args:
            key (Hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.

        """
        if self.prepickle:
            key = pickle.dumps(key)
        return key in self.keys

    def remove(self, key: Hashable) -> None:
        """Remove the key from the index.

        Args:
            key (Hashable): The unique identifier of a set.

        Raises:
            ValueError: If the key does not exist.

        """
        self._remove(key, buffer=False)

    def _remove(self, key: Hashable, buffer: bool = False) -> None:
        """Internal remove method with optional buffering support.

        Args:
            key (Hashable): The unique identifier of a set.
            buffer (bool): If True, buffer the removal operations.

        Raises:
            ValueError: If the key does not exist.

        """
        if self.prepickle:
            key = pickle.dumps(key)
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable.remove_val(H, key, buffer=buffer)
            if not hashtable.get(H):
                hashtable.remove(H, buffer=buffer)
        self.keys.remove(key, buffer=buffer)

    def is_empty(self) -> bool:
        """Returns:
        bool: `True` only if the index is empty.

        """
        return any(t.size() == 0 for t in self.hashtables)

    def _byteswap(self, hs):
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs):
        if self.hashfunc is None:
            raise RuntimeError("Hash function not configured.")
        return self.hashfunc(bytes(hs.byteswap().data))

    def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        return candidates

    def get_counts(self) -> list[dict[Hashable, int]]:
        """Returns a list of length :attr:`b` (i.e., number of hash tables) with
        each element a dictionary mapping hash table bucket key to the number
        of indexed keys stored under each bucket.

        Returns:
            list: a list of dictionaries.

        """
        return [hashtable.itemcounts() for hashtable in self.hashtables]

    def get_subset_counts(self, *keys: Hashable) -> list[dict[Hashable, int]]:
        """Returns the bucket allocation counts (see :meth:`get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (Hashable) : the keys for which to get the bucket allocation
                counts.

        Returns:
            list: a list of dictionaries.

        """
        key_set = [pickle.dumps(key) for key in set(keys)] if self.prepickle else list(set(keys))
        hashtables = [unordered_storage({"type": "dict"}) for _ in range(self.b)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class MinHashLSHCorrectionBatch:
    """An atomic batch of record corrections (overwrites and deletes) for a
    :class:`MinHashLSH`.

    Create one with :meth:`MinHashLSH.correction_batch`. Corrections are staged
    in memory first and only touch the index when :meth:`commit` is called. All
    validation happens while staging, so a commit either applies every staged
    correction or fails without changing the index at all. When several
    corrections target the same key, they are applied in input order and the
    last one determines the key's final state.

    On a successful commit, :meth:`MinHashLSH.query`, ``key in lsh`` and
    :meth:`MinHashLSH.remove` all reflect the same current records: an
    overwrite removes the key from every old bucket before adding it to the new
    ones, leaving no stale candidates behind.

    The batch can be used as a context manager. Leaving the ``with`` block
    without calling :meth:`commit` discards all staged corrections.

    Args:
        lsh (MinHashLSH): The index the corrections will be applied to.
        scheme (Optional[str]): Internal use only -- the MinHash permutation
            scheme to remember on commit when the index has not seen a MinHash
            yet. Public callers leave this unset.

    Example:

        .. code-block:: python

            with lsh.correction_batch() as batch:
                batch.upsert("record-1", corrected_minhash)
                batch.remove("record-2")
                batch.commit()

    """

    # Sentinel marking a staged correction that deletes the key.
    _DELETE = object()

    def __init__(self, lsh: MinHashLSH, scheme: Optional[str] = None):
        self.lsh = lsh
        # Storage-level key -> list of per-band bucket hashes, or _DELETE.
        self._staged: dict = {}
        self._closed = False
        # Scheme learned from staged MinHashes; validated against the index on
        # commit so merely staging a wrong-scheme sketch cannot mutate the index.
        self._scheme = scheme

    def __enter__(self) -> MinHashLSHCorrectionBatch:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # An uncommitted batch leaves the index untouched.
        if not self._closed:
            self.discard()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This correction batch has already been committed or discarded")

    def _stage_hashes(self, storage_key: Hashable, Hs: list) -> None:
        """Stage a precomputed overwrite using storage-level key and hashes.

        Internal entry point used by :meth:`MinHashLSH._merge`, where the
        source index already stores serialized keys and band hashes.
        """
        self._ensure_open()
        self._staged[storage_key] = Hs

    def upsert(self, key: Hashable, minhash: Union[MinHash, WeightedMinHash]) -> None:
        """Stage an overwrite (or insertion) of ``key`` with ``minhash``.

        If the key is already indexed, its record is replaced on commit -- the
        old bucket memberships are removed and the new MinHash is the only
        record visible to queries. If it is not indexed, this behaves like an
        insertion. The same key may be corrected several times within one
        batch; corrections take effect in input order.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinHash]): The corrected MinHash.

        Raises:
            ValueError: If the MinHash has the wrong number of permutations or
                a different permutation scheme than the index.
            TypeError: If the key does not satisfy the index's key requirements.

        """
        self._ensure_open()
        if len(minhash) != self.lsh.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.lsh.h, len(minhash)))
        self._scheme = _check_scheme_consistency(self._scheme, minhash)
        storage_key = self.lsh._prepare_key(key)
        self._staged[storage_key] = self.lsh._band_hashes(minhash)

    def remove(self, key: Hashable) -> None:
        """Stage the deletion of ``key`` from the index.

        A staged deletion of a key that is neither indexed nor staged for
        insertion raises immediately, mirroring :meth:`MinHashLSH.remove`. A
        key staged for overwrite within the same batch can be deleted again;
        the last correction in input order wins.

        Args:
            key (Hashable): The unique identifier of the set.

        Raises:
            ValueError: If the key does not exist in the index and has no
                staged correction in this batch.
            TypeError: If the key does not satisfy the index's key requirements.

        """
        self._ensure_open()
        storage_key = self.lsh._prepare_key(key)
        if storage_key not in self.lsh.keys and storage_key not in self._staged:
            raise ValueError("The given key does not exist")
        self._staged[storage_key] = self._DELETE

    def commit(self) -> None:
        """Apply all staged corrections atomically.

        Every correction is validated before the index is touched. The commit
        replays the corrections in staging order while recording inverse
        operations; if any storage operation fails, the inverses are replayed
        to restore the exact pre-commit state before the exception is raised.
        With the in-memory ``dict`` storage backend this guarantees the whole
        batch either succeeds or has no effect.

        Raises:
            ValueError: If a staged MinHash uses a different permutation scheme
                than the MinHashes already indexed.
            RuntimeError: If the batch was already committed or discarded.

        """
        self._ensure_open()
        known_scheme = getattr(self.lsh, "_minhash_scheme", None)
        if known_scheme is not None and self._scheme is not None and known_scheme != self._scheme:
            raise ValueError(
                "MinHash scheme %r does not match scheme %r of the MinHash previously given to this index"
                % (self._scheme, known_scheme)
            )

        undo_journal: list = []
        try:
            for storage_key, Hs in self._staged.items():
                if Hs is self._DELETE:
                    self._apply_delete(storage_key, undo_journal)
                else:
                    self._apply_upsert(storage_key, Hs, undo_journal)
        except BaseException:
            for inverse in reversed(undo_journal):
                inverse()
            raise
        if known_scheme is None:
            self.lsh._minhash_scheme = self._scheme
        self._closed = True

    def discard(self) -> None:
        """Drop every staged correction without touching the index."""
        self._staged.clear()
        self._closed = True

    def _record(self, undo_journal: list, mutation, inverse) -> None:
        """Run one storage mutation and remember how to undo it immediately."""
        mutation()
        undo_journal.append(inverse)

    def _apply_upsert(self, storage_key: Hashable, Hs: list, undo_journal: list) -> None:
        lsh = self.lsh
        if storage_key in lsh.keys:
            old_Hs = list(lsh.keys.get(storage_key))
            if old_Hs == Hs:
                # The record is already identical: nothing to do, nothing to undo.
                return
            # Remove every old bucket membership. Empty buckets are dropped,
            # exactly as MinHashLSH._remove does. Each step is journaled so a
            # failure halfway through still restores the complete old record.
            for old_H, hashtable in zip(old_Hs, lsh.hashtables):

                def remove_membership(h=old_H, table=hashtable):
                    table.remove_val(h, storage_key)
                    if not table.get(h):
                        table.remove(h)

                def restore_membership(h=old_H, table=hashtable):
                    table.insert(h, storage_key)

                self._record(undo_journal, remove_membership, restore_membership)
            self._record(
                undo_journal,
                lambda: lsh.keys.remove(storage_key),
                lambda hashes=old_Hs: lsh.keys.insert(storage_key, *hashes),
            )
        else:
            # New key: failure before the new record is fully written must not
            # leave a partial entry behind.
            self._record(
                undo_journal,
                lambda: lsh.keys.insert(storage_key, *Hs),
                lambda: lsh.keys.remove(storage_key) if storage_key in lsh.keys else None,
            )
            for H, hashtable in zip(Hs, lsh.hashtables):
                self._record(
                    undo_journal,
                    lambda h=H, table=hashtable: table.insert(h, storage_key),
                    lambda h=H, table=hashtable: self._remove_membership(table, h, storage_key),
                )
            return

        # Existing key: write the replacement record, step by step.
        self._record(
            undo_journal,
            lambda: lsh.keys.insert(storage_key, *Hs),
            lambda: lsh.keys.remove(storage_key),
        )
        for H, hashtable in zip(Hs, lsh.hashtables):
            self._record(
                undo_journal,
                lambda h=H, table=hashtable: table.insert(h, storage_key),
                lambda h=H, table=hashtable: self._remove_membership(table, h, storage_key),
            )

    @staticmethod
    def _remove_membership(hashtable, H, storage_key) -> None:
        hashtable.remove_val(H, storage_key)
        if not hashtable.get(H):
            hashtable.remove(H)

    def _apply_delete(self, storage_key: Hashable, undo_journal: list) -> None:
        lsh = self.lsh
        # The key was validated at staging time; an earlier correction in this
        # same batch may still have removed it.
        if storage_key not in lsh.keys:
            return
        old_Hs = list(lsh.keys.get(storage_key))
        for H, hashtable in zip(old_Hs, lsh.hashtables):
            self._record(
                undo_journal,
                lambda h=H, table=hashtable: self._remove_membership(table, h, storage_key),
                lambda h=H, table=hashtable: table.insert(h, storage_key),
            )
        self._record(
            undo_journal,
            lambda: lsh.keys.remove(storage_key),
            lambda hashes=old_Hs: lsh.keys.insert(storage_key, *hashes),
        )


class MinHashLSHInsertionSession:
    """Context manager for batch insertion of documents into a MinHashLSH.

    Args:
        lsh (MinHashLSH): The MinHashLSH to insert into.
        buffer_size (int): The buffer size for insert_session mode.

    """

    def __init__(self, lsh: MinHashLSH, buffer_size: int):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> MinHashLSHInsertionSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication=True,
    ) -> None:
        """Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinhash]): The MinHash of the set.

        """
        self.lsh._insert(key, minhash, check_duplication=check_duplication, buffer=True)


class MinHashLSHDeletionSession:
    """Context manager for batch deletion of documents from a MinHashLSH.

    Args:
        lsh (MinHashLSH): The MinHashLSH to delete from.
        buffer_size (int): The buffer size for deletion operations.

    """

    def __init__(self, lsh: MinHashLSH, buffer_size: int):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> MinHashLSHDeletionSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def remove(self, key: Hashable) -> None:
        """Remove a key from the index.

        Args:
            key (Hashable): The unique identifier to remove.

        Raises:
            ValueError: If the key does not exist.

        """
        self.lsh._remove(key, buffer=True)
