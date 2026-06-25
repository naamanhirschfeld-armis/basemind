//! Compact integer encodings for the git-history posting lists.
//!
//! A path's posting list is the set of commit *ordinals* that touched it. Ordinals are dense
//! `u32`s assigned in chronological order (oldest = 0), so a path's list is strictly ascending.
//! We store it as **delta + LEB128 unsigned varint**: each entry is `uvarint(ord - prev_ord)`
//! with `prev_ord` starting at 0. Ascending dense ordinals make the deltas tiny (usually a
//! 1–2 byte varint), which is what keeps a 2.5M-edge monorepo's posting store in the single-digit
//! MB range rather than `4 * edges` bytes raw.
//!
//! Newest-first reads ([`decode_ords_tail`]) must still scan the whole buffer because a delta
//! needs the running sum of every earlier delta, but they retain only the last `n` ordinals — so
//! a `commits_touching(path, limit=N)` over a hot file allocates `N`, not the whole history.

/// Append `value` to `out` as an LEB128 unsigned varint (7 bits/byte, high bit = continuation).
pub fn write_uvarint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Decode one LEB128 uvarint starting at `*cursor`, advancing the cursor past it. Returns `None`
/// on truncation or on an overflow past `u64` (more than 10 continuation bytes).
pub fn read_uvarint(buf: &[u8], cursor: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = *buf.get(*cursor)?;
        *cursor += 1;
        if shift >= 64 {
            return None; // would overflow u64
        }
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
    }
}

/// Encode an **ascending** slice of commit ordinals as delta-varints. The caller guarantees
/// ascending order (a commit touches a path at most once, and ordinals are assigned monotonically),
/// so deltas are positive; a non-ascending input still round-trips but wastes space.
pub fn encode_ords(ords: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ords.len() * 2);
    let mut prev: u32 = 0;
    for &ord in ords {
        let delta = ord.wrapping_sub(prev);
        write_uvarint(&mut out, u64::from(delta));
        prev = ord;
    }
    out
}

/// Decode a full delta-varint posting list back to absolute ordinals, ascending. A malformed tail
/// stops decoding and returns what was read cleanly (best-effort, mirroring the index module's
/// `None`-skip philosophy — a corrupt byte never panics a query).
pub fn decode_ords(buf: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let mut acc: u32 = 0;
    while cursor < buf.len() {
        let Some(delta) = read_uvarint(buf, &mut cursor) else {
            break;
        };
        acc = acc.wrapping_add(delta as u32);
        out.push(acc);
    }
    out
}

/// Decode only the last `n` ordinals (the newest, since ordinals ascend with commit time) without
/// materializing the whole list. Still scans the buffer forward — deltas require the running sum —
/// but retains a ring of at most `n`, so a hot path with 100k commits costs O(list) time but O(n)
/// space. Returns ascending order; the caller reverses for newest-first.
pub fn decode_ords_tail(buf: &[u8], n: usize) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    let mut ring: std::collections::VecDeque<u32> = std::collections::VecDeque::with_capacity(n);
    let mut cursor = 0;
    let mut acc: u32 = 0;
    while cursor < buf.len() {
        let Some(delta) = read_uvarint(buf, &mut cursor) else {
            break;
        };
        acc = acc.wrapping_add(delta as u32);
        if ring.len() == n {
            ring.pop_front();
        }
        ring.push_back(acc);
    }
    ring.into_iter().collect()
}

/// Append a length-prefixed (`uvarint(len) ‖ bytes`) byte string.
fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    write_uvarint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Read a length-prefixed byte string at `*cursor`, advancing past it.
fn read_bytes<'a>(buf: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    let len = read_uvarint(buf, cursor)? as usize;
    let end = cursor.checked_add(len)?;
    let out = buf.get(*cursor..end)?;
    *cursor = end;
    Some(out)
}

/// Compact, framing-free encoding of one commit's stored metadata. Replaces msgpack (which repeats
/// field names and frames every `(path_id, kind)` tuple) — on a 240k-commit monorepo this cut the
/// per-commit partition roughly in half. Layout:
///
/// ```text
/// sha[20] ‖ time:zigzag-varint ‖ author:len-prefixed ‖ summary:len-prefixed
///         ‖ uvarint(file_count) ‖ file_count × ( uvarint(Δpath_id) ‖ kind:u8 )
/// ```
///
/// `files` is sorted by `path_id` ascending so the deltas are small; order is irrelevant to the
/// caller (a commit's changed-file set).
pub fn encode_commit_meta(
    sha20: &[u8; 20],
    author_time_unix: i64,
    author: &[u8],
    summary: &[u8],
    files: &[(u32, u8)],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + 16 + author.len() + summary.len() + files.len() * 2);
    out.extend_from_slice(sha20);
    write_uvarint(&mut out, zigzag(author_time_unix));
    write_bytes(&mut out, author);
    write_bytes(&mut out, summary);

    let mut sorted: Vec<(u32, u8)> = files.to_vec();
    sorted.sort_unstable_by_key(|&(id, _)| id);
    write_uvarint(&mut out, sorted.len() as u64);
    let mut prev: u32 = 0;
    for (path_id, kind) in sorted {
        write_uvarint(&mut out, u64::from(path_id.wrapping_sub(prev)));
        out.push(kind);
        prev = path_id;
    }
    out
}

/// Decoded form of [`encode_commit_meta`]: borrows `author`/`summary` out of the buffer.
pub struct DecodedCommitMeta<'a> {
    pub sha20: [u8; 20],
    pub author_time_unix: i64,
    pub author: &'a [u8],
    pub summary: &'a [u8],
    pub files: Vec<(u32, u8)>,
}

/// Decode a [`encode_commit_meta`] payload. `None` on truncation/corruption (the caller treats a
/// bad row as a miss, never panics).
pub fn decode_commit_meta(buf: &[u8]) -> Option<DecodedCommitMeta<'_>> {
    let mut cursor = 0;
    let sha20: [u8; 20] = buf.get(..20)?.try_into().ok()?;
    cursor += 20;
    let author_time_unix = unzigzag(read_uvarint(buf, &mut cursor)?);
    let author = read_bytes(buf, &mut cursor)?;
    let summary = read_bytes(buf, &mut cursor)?;
    let count = read_uvarint(buf, &mut cursor)? as usize;
    let mut files = Vec::with_capacity(count);
    let mut acc: u32 = 0;
    for _ in 0..count {
        acc = acc.wrapping_add(read_uvarint(buf, &mut cursor)? as u32);
        let kind = *buf.get(cursor)?;
        cursor += 1;
        files.push((acc, kind));
    }
    Some(DecodedCommitMeta {
        sha20,
        author_time_unix,
        author,
        summary,
        files,
    })
}

/// Head fields of a stored commit — everything [`decode_commit_meta`] returns except the
/// changed-file list. Decoding stops right after `summary`, so no `files` Vec is allocated.
pub struct DecodedCommitHead<'a> {
    pub sha20: [u8; 20],
    pub author_time_unix: i64,
    pub author: &'a [u8],
    pub summary: &'a [u8],
}

/// Decode only the head of a [`encode_commit_meta`] payload (sha, time, author, summary), skipping
/// the file-change section entirely. The read paths that pass `include_files=false`
/// (`commits_touching`, `recent_changes`) never inspect the changed-file set, so this avoids the
/// `uvarint(count)` + per-edge delta loop + the `Vec<(u32, u8)>` allocation on every decoded commit.
pub fn decode_commit_meta_head(buf: &[u8]) -> Option<DecodedCommitHead<'_>> {
    let mut cursor = 0;
    let sha20: [u8; 20] = buf.get(..20)?.try_into().ok()?;
    cursor += 20;
    let author_time_unix = unzigzag(read_uvarint(buf, &mut cursor)?);
    let author = read_bytes(buf, &mut cursor)?;
    let summary = read_bytes(buf, &mut cursor)?;
    Some(DecodedCommitHead {
        sha20,
        author_time_unix,
        author,
        summary,
    })
}

/// Zig-zag map a signed int to unsigned so small-magnitude values (commit times fit in ~31 bits,
/// but the encoding is future-proof) varint-encode compactly.
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uvarint_round_trips_boundary_values() {
        for value in [0u64, 1, 127, 128, 16383, 16384, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            write_uvarint(&mut buf, value);
            let mut cursor = 0;
            assert_eq!(read_uvarint(&buf, &mut cursor), Some(value));
            assert_eq!(cursor, buf.len(), "cursor consumed exactly the varint");
        }
    }

    #[test]
    fn read_uvarint_returns_none_on_truncation() {
        // A lone continuation byte (high bit set) with no follow-up byte.
        let mut cursor = 0;
        assert_eq!(read_uvarint(&[0x80], &mut cursor), None);
    }

    #[test]
    fn ords_round_trip_through_encode_decode() {
        let ords = vec![0u32, 1, 2, 5, 100, 101, 250_000, 1_000_000];
        let buf = encode_ords(&ords);
        assert_eq!(decode_ords(&buf), ords);
    }

    #[test]
    fn empty_ord_list_encodes_to_empty_buffer() {
        assert!(encode_ords(&[]).is_empty());
        assert_eq!(decode_ords(&[]), Vec::<u32>::new());
    }

    #[test]
    fn tail_returns_last_n_in_ascending_order() {
        let ords: Vec<u32> = (0..1000).map(|i| i * 3).collect();
        let buf = encode_ords(&ords);
        let tail = decode_ords_tail(&buf, 5);
        assert_eq!(tail, vec![2985, 2988, 2991, 2994, 2997]);
    }

    #[test]
    fn tail_larger_than_list_returns_whole_list() {
        let ords = vec![10u32, 20, 30];
        let buf = encode_ords(&ords);
        assert_eq!(decode_ords_tail(&buf, 100), ords);
    }

    #[test]
    fn tail_of_zero_is_empty() {
        let buf = encode_ords(&[1, 2, 3]);
        assert_eq!(decode_ords_tail(&buf, 0), Vec::<u32>::new());
    }

    #[test]
    fn zigzag_round_trips() {
        for v in [0i64, 1, -1, i64::MAX, i64::MIN, 1_700_000_000, -42] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn commit_meta_round_trips() {
        let sha = [7u8; 20];
        let files = vec![(5u32, 1u8), (2, 0), (100, 2)]; // unsorted on input
        let buf = encode_commit_meta(&sha, 1_700_000_000, b"Ada <a@x>", b"fix: thing", &files);
        let decoded = decode_commit_meta(&buf).expect("decodes");
        assert_eq!(decoded.sha20, sha);
        assert_eq!(decoded.author_time_unix, 1_700_000_000);
        assert_eq!(decoded.author, b"Ada <a@x>");
        assert_eq!(decoded.summary, b"fix: thing");
        // Files come back sorted by path_id (set semantics).
        assert_eq!(decoded.files, vec![(2, 0), (5, 1), (100, 2)]);
    }

    #[test]
    fn commit_meta_head_decodes_without_files() {
        let sha = [9u8; 20];
        let files = vec![(5u32, 1u8), (2, 0), (100, 2)];
        let buf = encode_commit_meta(&sha, 1_700_000_000, b"Ada <a@x>", b"fix: thing", &files);
        let head = decode_commit_meta_head(&buf).expect("head decodes");
        assert_eq!(head.sha20, sha);
        assert_eq!(head.author_time_unix, 1_700_000_000);
        assert_eq!(head.author, b"Ada <a@x>");
        assert_eq!(head.summary, b"fix: thing");
        // Head decode agrees with the full decode on every non-file field, but allocates no Vec.
        let full = decode_commit_meta(&buf).expect("full decodes");
        assert_eq!(head.sha20, full.sha20);
        assert_eq!(head.summary, full.summary);
        assert!(
            decode_commit_meta_head(&buf[..5]).is_none(),
            "truncated → None"
        );
    }

    #[test]
    fn commit_meta_empty_files_and_truncation() {
        let buf = encode_commit_meta(&[0u8; 20], 0, b"", b"", &[]);
        let decoded = decode_commit_meta(&buf).expect("decodes");
        assert!(decoded.files.is_empty());
        assert!(decode_commit_meta(&buf[..10]).is_none(), "truncated → None");
    }

    #[test]
    fn delta_encoding_is_compact_for_dense_ascending() {
        // 1000 consecutive ordinals → 1000 single-byte deltas (delta 1 = one varint byte each),
        // far smaller than 4 bytes/ord raw.
        let ords: Vec<u32> = (5000..6000).collect();
        let buf = encode_ords(&ords);
        assert!(
            buf.len() <= 1000 + 2,
            "dense deltas stay ~1 byte each, got {}",
            buf.len()
        );
    }
}
