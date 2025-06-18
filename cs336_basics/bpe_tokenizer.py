import os, regex as re, multiprocessing as mp
from typing import Iterable, Iterator
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import pathlib, pickle, statistics, binascii, random, time

# ---------- 全局配置 ----------
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# ---------- 多进程友好的预分词 ----------
def pre_tokenization(chunk: str) -> Counter[(tuple[bytes, ...])]:
    """把文本块 token 化，返回 Counter[(tuple[bytes, ...]) → 次数]。"""
    c = Counter()
    for m in re.finditer(PAT, chunk):
        b = m.group().encode("utf-8")
        tok = tuple(b[i : i + 1] for i in range(len(b)))
        c[tok] += 1
    return c


# ---------- 构建初始 pair 索引 ----------
def generate_pairs(freqs: Counter):
    pairs, occ = {}, defaultdict(set)
    for tok, n in freqs.items():
        for i in range(len(tok) - 1):
            p = (tok[i], tok[i + 1])
            pairs[p] = pairs.get(p, 0) + n
            occ[p].add(tok)
    return pairs, occ


# ---------- 初始化 vocab ----------
def init_vocabs(special):
    v = {i: bytes([i]) for i in range(256)}
    for idx, t in enumerate(special):
        v[256 + idx] = t.encode()
    return v


# ---------- pair 计数增删 ----------
def add_pair(p, tok, n, pairs, occ):
    pairs[p] = pairs.get(p, 0) + n
    occ[p].add(tok)


def del_pair(p, tok, n, pairs, occ):
    cnt = pairs[p] - n
    if cnt <= 0:
        pairs.pop(p, None)
        occ.pop(p, None)
    else:
        pairs[p] = cnt
        s = occ[p]
        s.discard(tok)
        if not s:
            occ.pop(p, None)


# ---------- 单次 merge ----------
def merge(freqs, pairs, occ, vocabs, merges):
    max_pair = max(pairs, key=lambda k: (pairs[k], k))
    affected = list(occ[max_pair])

    for old in affected:
        n = freqs.pop(old)
        for i in range(len(old) - 1):
            del_pair((old[i], old[i + 1]), old, n, pairs, occ)

        merged_b = max_pair[0] + max_pair[1]
        new = []
        i = 0
        while i < len(old):
            if i < len(old) - 1 and old[i] == max_pair[0] and old[i + 1] == max_pair[1]:
                new.append(merged_b)
                i += 2
            else:
                new.append(old[i])
                i += 1
        new = tuple(new)

        freqs[new] += n
        for i in range(len(new) - 1):
            add_pair((new[i], new[i + 1]), new, n, pairs, occ)

    pairs.pop(max_pair, None)
    occ.pop(max_pair, None)
    vocabs[len(vocabs)] = merged_b
    merges.append(max_pair)


# ---------- 训练入口 ----------
def train_bpe(path: str, vocab_size: int, special: list[str] | None=None):
    with open(path, "rb") as f:
        bounds = find_chunk_boundaries(f, os.cpu_count(), b"<|endoftext|>")

        # 准备文本片段
        esc = map(re.escape, special)
        splitter = re.compile("|".join(esc))
        pieces = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            f.seek(s)
            chunk = f.read(e - s).decode("utf-8", "ignore")
            pieces.extend(splitter.split(chunk))

    # 多进程 token 计数
    with mp.Pool(processes=2 * os.cpu_count()) as pool:
        counters = pool.map(pre_tokenization, pieces)

    freqs = Counter()
    for c in counters:
        freqs.update(c)

    pairs, occ = generate_pairs(freqs)
    vocabs, merges = init_vocabs(special), []

    while len(vocabs) < vocab_size:
        merge(freqs, pairs, occ, vocabs, merges)

    return vocabs, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocabs = vocab
        self.vocabs_lookup = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.vocab_size = len(vocab)
        self.special_tokens = special_tokens or []
        self.special_tokens.sort(key=len, reverse=True)
        self.special_tokens_set = set(self.special_tokens)
        self.esc = map(re.escape, self.special_tokens)
        self.splitter = re.compile("({})".format("|".join(self.esc)))
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)
    
    def _get_merged_tokens(
        self, tokens: list[bytes]
    ) -> list[int]:
        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(tokens)-1):
                r = self.merge_rank.get((tokens[i], tokens[i+1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_idx = r, i
            if best_idx == -1:
                break
            tokens[best_idx:best_idx+2] = [tokens[best_idx] + tokens[best_idx+1]]
        return [self.vocabs_lookup[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            splits = re.split(self.splitter, text)
        else:
            splits = [text]
        result = []
        for word in splits:
            if word in self.special_tokens_set:
                result.append(self.vocabs_lookup.get(word.encode("utf-8"), -1))
                continue
            for m in re.finditer(PAT, word):
                b = m.group().encode("utf-8")
                tokens = [b[i : i + 1] for i in range(len(b))]
                merged_tokens = self._get_merged_tokens(tokens)
                result.extend(merged_tokens)
        return result
        
    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocabs[i] for i in ids if i in self.vocabs]
        return b"".join(tokens).decode("utf-8", "ignore")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

# ---------- example ----------
if __name__ == "__main__":
    start_time = time.time()
    voc, mer = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f} 秒")
    print("done:", len(voc), "tokens,", len(mer), "merges")
    # ---------- 1. 持久化 ----------
    out_pkl = pathlib.Path("bpe_model.pkl")
    out_pkl.write_bytes(pickle.dumps((voc, mer)))
    print(f"✔️  vocabs/merges 已保存到: {out_pkl.resolve()}")

    # ---------- 2. 统计信息 ----------
    lens = {tid: len(tok) for tid, tok in voc.items()}
    max_len = max(lens.values())
    avg_len = statistics.mean(lens.values())
    longest = [tid for tid, l in lens.items() if l == max_len]


    def tok_repr(b: bytes, limit=20):
        try:
            s = b.decode("utf-8")
            if s.isprintable() and not s.isspace():
                if len(s) > limit:
                    s = s[:limit] + "…"
                return repr(s)
        except UnicodeDecodeError:
            pass
        hexs = binascii.hexlify(b).decode()
        return "0x" + (hexs if len(b) <= limit else hexs[: limit * 2] + "…")


    # 随机抽样 10 个 token（排除特殊符号）
    ids = list(voc.keys())
    random.shuffle(ids)
    sample_ids = ids[:10]

    # ---------- 3. 打印 / 写 Markdown ----------
    lines = (
        [
            "# BPE 模型报告",
            f"- 词表大小：**{len(voc)}**",
            f"- 合并步数：**{len(mer)}**",
            f"- 最长 token 长度：`{max_len}` bytes",
            f"- 平均 token 长度：`{avg_len:.2f}` bytes",
            "",
            "## 最长 token 列表",
        ]
        + [f"- ID {tid}: {tok_repr(voc[tid])}" for tid in longest]
        + [
            "",
            "## 随机抽样 10 个 token",
        ]
        + [f"- ID {tid} (len={lens[tid]}): {tok_repr(voc[tid])}" for tid in sample_ids]
    )

    report_path = pathlib.Path("bpe_report.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"✔️  报告已写入 {report_path.resolve()}")


