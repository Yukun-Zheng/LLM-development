from astra_codex.tokenizer import ByteBPETokenizer, ByteTokenizer


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "深圳 AI + robot 🤖"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text


def test_bpe_roundtrip_and_compression() -> None:
    corpus = ["banana banana banana", "bandana banana"]
    tokenizer = ByteBPETokenizer.train(corpus, vocab_size=280, min_pair_frequency=2)
    text = "banana banana"
    byte_length = len(text.encode("utf-8"))
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(ids) < byte_length
