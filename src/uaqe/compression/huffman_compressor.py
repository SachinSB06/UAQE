"""Huffman-coding ``ICompressionStrategy`` implementation
(``CompressionType.HUFFMAN``).

``HuffmanCompressor`` losslessly compresses every layer parameter's raw
byte buffer using a per-tensor Huffman code: byte values are ranked by
frequency and assigned variable-length bit codes (shorter codes for
more frequent bytes), which is most effective on tensors with a skewed
byte-value distribution — quantized or clustered tensors, in
particular. Like :class:`~uaqe.compression.rle_compressor.RLECompressor`,
it operates generically on any ``IMRTensor`` regardless of ``dtype``,
treating ``data`` as an opaque byte stream.

The generated code table is not recoverable from the bitstream alone,
so this module writes a self-contained header ahead of the packed bits
recording the original element count and every ``(byte_value,
code_length, code_bits)`` triple needed to rebuild the decoding tree,
so that ``tensor.dtype`` plus ``tensor.data`` alone are sufficient to
reverse the transform.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Dict, List, Tuple

from uaqe.common.exceptions import CompressionError
from uaqe.common.imr import IMR, IMRTensor
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.value_objects import CompressionConfig

#: This strategy's unique registration name, returned by
#: :meth:`HuffmanCompressor.name`.
STRATEGY_NAME = "huffman"

#: Dtype suffix appended to a tensor's original dtype to produce the
#: dtype recorded on the Huffman-encoded ``IMRTensor``.
HUFFMAN_DTYPE_SUFFIX = "_huffman"

#: Byte width of each little-endian header count field (original byte
#: length, and the number of codebook entries).
_HEADER_FIELD_BYTE_WIDTH = 4


class HuffmanCompressor(ICompressionStrategy):
    """Losslessly Huffman-encodes every parameter tensor of an ``IMR``.

    Attributes:
        _logger: Structured logging sink.
    """

    def __init__(self, logger: ILogger) -> None:
        """Initialize the strategy.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
        """
        self._logger = logger

    def apply(self, imr: IMR, plan: CompressionConfig) -> IMR:
        """Huffman-encode every parameter tensor's raw bytes.

        Args:
            imr: The model to compress.
            plan: The compression configuration. Not consulted by this
                strategy (Huffman coding has no tunable target ratio of
                its own), accepted only to satisfy the
                ``ICompressionStrategy`` contract.

        Returns:
            A new ``IMR`` with every parameter tensor's ``data``
            replaced by its Huffman-encoded form (self-contained header
            plus packed bitstream) and ``dtype`` suffixed with
            :data:`HUFFMAN_DTYPE_SUFFIX`. Already-Huffman-encoded
            tensors are left unchanged rather than double-encoded, and
            an empty tensor is left unchanged (nothing to encode). The
            input ``imr`` is not modified.
        """
        del plan  # Huffman coding has no per-run tunable parameters.
        original_total_bytes = 0
        encoded_total_bytes = 0
        new_layers = []
        for layer in imr.layers:
            new_parameters = {}
            for param_name, tensor in layer.parameters.items():
                if tensor.dtype.endswith(HUFFMAN_DTYPE_SUFFIX) or not tensor.data:
                    new_parameters[param_name] = tensor
                    continue
                encoded_tensor = self._encode_tensor(tensor)
                new_parameters[param_name] = encoded_tensor
                original_total_bytes += len(tensor.data)
                encoded_total_bytes += len(encoded_tensor.data)
            new_layers.append(dataclasses.replace(layer, parameters=new_parameters))

        self._logger.info(
            "Applied Huffman encoding.",
            strategy=self.name(),
            original_bytes=original_total_bytes,
            encoded_bytes=encoded_total_bytes,
        )
        return dataclasses.replace(imr, layers=new_layers)

    def name(self) -> str:
        """Return this strategy's unique registration name."""
        return STRATEGY_NAME

    def _encode_tensor(self, tensor: IMRTensor) -> IMRTensor:
        """Huffman-encode one tensor's raw byte buffer.

        Args:
            tensor: The tensor to encode. Must have non-empty ``data``.

        Returns:
            A new ``IMRTensor`` with the same ``shape``, ``dtype``
            suffixed with :data:`HUFFMAN_DTYPE_SUFFIX`, and
            header-plus-bitstream ``data``.
        """
        code_by_byte = _build_huffman_codes(tensor.data)
        encoded = _pack_header(len(tensor.data), code_by_byte) + _pack_bits(
            tensor.data, code_by_byte
        )
        return IMRTensor(
            shape=tensor.shape, dtype=tensor.dtype + HUFFMAN_DTYPE_SUFFIX, data=encoded
        )

    def decode_tensor(self, tensor: IMRTensor) -> IMRTensor:
        """Reconstruct a tensor's original bytes from its Huffman
        encoding.

        Used by evaluation/benchmarking stages (and by tests validating
        this strategy's round-trip correctness) rather than by
        ``apply`` itself, which only moves forward.

        Args:
            tensor: A tensor previously produced by :meth:`apply`
                (``dtype`` ends with :data:`HUFFMAN_DTYPE_SUFFIX`).

        Returns:
            A new ``IMRTensor`` with the Huffman suffix stripped from
            ``dtype`` and ``data`` restored to its original bytes.

        Raises:
            CompressionError: If ``tensor.dtype`` does not end with
                :data:`HUFFMAN_DTYPE_SUFFIX`.
        """
        if not tensor.dtype.endswith(HUFFMAN_DTYPE_SUFFIX):
            raise CompressionError(
                f"Cannot decode tensor of dtype {tensor.dtype!r}; "
                f"expected a dtype ending in {HUFFMAN_DTYPE_SUFFIX!r}.",
                code="COMPRESS_HUFFMAN_UNSUPPORTED_DTYPE",
            )
        original_dtype = tensor.dtype[: -len(HUFFMAN_DTYPE_SUFFIX)]
        original_length, code_by_byte, bitstream_offset = _unpack_header(tensor.data)
        decoded = _unpack_bits(
            tensor.data[bitstream_offset:], code_by_byte, original_length
        )
        return IMRTensor(shape=tensor.shape, dtype=original_dtype, data=decoded)


def _build_huffman_codes(data: bytes) -> Dict[int, str]:
    """Build a canonical-order Huffman code for every distinct byte in
    ``data``.

    Args:
        data: The raw bytes to build a code table for. Must be
            non-empty.

    Returns:
        A mapping of byte value to its code, expressed as a string of
        ``"0"``/``"1"`` characters (most-significant bit first). A
        single-distinct-byte input is assigned the one-bit code
        ``"0"``, since Huffman coding is undefined for a zero-entropy
        alphabet but a fixed one-bit-per-symbol code still round-trips
        correctly.
    """
    frequencies: Dict[int, int] = {}
    for byte in data:
        frequencies[byte] = frequencies.get(byte, 0) + 1

    if len(frequencies) == 1:
        (only_byte,) = frequencies.keys()
        return {only_byte: "0"}

    # Min-heap of (frequency, insertion_order, node), where a node is
    # either a raw byte value (leaf) or a (left, right) tuple of
    # sub-nodes; insertion_order breaks ties deterministically.
    heap: List[Tuple[int, int, object]] = [
        (freq, order, byte) for order, (byte, freq) in enumerate(frequencies.items())
    ]
    heapq.heapify(heap)
    next_order = len(heap)
    while len(heap) > 1:
        freq_a, _, node_a = heapq.heappop(heap)
        freq_b, _, node_b = heapq.heappop(heap)
        heapq.heappush(heap, (freq_a + freq_b, next_order, (node_a, node_b)))
        next_order += 1

    _, _, root = heap[0]
    codes: Dict[int, str] = {}
    _assign_codes(root, "", codes)
    return codes


def _assign_codes(node: object, prefix: str, codes: Dict[int, str]) -> None:
    """Recursively assign bit-string codes to every leaf under ``node``.

    Args:
        node: Either a raw byte value (leaf) or a ``(left, right)``
            tuple of sub-nodes.
        prefix: The bit-string path taken to reach ``node`` so far.
        codes: The mapping being populated, keyed by byte value.
    """
    if isinstance(node, tuple):
        left, right = node
        _assign_codes(left, prefix + "0", codes)
        _assign_codes(right, prefix + "1", codes)
    else:
        codes[node] = prefix


def _pack_header(original_length: int, code_by_byte: Dict[int, str]) -> bytes:
    """Serialize the self-contained header preceding a Huffman
    bitstream.

    Layout::

        [original_length: uint32 little-endian]
        [entry_count: uint32 little-endian]
        entry_count x [byte_value: uint8][code_length: uint8][code_bits...]

    Each entry's ``code_bits`` are packed MSB-first into
    ``ceil(code_length / 8)`` bytes.

    Args:
        original_length: The decoded byte length to record.
        code_by_byte: The code table to serialize.

    Returns:
        The serialized header bytes.
    """
    header = bytearray()
    header += original_length.to_bytes(_HEADER_FIELD_BYTE_WIDTH, "little")
    header += len(code_by_byte).to_bytes(_HEADER_FIELD_BYTE_WIDTH, "little")
    for byte_value, code in code_by_byte.items():
        header.append(byte_value)
        header.append(len(code))
        header += _bits_to_bytes(code)
    return bytes(header)


def _unpack_header(data: bytes) -> Tuple[int, Dict[int, str], int]:
    """Parse the header written by :func:`_pack_header`.

    Args:
        data: The full encoded tensor buffer, header first.

    Returns:
        A tuple of ``(original_length, code_by_byte, bitstream_offset)``
        — the decoded byte length, the recovered code table, and the
        byte offset in ``data`` at which the packed bitstream begins.

    Raises:
        CompressionError: If ``data`` is shorter than its declared
            header fields.
    """
    if len(data) < 2 * _HEADER_FIELD_BYTE_WIDTH:
        raise CompressionError(
            "Huffman-encoded buffer is shorter than its fixed header.",
            code="COMPRESS_HUFFMAN_MALFORMED_BUFFER",
        )
    original_length = int.from_bytes(data[:_HEADER_FIELD_BYTE_WIDTH], "little")
    entry_count = int.from_bytes(
        data[_HEADER_FIELD_BYTE_WIDTH : 2 * _HEADER_FIELD_BYTE_WIDTH], "little"
    )

    offset = 2 * _HEADER_FIELD_BYTE_WIDTH
    code_by_byte: Dict[int, str] = {}
    for _ in range(entry_count):
        if offset + 2 > len(data):
            raise CompressionError(
                "Huffman-encoded buffer's codebook section is truncated.",
                code="COMPRESS_HUFFMAN_MALFORMED_BUFFER",
            )
        byte_value = data[offset]
        code_length = data[offset + 1]
        offset += 2
        code_byte_width = (code_length + 7) // 8
        code_bytes = data[offset : offset + code_byte_width]
        offset += code_byte_width
        code_by_byte[byte_value] = _bytes_to_bits(code_bytes, code_length)

    return original_length, code_by_byte, offset


def _pack_bits(data: bytes, code_by_byte: Dict[int, str]) -> bytes:
    """Encode ``data`` as a packed bitstream using ``code_by_byte``.

    Args:
        data: The raw bytes to encode.
        code_by_byte: The Huffman code for every distinct byte in
            ``data``.

    Returns:
        The packed bitstream, MSB-first, zero-padded to a whole number
        of bytes.
    """
    bits = "".join(code_by_byte[byte] for byte in data)
    return _bits_to_bytes(bits)


def _unpack_bits(
    packed: bytes, code_by_byte: Dict[int, str], original_length: int
) -> bytes:
    """Decode a packed Huffman bitstream back into raw bytes.

    Args:
        packed: The packed, zero-padded bitstream produced by
            :func:`_pack_bits`.
        code_by_byte: The code table the bitstream was encoded with.
        original_length: The exact number of original bytes to decode;
            bounds decoding so that any zero-padding at the end of
            ``packed`` is not misread as additional symbols.

    Returns:
        The decoded, original byte string.

    Raises:
        CompressionError: If the bitstream is exhausted before
            ``original_length`` bytes have been decoded, or contains a
            bit sequence not present in ``code_by_byte``.
    """
    code_to_byte = {code: byte for byte, code in code_by_byte.items()}
    bits = _bytes_to_bits(packed, len(packed) * 8)

    decoded = bytearray()
    current_code = ""
    bit_index = 0
    while len(decoded) < original_length:
        if bit_index >= len(bits):
            raise CompressionError(
                "Huffman bitstream exhausted before decoding the "
                "expected number of bytes.",
                code="COMPRESS_HUFFMAN_MALFORMED_BUFFER",
            )
        current_code += bits[bit_index]
        bit_index += 1
        if current_code in code_to_byte:
            decoded.append(code_to_byte[current_code])
            current_code = ""

    return bytes(decoded)


def _bits_to_bytes(bits: str) -> bytes:
    """Pack a ``"0"``/``"1"`` bit string MSB-first into bytes, zero-
    padding the final byte if needed."""
    padded = bits + "0" * ((8 - len(bits) % 8) % 8)
    return bytes(
        int(padded[i : i + 8], 2) for i in range(0, len(padded), 8)
    )


def _bytes_to_bits(data: bytes, bit_count: int) -> str:
    """Unpack bytes MSB-first into a ``"0"``/``"1"`` bit string,
    truncated to exactly ``bit_count`` bits."""
    bits = "".join(format(byte, "08b") for byte in data)
    return bits[:bit_count]
